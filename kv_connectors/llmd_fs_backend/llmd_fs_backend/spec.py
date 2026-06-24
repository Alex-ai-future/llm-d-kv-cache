# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.worker import (
    DEFAULT_MAX_STAGING_MEMORY_GB,
    DEFAULT_MAX_WRITE_QUEUED_SECONDS,
    DEFAULT_READ_PREFERRING_WORKERS_RATIO,
    DEFAULT_THREADS_PER_GPU,
    StorageOffloadingHandlers,
)

DEFAULT_STORAGE_BLOCK_SIZE = 256


class SharedStorageOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Return Prometheus metric definitions for FS-backend-specific metrics.

        Compatibility note
        ~~~~~~~~~~~~~~~~~~
        build_metric_definitions() was added to vLLM's OffloadingSpec base class
        after v0.22.0 (in PR #35669).  When running against vLLM v0.22.0, this
        method exists but is never called by OffloadPromMetrics — it is a no-op
        placeholder that becomes functional once the vLLM dependency is upgraded.

        Architecture
        ~~~~~~~~~~~~
        Generic transfer metrics (load_bytes, store_bytes, load_time, store_time,
        load_size, store_size) are already declared by vLLM's
        get_connector_metric_definitions() and automatically collected from
        TransferResult by the OffloadingConnectorWorker.  This connector does NOT
        re-declare them — doing so would be redundant and gets overwritten by
        vLLM's definitions during OffloadPromMetrics initialisation:

            self._offloading_metric_metadata = {
                **spec_cls.build_metric_definitions(extra_config),
                **get_connector_metric_definitions(),  # overwrites same-name keys
            }

        To add FS-specific metrics (e.g. lookup_hit/miss counters, file existence
        check stats, GDS mode metrics), add entries here AND implement
        SharedStorageOffloadingManager.get_stats() to return the actual values
        via OffloadingConnectorStats.

        See vllm-project/vllm#44008 (KV Offloading Metrics Redesign) and
        vllm-project/vllm#35669 (Offloading Manager Stats) for the full
        metrics architecture.
        """
        # Currently no FS-backend-specific metrics.
        # Generic transfer metrics are handled by vLLM automatically.
        # Future FS-specific metrics should be added here.
        return {}

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        # Hide "block_size" from the base class to bypass the uniformity
        # assertion on hybrid models (we derive the factor ourselves below).
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        hidden_block_size = extra_config.pop("block_size", None)
        try:
            super().__init__(vllm_config, kv_cache_config)
        finally:
            if hidden_block_size is not None:
                extra_config["block_size"] = hidden_block_size

        self._manager: OffloadingManager | None = None
        # worker-side
        self._handlers: StorageOffloadingHandlers | None = None

        self.threads_per_gpu = int(
            self.extra_config.get("threads_per_gpu", DEFAULT_THREADS_PER_GPU)
        )
        shared_storage_path = self.extra_config.get(
            "shared_storage_path", "/tmp/shared-kv"
        )
        self.max_staging_memory_gb = int(
            self.extra_config.get(
                "max_staging_memory_gb", DEFAULT_MAX_STAGING_MEMORY_GB
            )
        )  # Max staging CPU buffer in GB

        self.offloaded_block_size = int(
            self.extra_config.get("block_size", DEFAULT_STORAGE_BLOCK_SIZE)
        )

        # hash_block_size = GCD of all groups' block sizes (the granularity at
        # which Request.block_hashes are computed); use it instead of
        # cache_config.block_size which can be larger on hybrid models (e.g. DSv4).
        assert self.offloaded_block_size % self.hash_block_size == 0, (
            "offloaded_block_size must be a multiple of hash_block_size"
        )
        self.gpu_blocks_per_file = self.offloaded_block_size // self.hash_block_size

        # Derive block_size_factor from file layout instead of base class.
        self.block_size_factor = self.gpu_blocks_per_file

        self.read_preferring_ratio = float(
            self.extra_config.get(
                "read_preferring_ratio", DEFAULT_READ_PREFERRING_WORKERS_RATIO
            )
        )
        self.max_write_queued_seconds = float(
            self.extra_config.get(
                "max_write_queued_seconds", DEFAULT_MAX_WRITE_QUEUED_SECONDS
            )
        )

        parallel_config = vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        assert parallel_config.world_size == tp_size * pp_size * pcp_size

        self.file_mapper = FileMapper.from_vllm_config(
            root_dir=shared_storage_path,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            gpu_blocks_per_file=self.gpu_blocks_per_file,
        )
        self.file_mapper.write_run_config()

    def get_manager(self) -> OffloadingManager:
        assert self.vllm_config.parallel_config.rank == 0, "Scheduler rank should be 0"
        if not self._manager:
            backend = self.extra_config.get("backend", "POSIX")
            if backend == "OBJ":
                from llmd_nixl.manager import NixlStorageOffloadingManager

                self.extra_config.setdefault("storage_medium", "OBJECT_STORE")
                self._manager = NixlStorageOffloadingManager(
                    file_mapper=self.file_mapper,
                    extra_config=self.extra_config,
                )
            else:
                self.extra_config.setdefault("storage_medium", "SHARED_STORAGE")
                self._manager = SharedStorageOffloadingManager(
                    file_mapper=self.file_mapper,
                    extra_config=self.extra_config,
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            backend = self.extra_config.get("backend", "POSIX")
            if backend == "OBJ":
                from llmd_nixl.worker import NixlStorageOffloadingHandlers

                handlers_cls = NixlStorageOffloadingHandlers
            else:
                handlers_cls = StorageOffloadingHandlers
            self._handlers = handlers_cls(
                file_mapper=self.file_mapper,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                gpu_block_size=self.hash_block_size,
                kv_caches=kv_caches,
                threads_per_gpu=self.threads_per_gpu,
                max_staging_memory_gb=self.max_staging_memory_gb,
                read_preferring_ratio=self.read_preferring_ratio,
                max_write_queued_seconds=self.max_write_queued_seconds,
                extra_config=self.extra_config,
            )

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            SharedStorageLoadStoreSpec,
            self._handlers.gpu_to_storage_handler,
        )
        yield (
            SharedStorageLoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.storage_to_gpu_handler,
        )
