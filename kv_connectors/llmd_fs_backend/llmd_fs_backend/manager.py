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

import os
from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
from zmq import ZMQError

from llmd_fs_backend.event_publisher import StorageMedium
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        extra_config: dict | None = None,
        event_publisher=None,
    ) -> None:
        self.file_mapper: FileMapper = file_mapper
        self._event_publisher = (
            event_publisher
            if event_publisher is not None
            else self._create_event_publisher(
                self.file_mapper.model_name, extra_config or {}
            )
        )

    @staticmethod
    def _create_event_publisher(model_name: str, extra_config: dict):
        """Create a StorageEventPublisher if events are enabled in *extra_config*."""
        if not extra_config.get("enable_events", False):
            return None

        endpoint = extra_config.get("storage_events_endpoint")
        if not endpoint:
            return None

        kwargs = {}
        if "storage_medium" in extra_config:
            kwargs["medium"] = StorageMedium(extra_config["storage_medium"])
        if "storage_events_hwm" in extra_config:
            kwargs["sndhwm"] = int(extra_config["storage_events_hwm"])

        try:
            from llmd_fs_backend.event_publisher import StorageEventPublisher

            return StorageEventPublisher(
                endpoint=endpoint,
                model_name=model_name,
                **kwargs,
            )
        except ZMQError:
            logger.warning(
                "failed to create storage event publisher for %s",
                endpoint,
                exc_info=True,
            )
            return None

    def _publish_blocks_stored(self, keys: Collection[OffloadKey]) -> None:
        if self._event_publisher is None:
            return
        try:
            block_hashes = [get_offload_block_hash(k) for k in keys]
            self._event_publisher.publish_blocks_stored(block_hashes)
        except Exception:
            logger.warning("failed to publish storage event", exc_info=True)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # We don't have per-request offload policy — every request uses the
        # same storage tier with the same rules. Return defaults.
        return RequestOffloadingContext()

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready to be read.
        """
        file_path = self.file_mapper.get_file_name(key)
        return os.path.exists(file_path)

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        For shared storage, loading is stateless - return specs that point to files.
        """
        return SharedStorageLoadStoreSpec(keys)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Update access times if desired.
        Shared storage version does nothing here because updates are handled
        by the file thread for performance reasons.
        """
        pass

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Stateless load - no post-load action needed."""
        pass

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare storing new blocks.
        Shared storage always accepts new blocks. Eviction is not needed.
        If a file already exists, the file thread handles it.
        """
        keys_to_store = list(keys)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(keys_to_store)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],  # no eviction needed
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        For shared storage, storing is stateless but we emit events for stored blocks.
        """
        if success:
            self._publish_blocks_stored(keys)

    def get_stats(self) -> "OffloadingConnectorStats | None":
        """Return FS-backend-specific connector stats since the last call.

        Compatibility note
        ~~~~~~~~~~~~~~~~~~
        get_stats() exists on vLLM's OffloadingManager base class but returns
        None by default.  When running against vLLM v0.22.0, this method
        overrides the base class default.  The scheduler calls it every step,
        but since we return None the scheduler skips FS-backend-specific stats
        (generic transfer metrics are still collected automatically by vLLM's
        worker).

        This is the counterpart to
        SharedStorageOffloadingSpec.build_metric_definitions().
        To add FS-specific metrics:

        1. Declare the metrics in build_metric_definitions() (spec.py)
        2. Accumulate data in the manager (e.g. lookup hit/miss counters)
        3. Return the accumulated values here as OffloadingConnectorStats

        Example::

            stats = OffloadingConnectorStats()
            stats.increase_counter("vllm:kv_offload_fs_lookup_hit", self._lookup_hits)
            self._lookup_hits = 0
            return stats

        Currently returns None because this connector has no FS-specific
        metrics.  Generic transfer metrics (load_bytes, store_bytes, etc.)
        are automatically collected from TransferResult by the
        OffloadingConnectorWorker and reported by the scheduler — they do
        NOT go through this method.

        See vllm-project/vllm#44008 and vllm-project/vllm#35669.
        """
        # Currently no FS-backend-specific stats to report.
        # Returning None lets the scheduler skip this manager in get_stats().
        return None

    def shutdown(self) -> None:
        if self._event_publisher is not None:
            self._event_publisher.close()
