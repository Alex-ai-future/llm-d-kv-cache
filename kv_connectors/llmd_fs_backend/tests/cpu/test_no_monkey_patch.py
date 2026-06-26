# Copyright 2026 The llm-d Authors.
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

"""
Prove that the OffloadPromMetrics monkey-patch is no longer needed.

vLLM PR #40010 (merged as 4b7f5ea1a) fixed the duplicate Prometheus
timeseries issue in MultiConnector.build_prom_metrics() via a
seen_classes dedup mechanism. Since llmd_fs_backend pins vllm==0.22.0,
the monkey-patch (previously in llmd_fs_backend/metrics.py) is obsolete.
"""

import pytest

pytestmark = pytest.mark.no_cuda_required


def test_multi_connector_prom_metrics_dedup():
    """
    Verify MultiConnector.build_prom_metrics() uses seen_classes dedup
    so same connector class only registers Prometheus metrics once.

    This is the upstream fix from vLLM PR #40010 that makes the
    llmd_fs_backend monkey-patch obsolete.
    """
    vllm = pytest.importorskip("vllm")

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    import inspect

    src = inspect.getsource(MultiConnector.build_prom_metrics)

    assert "seen_classes" in src, (
        "MultiConnector.build_prom_metrics() should have seen_classes dedup "
        "from vLLM PR #40010"
    )
    assert "if connector_cls in seen_classes" in src, (
        "seen_classes should skip already-registered connector classes"
    )
    assert "seen_classes.add(connector_cls)" in src, (
        "seen_classes should track each connector class after first registration"
    )
