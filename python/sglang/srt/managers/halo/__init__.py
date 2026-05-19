"""Project Halo — request-level admission control + tracking for SGLang.

Halo is the umbrella name for the request-level admission/tracking
feature. Two components live under this package:

  - request_tracker/    — RequestTracker: the single per-request state store.
  - admission_control/  — the policy-pluggable admission gate (mooncake /
                          vss / reactive policies).

The scheduler owns one HaloController. See managers/halo/CLAUDE.md.
"""

from sglang.srt.managers.halo.controller import (
    REASON_DISABLED,
    HaloConfig,
    HaloController,
    HaloRejectError,
    build_halo_controller_from_server_args,
)
from sglang.srt.managers.halo.cost_model_sampler import (
    HaloCostModelSampler,
    build_halo_cost_sampler_from_server_args,
)
from sglang.srt.managers.halo.metrics import HaloMetrics
from sglang.srt.managers.halo.request_tracker import (
    RequestRecord,
    RequestState,
    RequestTracker,
    ServerStateSnapshot,
)

__all__ = [
    "REASON_DISABLED",
    "HaloConfig",
    "HaloController",
    "HaloCostModelSampler",
    "HaloMetrics",
    "HaloRejectError",
    "RequestRecord",
    "RequestState",
    "RequestTracker",
    "ServerStateSnapshot",
    "build_halo_controller_from_server_args",
    "build_halo_cost_sampler_from_server_args",
]
