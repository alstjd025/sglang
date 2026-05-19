"""halo_request_tracker — the single per-request state store.

`RequestTracker` holds one `RequestRecord` per in-flight (or recently
finished) request and is the *sole writer* of that state. The admission
gate, admission policies, and metrics are read-only consumers.

See managers/halo/CLAUDE.md for the request-level design.
"""

from sglang.srt.managers.halo.request_tracker.record import (
    RequestRecord,
    RequestState,
    ServerStateSnapshot,
)
from sglang.srt.managers.halo.request_tracker.tracker import RequestTracker

__all__ = [
    "RequestRecord",
    "RequestState",
    "ServerStateSnapshot",
    "RequestTracker",
]
