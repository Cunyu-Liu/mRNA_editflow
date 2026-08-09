"""A0--A9 unconditional hard-disable for Route-A V3 sealed execution.

This module is the only local project module imported by the runner before its
command-line mode is known.  A0 intentionally contains no latent authorization
implementation: A9 must replace this guard under a separately reviewed,
hash-frozen contract before any sealed-final invocation can proceed.
"""
from __future__ import annotations


HARD_DISABLED = "ROUTE_A_V3_SEALED_HARD_DISABLED_A0_A9"


class RouteAV3SealedHardDisabled(RuntimeError):
    """Raised before any call argument, path, runtime module, or state access."""


def assert_sealed_final_authorized(
    call_args: object = None,
    repo_root: object = None,
) -> None:
    """Unconditionally reject sealed-final throughout A0--A9.

    The arguments are intentionally not inspected.  In particular, this
    function performs no config, authorization, readiness, manifest, dataset,
    checkpoint, restricted-store, output, GPU, or access-state I/O.
    """
    raise RouteAV3SealedHardDisabled(HARD_DISABLED)
