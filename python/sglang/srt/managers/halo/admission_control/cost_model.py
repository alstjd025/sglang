"""Cost models for admission control AND for Halo step-latency prediction.

See managers/halo/admission_control/CLAUDE.md for the admission-control design and
ms_dev/halo_dev/prediction_model.md for the Halo step model.

Three coexisting models live in this file:

1. PrefillCostModel   (legacy, Mooncake-like, per-request)
   T_prefill_ms ≈ α·d² + β·d + γ + δ·p     where d = max(0, n − p)

2. TBTCostModel       (legacy, Mooncake-like, per-step approximation)
   T_tbt_ms ≈ a + b·batch_size + c·per_req_kv

3. HaloStepCostModel  (new, post-Phase-1 follow-up — see §18 cliff)
   T_step_ms ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
             + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c
   Caller decides what Σ contains:
     - solo mode (Halo R1 slowdown denominator): batch of 1
     - batched mode (admission stage 2 / R3 lookahead): real or hypothetical batch

The Phase-1 split between PrefillCostModel/TBTCostModel is preserved verbatim;
HaloStepCostModel coexists and is selected via a separate CLI flag.

All three load coefficients from a JSON file. Malformed / missing files raise a
typed CostModelLoadError that loaders catch and downgrade to "stage disabled"
rather than fatal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CostModelLoadError(Exception):
    """Raised when a cost-model JSON cannot be loaded or validated."""


@dataclass(frozen=True)
class PrefillCostModel:
    """Estimate prefill latency from prompt length and prefix-cache match length.

    Form:    T_ms ≈ α·d² + β·d + γ + δ·p           where d = max(0, n - p)

    - α, β capture the actual prefill compute (Mooncake Eq. 1's quadratic-in-
      prompt-length term collapsed to fit coefficients per
      (model, hardware, kernel config)).
    - γ is the fixed per-request overhead.
    - δ captures the cost of loading the matched prefix from radix cache —
      empirically ~7 µs/token on B200×4 with Llama-3.3-70B. Older models
      that omit `delta` from JSON load with δ=0 (backward-compatible).
    """

    alpha: float
    beta: float
    gamma: float
    delta: float = 0.0
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def estimate_ms(self, prompt_len: int, prefix_len: int) -> float:
        d = max(0, prompt_len - prefix_len)
        return (
            self.alpha * d * d
            + self.beta * d
            + self.gamma
            + self.delta * prefix_len
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PrefillCostModel":
        data = _read_json(path, "prefill cost model")
        try:
            return cls(
                alpha=float(data["alpha"]),
                beta=float(data["beta"]),
                gamma=float(data["gamma"]),
                delta=float(data.get("delta", 0.0)),  # back-compat: 0 if absent
                metadata=data.get("fit_metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CostModelLoadError(
                f"Invalid prefill cost model at {path}: missing/non-numeric "
                f"alpha/beta/gamma ({e})"
            ) from e

    def to_json(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "fit_metadata": self.metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2))


@dataclass(frozen=True)
class TBTCostModel:
    """Estimate per-step decode latency (TBT) given current batch composition.

    Linear approximation: TBT grows with batch size (compute-side overhead)
    and per-request KV span (attention memory bandwidth per step). The
    caller is responsible for computing `per_req_kv = total_kv // batch_size`.
    Higher-order terms are absorbed by re-fitting when hardware/config changes.
    """

    a: float
    b: float
    c: float
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def estimate_ms(self, batch_size: int, per_req_kv: int) -> float:
        return self.a + self.b * batch_size + self.c * per_req_kv

    @classmethod
    def from_json(cls, path: str | Path) -> "TBTCostModel":
        data = _read_json(path, "TBT cost model")
        try:
            return cls(
                a=float(data["a"]),
                b=float(data["b"]),
                c=float(data["c"]),
                metadata=data.get("fit_metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CostModelLoadError(
                f"Invalid TBT cost model at {path}: missing/non-numeric "
                f"a/b/c ({e})"
            ) from e

    def to_json(self, path: str | Path) -> None:
        payload = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "fit_metadata": self.metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2))


def _read_json(path: str | Path, what: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CostModelLoadError(f"{what} JSON not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise CostModelLoadError(f"{what} JSON malformed at {p}: {e}") from e
    if not isinstance(data, dict):
        raise CostModelLoadError(f"{what} JSON at {p} must be an object")
    return data


def try_load_prefill_cost_model(
    path: Optional[str],
) -> Optional[PrefillCostModel]:
    """Lenient loader: returns None and logs WARN on failure (controller path)."""
    if not path:
        return None
    try:
        model = PrefillCostModel.from_json(path)
    except CostModelLoadError as e:
        logger.warning(
            "admission control: prefill cost model disabled — %s", e
        )
        return None
    logger.info(
        "admission control: prefill cost model loaded from %s "
        "(alpha=%.4g beta=%.4g gamma=%.4g)",
        path, model.alpha, model.beta, model.gamma,
    )
    return model


def try_load_tbt_cost_model(path: Optional[str]) -> Optional[TBTCostModel]:
    """Lenient loader: returns None and logs WARN on failure (controller path)."""
    if not path:
        return None
    try:
        model = TBTCostModel.from_json(path)
    except CostModelLoadError as e:
        logger.warning("admission control: TBT cost model disabled — %s", e)
        return None
    logger.info(
        "admission control: TBT cost model loaded from %s "
        "(a=%.4g b=%.4g c=%.4g)",
        path, model.a, model.b, model.c,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Halo Step Cost Model (post-Phase-1 follow-up)
# Full design + rationale: ms_dev/halo_dev/prediction_model.md
# Validation + split-mode trade-offs: prediction_model.md §15–§17
# ─────────────────────────────────────────────────────────────────────────────

# Unified form — single shared constant θ_c, prefill + decode terms fit jointly.
HALO_STEP_FORM_V1 = "halo_step_v1"

# Split form — prefill and decode fit independently with their own constants
# (θ_c_p for prefill steps, θ_c_d for decode steps). Same 5 slope coefficients.
# See prediction_model.md §17 for when to use which.
HALO_STEP_FORM_SPLIT_V1 = "halo_step_split_v1"

HALO_STEP_FORMS = (HALO_STEP_FORM_V1, HALO_STEP_FORM_SPLIT_V1)


@dataclass(frozen=True)
class HaloStepCostModel:
    """Estimate one forward-step latency from batch composition.

    Two `form`s (selected at JSON-load time):

    1. ``halo_step_v1`` (unified) — 6 coefficients:
        T_step_ms ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
                  + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c

       Prefill and decode terms share the same constant θ_c. Single OLS fit
       over all (EXTEND + DECODE + MIXED) steps.

    2. ``halo_step_split_v1`` (split) — 7 coefficients:
        T_prefill_step ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ + θ_c_p
        T_decode_step  ≈ θ_d1·Σrⱼ  + θ_d2·bs_d + θ_c_d

       Each side's intercept is fit independently from its own step type.
       Justified when MIXED steps are absent (no `--enable-mixed-chunk`), so
       prefill and decode never share a single forward pass — see
       prediction_model.md §17 (validation showed θ_d2 sign flip from −1.10 to
       +0.18 under this regime, matching MuxWise's original separated form).

    Caller decides what goes into Σ. The same instance handles both:
        - Halo R1 slowdown denominator → estimate_solo_prefill_total_ms /
          estimate_solo_tbt_ms (single-request "solo" shortcut).
        - Admission stage-2 / Phase-2 R3 lookahead → estimate_step_ms with the
          actual or hypothetical batch composition.

    Backward-compatible: callers that ignore `form` and read theta_c get the
    unified-mode constant. In split mode, theta_c falls back to whichever
    constant best mimics the unified semantics (we use θ_c_d since most steps
    are DECODE in practice; never expected to be hit in the split code path).
    """

    theta_p1: float  # Σnᵢ²   (self-attention on new tokens)
    theta_p2: float  # Σ(nᵢrᵢ) (cross-attention: new tokens × cached prefix)
    theta_p3: float  # Σnᵢ    (per-token MLP/FFN)
    theta_d1: float  # Σrⱼ    (decode attention memory BW — batch KV sum)
    theta_d2: float  # bs_d   (per-request decode overhead)
    theta_c: float   # constant (unified mode: shared; split mode: see above)
    # Split-mode-only fields. None ⇒ unified form.
    theta_c_p: Optional[float] = None  # prefill-step intercept (split form)
    theta_c_d: Optional[float] = None  # decode-step intercept (split form)
    form: str = HALO_STEP_FORM_V1
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    # ---- helpers ----

    @property
    def is_split(self) -> bool:
        return self.form == HALO_STEP_FORM_SPLIT_V1

    def _prefill_const(self) -> float:
        if self.is_split and self.theta_c_p is not None:
            return self.theta_c_p
        return self.theta_c

    def _decode_const(self) -> float:
        if self.is_split and self.theta_c_d is not None:
            return self.theta_c_d
        return self.theta_c

    # ---- estimation ----

    def estimate_step_ms(
        self,
        prefill_infos: Iterable[Tuple[int, int]],
        decode_infos: Iterable[int],
    ) -> float:
        """Batched mode: full Σ over the supplied batch composition.

        In unified mode this is a single linear sum.
        In split mode, prefill-only steps use θ_c_p, decode-only steps use
        θ_c_d, and rare mixed inputs (shouldn't happen when MIXED is disabled)
        get the sum of both terms — preserving each side's slope contributions
        but adding both intercepts (over-estimates by one constant; acceptable
        since split mode assumes mixed steps don't occur).
        """
        sum_n_sq = 0.0
        sum_nr = 0.0
        sum_n = 0.0
        any_prefill = False
        for n, r in prefill_infos:
            sum_n_sq += n * n
            sum_nr += n * r
            sum_n += n
            any_prefill = True
        sum_r = 0.0
        bs_d = 0
        for r in decode_infos:
            sum_r += r
            bs_d += 1
        any_decode = bs_d > 0

        prefill_part = (
            self.theta_p1 * sum_n_sq
            + self.theta_p2 * sum_nr
            + self.theta_p3 * sum_n
        )
        decode_part = self.theta_d1 * sum_r + self.theta_d2 * bs_d

        if self.is_split:
            # Pick the intercept(s) matching what the step actually contains.
            if any_prefill and any_decode:
                # Mixed (rare in split-mode workloads) — sum of both
                # intercepts, see docstring caveat.
                const = self._prefill_const() + self._decode_const()
            elif any_prefill:
                const = self._prefill_const()
            else:
                const = self._decode_const()
            return prefill_part + decode_part + const

        return prefill_part + decode_part + self.theta_c

    def estimate_solo_prefill_total_ms(self, n: int, r: int) -> float:
        """Halo R1 denominator: 'this request alone, all prefill in one shot.'

        Equivalent to estimate_step_ms([(n,r)], []). In split mode uses the
        prefill-specific intercept θ_c_p.
        """
        return (
            self.theta_p1 * n * n
            + self.theta_p2 * n * r
            + self.theta_p3 * n
            + self._prefill_const()
        )

    def estimate_solo_tbt_ms(self, r: int) -> float:
        """Halo R1 denominator: per-decode-step latency for a solo decoder.

        Equivalent to estimate_step_ms([], [r]). In split mode uses the
        decode-specific intercept θ_c_d.
        """
        return self.theta_d1 * r + self.theta_d2 * 1 + self._decode_const()

    # ---- JSON I/O ----

    @classmethod
    def from_json(cls, path: str | Path) -> "HaloStepCostModel":
        data = _read_json(path, "Halo step cost model")
        form = data.get("form")
        if form not in HALO_STEP_FORMS:
            raise CostModelLoadError(
                f"Halo step cost model at {path}: expected form in "
                f"{HALO_STEP_FORMS}, got {form!r}"
            )
        try:
            if form == HALO_STEP_FORM_SPLIT_V1:
                theta_c_p = float(data["theta_c_p"])
                theta_c_d = float(data["theta_c_d"])
                return cls(
                    theta_p1=float(data["theta_p1"]),
                    theta_p2=float(data["theta_p2"]),
                    theta_p3=float(data["theta_p3"]),
                    theta_d1=float(data["theta_d1"]),
                    theta_d2=float(data["theta_d2"]),
                    # Mirror the decode intercept into theta_c so any
                    # caller that ignores form still sees a sensible value.
                    theta_c=theta_c_d,
                    theta_c_p=theta_c_p,
                    theta_c_d=theta_c_d,
                    form=form,
                    metadata=data.get("fit_metadata", {}),
                )
            return cls(
                theta_p1=float(data["theta_p1"]),
                theta_p2=float(data["theta_p2"]),
                theta_p3=float(data["theta_p3"]),
                theta_d1=float(data["theta_d1"]),
                theta_d2=float(data["theta_d2"]),
                theta_c=float(data["theta_c"]),
                form=form,
                metadata=data.get("fit_metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CostModelLoadError(
                f"Invalid Halo step cost model at {path}: missing/non-numeric "
                f"theta coefficient ({e})"
            ) from e

    def to_json(self, path: str | Path) -> None:
        if self.is_split:
            payload = {
                "form": HALO_STEP_FORM_SPLIT_V1,
                "theta_p1": self.theta_p1,
                "theta_p2": self.theta_p2,
                "theta_p3": self.theta_p3,
                "theta_c_p": self._prefill_const(),
                "theta_d1": self.theta_d1,
                "theta_d2": self.theta_d2,
                "theta_c_d": self._decode_const(),
                "fit_metadata": self.metadata or {},
            }
        else:
            payload = {
                "form": HALO_STEP_FORM_V1,
                "theta_p1": self.theta_p1,
                "theta_p2": self.theta_p2,
                "theta_p3": self.theta_p3,
                "theta_d1": self.theta_d1,
                "theta_d2": self.theta_d2,
                "theta_c": self.theta_c,
                "fit_metadata": self.metadata or {},
            }
        Path(path).write_text(json.dumps(payload, indent=2))


def try_load_halo_step_cost_model(
    path: Optional[str],
) -> Optional[HaloStepCostModel]:
    """Lenient loader: returns None and logs WARN on failure.

    Same contract as try_load_prefill_cost_model — the Halo controller treats
    a None return as "fall back to legacy two-model path" rather than fatal.
    Handles both unified (halo_step_v1) and split (halo_step_split_v1) forms.
    """
    if not path:
        return None
    try:
        model = HaloStepCostModel.from_json(path)
    except CostModelLoadError as e:
        logger.warning("halo: step cost model disabled — %s", e)
        return None
    if model.is_split:
        logger.info(
            "halo: step cost model loaded from %s [split form] "
            "(θ_p1=%.4g θ_p2=%.4g θ_p3=%.4g θ_c_p=%.4g | "
            "θ_d1=%.4g θ_d2=%.4g θ_c_d=%.4g)",
            path,
            model.theta_p1, model.theta_p2, model.theta_p3,
            model._prefill_const(),
            model.theta_d1, model.theta_d2,
            model._decode_const(),
        )
    else:
        logger.info(
            "halo: step cost model loaded from %s [unified form] "
            "(θ_p1=%.4g θ_p2=%.4g θ_p3=%.4g θ_d1=%.4g θ_d2=%.4g θ_c=%.4g)",
            path,
            model.theta_p1, model.theta_p2, model.theta_p3,
            model.theta_d1, model.theta_d2, model.theta_c,
        )
    return model
