"""Parameter-level randomization utilities.

This module provides a lightweight randomizer that operates on model-level
parameter dataclasses (from ``wheeledlab_models``) rather than directly
on environments. It samples concrete scalar values for numeric fields
using range-like specifications embedded directly in the dataclass
fields (via ``env_setup.params``).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Tuple, get_type_hints, get_origin, get_args, Union

import numpy as np


class ParamRandomizer:
    """Randomize high-level model parameters based on field values.

    The randomizer is generic: for each numeric field in the provided
    dataclass instance it inspects the current value:

    - scalar → treated as constant,
    - list/tuple ``[low, high]`` → uniform,
    - list/tuple ``[a, b, \"uniform\"|\"normal\"]`` where ``normal`` is
      interpreted as (mean, std).

    Non-numeric fields:
    - scalar → assigned as-is by the param setter,
    - list/tuple → raise a ValueError (unsupported randomization).
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _parse_range_spec(self, spec: Any, default: float) -> Tuple[float, float, str]:
        """Return (a, b, kind) where kind ∈ {\"constant\",\"uniform\",\"normal\"}."""
        if spec is None:
            v = float(default)
            return v, v, "constant"

        if isinstance(spec, (float, int)):
            v = float(spec)
            return v, v, "constant"

        try:
            seq = list(spec)
        except TypeError:
            v = float(default)
            return v, v, "constant"

        if len(seq) == 0:
            v = float(default)
            return v, v, "constant"
        if len(seq) == 1:
            v = float(seq[0])
            return v, v, "constant"

        a = float(seq[0])
        b = float(seq[1])

        kind = "uniform"
        if len(seq) >= 3 and isinstance(seq[2], str):
            tag = seq[2].strip().lower()
            if tag in ("uniform", "normal"):
                kind = tag

        return a, b, kind

    def _sample_value(self, spec: Any, default: float) -> float:
        """Sample a float according to the given specification."""
        a, b, kind = self._parse_range_spec(spec, default)

        if kind == "normal":
            mean, std = a, b
            if std == 0.0:
                return float(mean)
            return float(self.rng.normal(loc=mean, scale=std))

        # uniform / constant
        low, high = a, b
        if low == high:
            return float(low)
        return float(self.rng.uniform(low, high))

    def randomize_params(self, base: Any) -> Any:
        """Return a randomized copy of a parameter dataclass.

        Only scalar numeric fields (floats / ints) are randomized. For each
        such field ``name`` in ``base`` whose current value is a list/tuple,
        this method interprets that value as a range specification and
        replaces it with a sampled scalar. Non-numeric fields must not be
        lists/tuples; otherwise a ValueError is raised.
        """
        if not is_dataclass(base):
            return base
        params = replace(base)

        # Resolve type hints once so that we can treat Optional[float]/Optional[int]
        # fields with default None as numeric when appropriate.
        try:
            type_hints = get_type_hints(type(base))
        except Exception:
            type_hints = {}

        for f in fields(base):
            name = f.name
            current = getattr(base, name)
            default_val = f.default

            # Determine whether this is a numeric field.
            # Primary signal: default value is a float/int (excluding bools).
            is_numeric = isinstance(default_val, (float, int)) and not isinstance(default_val, bool)

            # If the default is not numeric (e.g., Optional[float] with None),
            # fall back to the (evaluated) type annotation to decide.
            if not is_numeric:
                anno = type_hints.get(name, None)
                if anno is not None:
                    origin = get_origin(anno)
                    if origin is None:
                        # Plain annotated type, e.g., float or int.
                        if anno in (float, int):
                            is_numeric = True
                    else:
                        # Handle Optional[float]/Optional[int] (i.e., Union[..., NoneType]).
                        if origin is Union:
                            args = get_args(anno)
                            if any(arg in (float, int) for arg in args):
                                is_numeric = True

            # Non-numeric field: lists are treated as erroneous specs,
            # tuples are allowed as fixed values (e.g. colors).
            if not is_numeric:
                if isinstance(current, list):
                    raise ValueError(
                        f"Randomization spec not supported for non-numeric field '{name}': {current}"
                    )
                continue

            # Numeric field: interpret list/tuple-like values as specs.
            # Scalar numeric => constant.
            # We also handle generic sequence types (e.g. OmegaConf ListConfig).
            seq = None
            if isinstance(current, (list, tuple)):
                seq = list(current)
            # Treat plain strings as scalar values, not range specs, to avoid
            # accidentally interpreting them as sequences of characters.
            elif isinstance(current, str):
                seq = None
            elif not isinstance(current, (float, int)):
                try:
                    seq = list(current)
                except TypeError:
                    seq = None

            if seq is not None:
                default = float(seq[0]) if len(seq) > 0 else float(default_val)
                new_val = self._sample_value(seq, default=default)
                setattr(params, name, new_val)
            # else: scalar numeric, keep as-is

        return params
