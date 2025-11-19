# cryoil_gui/core/metadata.py
import os
import re

_UNIT_TO_SECONDS = {
    "s": 1.0, "sec": 1.0,
    "ms": 1e-3, "msec": 1e-3,
    "us": 1e-6, "µs": 1e-6, "μs": 1e-6,
}

_FOLDER_RE = re.compile(
    r"""
    ^\s*
    (?P<material>[A-Za-z0-9\-]+)
    [_\-]
    (?P<temp>\d+)\s*K
    [_\-]
    (?P<interval>\d+(?:\.\d+)?)
    \s*
    (?P<unit>us|µs|μs|msec|ms|sec|s)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def parse_folder_metadata(folder_path: str):
    """
    Parse folder base name like 'GaN_68K_200ms' into:
      material: 'GaN'
      temperature_K: 68.0
      interval_s: 0.200
    Returns dict with None values if not detected.
    """
    base = os.path.basename(os.path.normpath(folder_path))
    m = _FOLDER_RE.match(base)
    if not m:
        return {"material": None, "temperature_K": None, "interval_s": None, "raw": base}

    material = m.group("material")
    temperature_K = float(m.group("temp"))
    val = float(m.group("interval"))
    unit = m.group("unit").lower()
    interval_s = val * _UNIT_TO_SECONDS.get(unit, 1.0)

    return {
        "material": material,
        "temperature_K": temperature_K,
        "interval_s": interval_s,
        "raw": base,
    }
