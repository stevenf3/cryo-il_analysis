# cryoil_gui/core/filtering.py
import numpy as np
from .io import read_spectrum_txt

class FilterParams:
    def __init__(
        self,
        metric: str = "snr_mad",          # "max" | "area" | "snr_mad"
        threshold: float = 50.0,
        rule: str = ">=",                 # ">=" or "<="
        roi_min: float | None = None,     # wavelength nm
        roi_max: float | None = None,
        enable_spike: bool = True,
        spike_ratio_thresh: float = 10.0, # max/second_max
        support_frac: float = 0.3,        # fraction of max to count support points
        support_pts_max: int = 3          # <= N points above support threshold → spike-like
    ):
        self.metric = metric
        self.threshold = float(threshold)
        self.rule = rule
        self.roi_min = roi_min
        self.roi_max = roi_max
        self.enable_spike = bool(enable_spike)
        self.spike_ratio_thresh = float(spike_ratio_thresh)
        self.support_frac = float(support_frac)
        self.support_pts_max = int(support_pts_max)

class FilterEngine:
    @staticmethod
    def _roi_mask(x, lo, hi):
        if lo is None and hi is None:
            return np.ones_like(x, dtype=bool)
        if lo is None:
            lo = float(np.nanmin(x))
        if hi is None:
            hi = float(np.nanmax(x))
        if lo > hi:
            lo, hi = hi, lo
        return (x >= lo) & (x <= hi)

    @staticmethod
    def _snr_mad(y):
        y = np.asarray(y, float)
        if y.size == 0:
            return 0.0
        med = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - med))
        denom = 1.4826 * mad if mad > 0 else (np.nanstd(y) or 1e-12)
        return float((np.nanmax(y) - med) / denom)

    @staticmethod
    def _metrics(x, y):
        if x.size == 0 or y.size == 0:
            return {"max": 0.0, "area": 0.0, "snr_mad": 0.0, "spikiness": 0.0, "support_pts": 0}
        m = float(np.nanmax(y))
        area = float(np.trapezoid(y, x)) if x.size > 1 else 0.0
        snr = FilterEngine._snr_mad(y)

        # spikiness: max / second_max (robust to ties)
        if y.size >= 2:
            # get top two values without full sort
            part = np.partition(y, -2)
            second = float(part[-2])
        else:
            second = 1e-12
        spikiness = float(m / (second + 1e-12))
        return {"max": m, "area": area, "snr_mad": snr, "spikiness": spikiness, "support_pts": 0}

    @staticmethod
    def _support_points(y, support_frac):
        if y.size == 0:
            return 0
        ymax = float(np.nanmax(y))
        thr = ymax * float(support_frac)
        return int(np.count_nonzero(y >= thr))

    def _load_and_roi(self, path: str, params: FilterParams):
        x, y = read_spectrum_txt(path)
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.size and np.any(np.diff(x) <= 0):
            x, idx = np.unique(x, return_index=True)
            y = y[idx]
        if x.size:
            mask = self._roi_mask(x, params.roi_min, params.roi_max)
            x, y = x[mask], y[mask]
        return x, y

    def _passes_signal(self, metrics: dict, params: FilterParams) -> bool:
        val = metrics[params.metric]
        return (val >= params.threshold) if params.rule == ">=" else (val <= params.threshold)

    def _is_spike_like(self, y, metrics: dict, params: FilterParams) -> bool:
        if not params.enable_spike:
            return False
        sp = metrics["spikiness"]
        if sp < params.spike_ratio_thresh:
            return False
        # few points above a fraction of the peak → suspicious narrow spike
        support_pts = self._support_points(y, params.support_frac)
        metrics["support_pts"] = support_pts
        return support_pts <= params.support_pts_max

    # public
    def preview(self, files: list[str], params: FilterParams):
        kept = 0; removed = 0
        for p in files:
            x, y = self._load_and_roi(p, params)
            m = self._metrics(x, y)
            if self._passes_signal(m, params) and not self._is_spike_like(y, m, params):
                kept += 1
            else:
                removed += 1
        return kept, removed

    def apply(self, files: list[str], params: FilterParams) -> list[str]:
        out = []
        for p in files:
            x, y = self._load_and_roi(p, params)
            m = self._metrics(x, y)
            if self._passes_signal(m, params) and not self._is_spike_like(y, m, params):
                out.append(p)
        return out
