# cryoil_gui/ui/controls_filter.py
from ttkbootstrap import ttk
import tkinter as tk
from ..core.filtering import FilterParams

class FilterBar:
    """
    Filter control strip:
      - primary signal threshold (metric/rule/threshold, optional ROI)
      - optional spike-removal (ratio + support controls)
      - Preview / Apply / Reset
    """
    def __init__(self, parent, on_preview, on_apply, on_reset):
        self.frame = ttk.Labelframe(parent, text="Noise / No-Signal Filter", padding=8)

        # Row 0: primary signal threshold
        r = 0
        ttk.Label(self.frame, text="Metric:").grid(row=r, column=0, sticky="w")
        self.metric = tk.StringVar(value="max")
        ttk.Combobox(self.frame, width=10, state="readonly", textvariable=self.metric,
                     values=["snr_mad", "max", "area"]).grid(row=r, column=1, padx=(4, 12))

        ttk.Label(self.frame, text="Rule:").grid(row=r, column=2, sticky="w")
        self.rule = tk.StringVar(value=">=")
        ttk.Combobox(self.frame, width=5, state="readonly", textvariable=self.rule,
                     values=[">=", "<="]).grid(row=r, column=3, padx=(4, 12))

        ttk.Label(self.frame, text="Threshold:").grid(row=r, column=4, sticky="w")
        self.thresh = tk.DoubleVar(value=35.0)
        ttk.Entry(self.frame, width=10, textvariable=self.thresh).grid(row=r, column=5, padx=(4, 12))

        ttk.Label(self.frame, text="λmin (nm):").grid(row=r, column=6, sticky="w")
        self.lam_min = tk.StringVar(value="")
        ttk.Entry(self.frame, width=10, textvariable=self.lam_min).grid(row=r, column=7, padx=(4, 12))

        ttk.Label(self.frame, text="λmax (nm):").grid(row=r, column=8, sticky="w")
        self.lam_max = tk.StringVar(value="")
        ttk.Entry(self.frame, width=10, textvariable=self.lam_max).grid(row=r, column=9, padx=(4, 12))

        ttk.Button(self.frame, text="Preview", bootstyle="secondary", command=on_preview)\
            .grid(row=r, column=10, padx=(6,6))
        ttk.Button(self.frame, text="Apply", bootstyle="primary", command=on_apply)\
            .grid(row=r, column=11, padx=(0,6))
        ttk.Button(self.frame, text="Reset", bootstyle="warning", command=on_reset)\
            .grid(row=r, column=12)

        # Row 1: spike removal
        r = 1
        self.enable_spike = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frame, text="Remove single-spike junk", variable=self.enable_spike)\
            .grid(row=r, column=0, columnspan=2, sticky="w", pady=(6,0))

        ttk.Label(self.frame, text="max/2nd ≥").grid(row=r, column=2, sticky="e", pady=(6,0))
        self.spike_ratio = tk.DoubleVar(value=1.5)
        ttk.Entry(self.frame, width=8, textvariable=self.spike_ratio).grid(row=r, column=3, padx=(4, 12), pady=(6,0))

        ttk.Label(self.frame, text="support ≥").grid(row=r, column=4, sticky="e", pady=(6,0))
        self.support_frac = tk.DoubleVar(value=0.3)
        ttk.Entry(self.frame, width=6, textvariable=self.support_frac).grid(row=r, column=5, padx=(4, 12), pady=(6,0))

        ttk.Label(self.frame, text="points ≤").grid(row=r, column=6, sticky="e", pady=(6,0))
        self.support_pts_max = tk.IntVar(value=20)
        ttk.Entry(self.frame, width=6, textvariable=self.support_pts_max).grid(row=r, column=7, padx=(4, 12), pady=(6,0))

        # layout weights
        for c in range(13):
            self.frame.columnconfigure(c, weight=0)
        self.frame.columnconfigure(12, weight=1)

    def get_params(self) -> FilterParams:
        lam_min = float(self.lam_min.get()) if self.lam_min.get().strip() else None
        lam_max = float(self.lam_max.get()) if self.lam_max.get().strip() else None
        return FilterParams(
            metric=self.metric.get(),
            threshold=float(self.thresh.get()),
            rule=self.rule.get(),
            roi_min=lam_min, roi_max=lam_max,
            enable_spike=bool(self.enable_spike.get()),
            spike_ratio_thresh=float(self.spike_ratio.get()),
            support_frac=float(self.support_frac.get()),
            support_pts_max=int(self.support_pts_max.get()),
        )
