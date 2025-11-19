# cryoil_gui/ui/info_panel.py
from ttkbootstrap import ttk
import tkinter as tk

class InfoPanel:
    """
    Compact readout bar with key metadata + live values.
    - set_static(material, temperature_K, interval_s)
    - set_flux(value_or_str)  # ions/cm^2/s
    - update(index=None, lam_nm=None, energy_ev=None, counts=None)
    Fluence = index * interval_s * flux (ions/cm^2)
    """
    def __init__(self, parent):
        self.frame = ttk.Labelframe(parent, text="Info", padding=(10, 6))
        for c in range(24):
            self.frame.columnconfigure(c, weight=0)
        self.frame.columnconfigure(23, weight=1)  # spacer

        # --- Static (material / temp / interval)
        ttk.Label(self.frame, text="Material:").grid(row=0, column=0, sticky="w")
        self.material = ttk.Label(self.frame, text="—")
        self.material.grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(self.frame, text="Temp (K):").grid(row=0, column=2, sticky="w")
        self.tempK = ttk.Label(self.frame, text="—")
        self.tempK.grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttk.Label(self.frame, text="Interval:").grid(row=0, column=4, sticky="w")
        self.interval = ttk.Label(self.frame, text="—")
        self.interval.grid(row=0, column=5, sticky="w", padx=(4, 12))

        # --- Dynamic (index / time / λ / E / counts)
        ttk.Label(self.frame, text="Index:").grid(row=0, column=6, sticky="w")
        self.index = ttk.Label(self.frame, text="—")
        self.index.grid(row=0, column=7, sticky="w", padx=(4, 12))

        ttk.Label(self.frame, text="Time (s):").grid(row=0, column=8, sticky="w")
        self.time_s = ttk.Label(self.frame, text="—")
        self.time_s.grid(row=0, column=9, sticky="w", padx=(4, 12))

        ttk.Label(self.frame, text="λ (nm):").grid(row=0, column=10, sticky="w")
        self.lam_nm = ttk.Label(self.frame, text="—")
        self.lam_nm.grid(row=0, column=11, sticky="w", padx=(4, 12))

        ttk.Label(self.frame, text="E (eV):").grid(row=0, column=12, sticky="w")
        self.energy = ttk.Label(self.frame, text="—")
        self.energy.grid(row=0, column=13, sticky="w", padx=(4, 12))

        ttk.Label(self.frame, text="Counts @ cursor:").grid(row=0, column=14, sticky="w")
        self.counts = ttk.Label(self.frame, text="—")
        self.counts.grid(row=0, column=15, sticky="w", padx=(4, 12))

        # --- NEW: Flux input + Fluence readout
        ttk.Label(self.frame, text="Flux (ions·cm⁻²·s⁻¹):").grid(row=0, column=16, sticky="w")
        self._flux_var = tk.StringVar(value="")  # user editable text
        self._flux_last = None                   # parsed float or None
        self.entry_flux = ttk.Entry(self.frame, width=14, textvariable=self._flux_var)
        self.entry_flux.grid(row=0, column=17, sticky="w", padx=(4, 6))
        ttk.Button(self.frame, text="Set", command=self._apply_flux).grid(row=0, column=18, sticky="w")

        ttk.Label(self.frame, text="Fluence (ions·cm⁻²):").grid(row=0, column=19, sticky="w", padx=(12, 0))
        self.fluence = ttk.Label(self.frame, text="—")
        self.fluence.grid(row=0, column=20, sticky="w", padx=(4, 12))

        # bind Enter for the flux box
        self.entry_flux.bind("<Return>", lambda e: self._apply_flux())

        # internal state
        self._interval_s = None
        self._last_time_s = None  # cached last computed time for fluence calc

    # ----- public API -----
    def set_static(self, material, temperature_K, interval_s):
        self.material.config(text=material if material else "—")
        self.tempK.config(text=f"{temperature_K:.2f}" if temperature_K is not None else "—")
        if interval_s is not None:
            if interval_s >= 1.0:
                self.interval.config(text=f"{interval_s:.3f} s")
            elif interval_s >= 1e-3:
                self.interval.config(text=f"{interval_s*1e3:.1f} ms")
            else:
                self.interval.config(text=f"{interval_s*1e6:.0f} µs")
        else:
            self.interval.config(text="—")
        self._interval_s = interval_s
        self._recompute_fluence()  # in case flux already present

    def set_flux(self, flux_value):
        """Set flux numerically; accepts float or str (scientific notation allowed)."""
        try:
            val = float(flux_value)
        except Exception:
            self._flux_last = None
            self._flux_var.set("")
        else:
            self._flux_last = val
            # use compact sci-notation formatting
            self._flux_var.set(f"{val:.6g}")
        self._recompute_fluence()

    def update(self, index=None, lam_nm=None, energy_ev=None, counts=None):
        # index & time
        if index is not None:
            self.index.config(text=f"{index:d}")
            if self._interval_s is not None:
                t = index * self._interval_s
                self._last_time_s = t
                self.time_s.config(text=f"{t:.6g}")
            else:
                self._last_time_s = None
                self.time_s.config(text="—")
            # time changed → recompute fluence if flux available
            self._recompute_fluence()

        # wavelength / energy
        if lam_nm is not None:
            self.lam_nm.config(text=f"{lam_nm:.3f}" if lam_nm == lam_nm else "—")
        if energy_ev is not None:
            self.energy.config(text=f"{energy_ev:.4f}" if energy_ev == energy_ev else "—")

        # counts
        if counts is not None:
            self.counts.config(text=f"{counts:.6g}" if counts == counts else "—")

    # ----- internal helpers -----
    def _apply_flux(self):
        txt = self._flux_var.get().strip()
        try:
            val = float(txt)
        except Exception:
            # invalid input → clear flux and fluence
            self._flux_last = None
            self._flux_var.set("")
            self.fluence.config(text="—")
            return
        self._flux_last = val
        # normalize display
        self._flux_var.set(f"{val:.6g}")
        self._recompute_fluence()

    def _recompute_fluence(self):
        """Fluence = flux * time; shows '—' if flux or time missing."""
        if self._flux_last is None or self._last_time_s is None:
            self.fluence.config(text="—")
            return
        phi = self._flux_last * self._last_time_s  # ions/cm^2
        # display with scientific notation if large/small
        self.fluence.config(text=f"{phi:.6g}")

    # convenience getters (optional)
    def get_flux(self):
        return self._flux_last
