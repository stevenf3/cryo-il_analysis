# cryoil_gui/ui/series_panel.py
from ttkbootstrap import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

class SeriesPanel:
    """
    Right-side panel to generate counts-vs-(time|fluence) series at the cursor.
    NEW: highlights the currently viewed spectrum index as a big orange dot.
    """
    def __init__(self, parent, on_generate, on_fit=None, on_save=None):
        self.frame = ttk.Labelframe(parent, text="Series @ Cursor", padding=(8, 6))
        self.on_generate = on_generate
        self.on_fit = on_fit
        self.on_save = on_save
        self._fit_line = None  # handle to overlay line

        self._lam_nm_last = None

        # layout
        self.frame.rowconfigure(0, weight=0)  # controls
        self.frame.rowconfigure(1, weight=1)  # plot
        self.frame.rowconfigure(2, weight=0)  # toolbar
        self.frame.rowconfigure(3, weight=0)  # status
        self.frame.columnconfigure(0, weight=1)

        # ---- controls
        ctrl = ttk.Frame(self.frame)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for c in range(8):
            ctrl.columnconfigure(c, weight=0)
        ctrl.columnconfigure(7, weight=1)

        ttk.Label(ctrl, text="X-axis:").grid(row=0, column=0, sticky="w")
        self.kind = tk.StringVar(value="time")
        self.rb_time = ttk.Radiobutton(ctrl, text="Time", value="time", variable=self.kind)
        self.rb_time.grid(row=0, column=1, padx=(4, 10))
        self.rb_fluence = ttk.Radiobutton(ctrl, text="Fluence", value="fluence", variable=self.kind)
        self.rb_fluence.grid(row=0, column=2, padx=(0, 16))

        ttk.Label(ctrl, text="Step (N):").grid(row=0, column=3, sticky="w")
        self.step_var = tk.IntVar(value=1)
        self.step_entry = ttk.Spinbox(ctrl, from_=1, to=9999, width=6, textvariable=self.step_var)
        self.step_entry.grid(row=0, column=4, padx=(4, 16))

        ttk.Label(ctrl, text="Style:").grid(row=0, column=5, sticky="w")
        self.style = tk.StringVar(value="scatter")
        style_box = ttk.Combobox(ctrl, width=8, state="readonly", textvariable=self.style,
                                 values=["scatter", "line"])
        style_box.grid(row=0, column=6, padx=(4, 16))

        ttk.Button(ctrl, text="Generate", bootstyle="primary",
                   command=self._click_generate).grid(row=0, column=9, sticky="e")
        
        # ---- NEW: Y-axis scale control ----
        ttk.Label(ctrl, text="Y-axis:").grid(row=0, column=7, sticky="w")
        self.y_scale = tk.StringVar(value="linear")
        scale_box = ttk.Combobox(
            ctrl, width=8, state="readonly",
            textvariable=self.y_scale, values=["linear", "log"]
        )

        scale_box.grid(row=0, column=8, padx=(4, 16))
        scale_box.bind("<<ComboboxSelected>>", lambda e: self._update_y_scale())

        # NEW: Fit button
        ttk.Button(ctrl, text="Fit model", bootstyle="secondary",
                   command=self._click_fit).grid(row=0, column=10, sticky="e", padx=(6,0))
        
        # after your existing buttons (e.g., Generate, Fit, Overlay, etc.)
        ttk.Button(ctrl, text="Save CSV", bootstyle="secondary",
                command=self._click_save).grid(row=0, column=12, sticky="e", padx=(6, 0))


        # ---- plot
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Counts")
        self.ax.grid(True, alpha=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(4, 4))

        tb = ttk.Frame(self.frame)
        tb.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb)
        self.toolbar.update()

        # status line
        self.status = ttk.Label(self.frame, text="", bootstyle="info")
        self.status.grid(row=3, column=0, sticky="w", pady=(4, 0))

        # state for fluence availability and last series
        self._fluence_enabled = True
        self._last_kind = "time"

        # NEW: storage for plotted data and highlight marker
        self._idxs = None        # list[int] corresponding to points in series
        self._xvals = None       # np.ndarray
        self._yvals = None       # np.ndarray
        self._marker = None      # PathCollection (scatter) for highlight

        self._fit_line = None  # matplotlib Line2D for overlay

    def _click_save(self):
        if callable(self.on_save):
            self.on_save()

    def _update_y_scale(self):
        """Update Y-axis scale between linear and log."""
        scale = self.y_scale.get()
        if scale not in ("linear", "log"):
            scale = "linear"
        try:
            self.ax.set_yscale(scale)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _click_generate(self):
        k = self.kind.get()
        if k == "fluence" and not self._fluence_enabled:
            k = "time"; self.kind.set("time")
        try:
            step = int(self.step_var.get())
            if step < 1: step = 1
        except Exception:
            step = 1; self.step_var.set(step)
        style = self.style.get() or "scatter"
        self.on_generate(k, step, style)

    def _click_fit(self):
        if self.on_fit is not None:
            self.on_fit()  # controller will read last data from this panel and fit


    def set_fluence_enabled(self, enabled: bool):
        self._fluence_enabled = bool(enabled)
        self.rb_fluence.configure(state=("normal" if enabled else "disabled"))
        if not enabled and self.kind.get() == "fluence":
            self.kind.set("time")

    def show_plot(self, x, y, kind: str, style: str, lam_nm: float, idxs: list[int]):
        """Render the series and remember data for highlight-by-index."""
        self.ax.clear()
        if kind == "fluence":
            self.ax.set_xlabel("Fluence (ions·cm⁻²)")
        else:
            self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Counts")
        self.ax.grid(True, alpha=0.25)

        if style == "line":
            self.ax.plot(x, y, lw=1.2)
        else:
            self.ax.scatter(x, y, s=12)

        # (Re)create highlight marker fresh each time since ax.clear() detaches artists
        if self._marker is not None:
            try:
                self._marker.remove()
            except Exception:
                pass
        self._marker = self.ax.scatter([], [], s=120, edgecolors="black",
                                       facecolors="orange", zorder=5)
        self._marker.set_visible(False)  # hidden until we place it

        


        if lam_nm and lam_nm == lam_nm:
            E_ev = 1239.84193 / lam_nm
            self.ax.set_title(f"Series at λ = {lam_nm:.2f} nm, E = {E_ev:.3f} eV")
        else:
            self.ax.set_title("Series at cursor")

        self._lam_nm_last = float(lam_nm) if lam_nm and lam_nm == lam_nm else None

        self.ax.set_yscale(self.y_scale.get())

        self.canvas.draw_idle()

        # remember last plotted data for highlight updates
        self._idxs = list(idxs)
        self._xvals = np.asarray(x, float) if x is not None else None
        self._yvals = np.asarray(y, float) if y is not None else None
        self._last_kind = kind

        # remove previous fit overlay if present
        if self._fit_line is not None:
            try:
                self._fit_line.remove()
            except Exception:
                pass
            self._fit_line = None

    def last_series(self):
        """Return the last plotted series as (x, y, kind, lam_nm)."""
        if self._xvals is None or self._yvals is None:
            return None
        return {
        "x": self._xvals.copy(),
        "y": self._yvals.copy(),
        "kind": getattr(self, "_last_kind", "time"),
        "idxs": list(self._idxs) if self._idxs is not None else None,
        "lam_nm": self._lam_nm_last
    }

    def last_y_scale(self):
        try:
            return self.y_scale.get()
        except Exception:
            return "linear"

    def overlay_fit(self, xfit, yfit, label=None):
        # remove existing fit if present
        if self._fit_line is not None:
            try:
                self._fit_line.remove()
            except Exception:
                pass
        (self._fit_line,) = self.ax.plot(xfit, yfit, lw=2.0, alpha=0.9)
        if label:
            self.ax.set_title((self.ax.get_title() + "  –  " + label).strip(" -"))
        self.canvas.draw_idle()

    def highlight_by_index(self, current_idx: int):
        """
        Move the big orange marker to the point corresponding to current_idx,
        snapping to the nearest included index that has a finite y-value.
        Safe if no series has been drawn yet.
        """
        if (self._idxs is None or self._xvals is None or self._yvals is None
                or len(self._idxs) == 0 or self._xvals.size == 0):
            return

        # ensure there is a marker on the current axes
        if self._marker is None:
            self._marker = self.ax.scatter([], [], s=120, edgecolors="black",
                                           facecolors="orange", zorder=5)
            self._marker.set_visible(False)

        idxs = np.asarray(self._idxs, dtype=int)
        y = np.asarray(self._yvals, float)
        finite = np.isfinite(y)

        if not finite.any():
            # nothing finite to highlight
            self._marker.set_offsets(np.empty((0, 2)))
            self._marker.set_visible(False)
            self.canvas.draw_idle()
            return

        # nearest included index with finite y
        diffs = np.abs(idxs[finite] - int(current_idx))
        j_rel = int(np.argmin(diffs))
        j = int(np.flatnonzero(finite)[j_rel])

        xh = float(self._xvals[j]); yh = float(self._yvals[j])
        self._marker.set_offsets(np.array([[xh, yh]]))
        self._marker.set_visible(True)
        self.canvas.draw_idle()

    def set_status(self, text: str):
        self.status.config(text=text)
