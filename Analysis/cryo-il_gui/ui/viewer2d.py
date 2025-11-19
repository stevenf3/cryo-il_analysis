import tkinter as tk
from ttkbootstrap import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import os

HC_EV_NM = 1239.841984  # eV·nm  (E[eV] = 1239.841984 / λ[nm])

class Spectrum2DView:
    """
    2D spectrum viewer with:
      - Matplotlib axes + toolbar
      - bottom row: index label, fixed-width filename, index slider (stable width)
      - top energy axis (eV)
      - cursor controls: units toggle (nm/eV) + slider -> red vertical line
    Public API:
      set_index_range(lo, hi)
      update_spectrum(x, y, idx, n_total, path)
    Emits:
      on_index_change(idx) via the index slider.
    """
    def __init__(self, parent, on_index_change, on_cursor_change=None, on_save_spectrum=None, on_toggle_include=None):
        self._on_cursor_change = on_cursor_change
        self.frame = ttk.Frame(parent)
        self.on_index_change = on_index_change
        self.on_save_spectrum = on_save_spectrum
        self._on_toggle_include = on_toggle_include
        # layout
        self.frame.rowconfigure(0, weight=1)  # plot
        self.frame.rowconfigure(1, weight=0)  # toolbar
        self.frame.rowconfigure(2, weight=0)  # status + index slider
        self.frame.rowconfigure(3, weight=0)  # cursor (nm/eV) slider
        self.frame.columnconfigure(0, weight=1)

        # ---- figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Counts (a.u.)')
        self.ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.25)
        (self.line,) = self.ax.plot([], [], color='tab:blue', lw=1)

        # top axis in eV mapped from wavelength (nm)
        self.secax = self.ax.secondary_xaxis('top',
                                             functions=(self._nm_to_ev, self._ev_to_nm))
        self.secax.set_xlabel('Photon Energy (eV)')
        self.secax.tick_params(axis='x', direction='in', pad=6)

        # red vertical cursor line
        self.vline = self.ax.axvline(np.nan, color='tab:red', linestyle='--', lw=1.2, alpha=0.95)
        self.vline.set_visible(False)

        # canvas + toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb = ttk.Frame(self.frame); tb.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb); self.toolbar.update()

        # ---- bottom area split into two rows: (1) labels+buttons, (2) full-width slider
        bottom = ttk.Frame(self.frame, padding=(0, 6, 0, 0))
        bottom.grid(row=2, column=0, sticky="ew")

        # first row: index/filename + buttons
        toprow = ttk.Frame(bottom)
        toprow.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        toprow.columnconfigure(0, weight=0)  # index label
        toprow.columnconfigure(1, weight=1)  # filename stretches
        toprow.columnconfigure(2, weight=0)  # Save button
        toprow.columnconfigure(3, weight=0)  # Include toggle

        # second row: slider (gets full width)
        sliderrow = ttk.Frame(bottom)
        sliderrow.grid(row=1, column=0, sticky="ew")
        sliderrow.columnconfigure(0, weight=1)

        # --- first row widgets ---
        self.idx_label = ttk.Label(toprow, text="Index: - / -", width=18, anchor="w")
        self.idx_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.file_var = tk.StringVar(value="")
        file_wrap = ttk.Frame(toprow, width=420)
        file_wrap.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        file_wrap.grid_propagate(False)
        self.file_label = ttk.Label(file_wrap, textvariable=self.file_var, anchor="w")
        self.file_label.pack(fill="both", expand=True)

        self.btn_save_spec = ttk.Button(
            toprow, text="Save Spectrum", bootstyle="secondary",
            command=self._click_save_spectrum
        )
        self.btn_save_spec.grid(row=0, column=2, sticky="e", padx=(10, 0))

        self._include_var = tk.BooleanVar(value=True)
        self.btn_include = ttk.Checkbutton(
            toprow, text="Include in Analysis",
            variable=self._include_var,
            command=self._click_include_toggle   # NOTE: call the local handler
        )
        self.btn_include.grid(row=0, column=3, sticky="e", padx=(10, 0))

        # --- second row: full-width slider ---
        self.scale = ttk.Scale(
            sliderrow, orient="horizontal", from_=0, to=0,
            command=self._on_scale
        )
        self.scale.grid(row=0, column=0, sticky="ew")


        # ---- cursor controls: units + slider + live readout + entry
        cursor = ttk.Labelframe(self.frame, text="Cursor (vertical line)", padding=(8, 6))
        cursor.grid(row=3, column=0, sticky="ew")
        for c in range(12): cursor.columnconfigure(c, weight=0)
        cursor.columnconfigure(8, weight=1)   # slider expands

        ttk.Label(cursor, text="Units:").grid(row=0, column=0, padx=(0, 6))
        self.cursor_unit = tk.StringVar(value="nm")  # "nm" or "eV"
        ttk.Radiobutton(cursor, text="nm", value="nm", variable=self.cursor_unit,
                        command=self._sync_cursor_slider_range)\
            .grid(row=0, column=1, padx=(0, 8))
        ttk.Radiobutton(cursor, text="eV", value="eV", variable=self.cursor_unit,
                        command=self._sync_cursor_slider_range)\
            .grid(row=0, column=2, padx=(0, 12))

        ttk.Label(cursor, text="Value:").grid(row=0, column=3, padx=(0, 6))
        self.cursor_readout = ttk.Label(cursor, text="—")
        self.cursor_readout.grid(row=0, column=4, padx=(0, 12))

        # NEW: type-in entry + Set button
        self.cursor_entry = ttk.Entry(cursor, width=10)
        self.cursor_entry.grid(row=0, column=5, padx=(0, 6))
        ttk.Button(cursor, text="Set", command=self._apply_cursor_entry)\
            .grid(row=0, column=6, padx=(0, 12))
        self.cursor_entry.bind("<Return>", lambda e: self._apply_cursor_entry())

        ttk.Label(cursor, text="Slider:").grid(row=0, column=7, padx=(0, 6))
        self.cursor_slider = ttk.Scale(cursor, orient="horizontal", from_=0, to=1,
                                       command=self._on_cursor_move)
        self.cursor_slider.grid(row=0, column=8, sticky="ew")


        # state to keep slider range updated from current data limits
        self._x_nm_min = np.nan
        self._x_nm_max = np.nan

    def _click_include_toggle(self):
        """User toggled the include checkbox; notify controller."""
        if callable(self._on_toggle_include):
            try:
                self._on_toggle_include(bool(self._include_var.get()))
            except Exception:
                pass

    def set_include_state(self, state: bool):
        """Programmatically update the include checkbox to reflect current index."""
        self._include_var.set(bool(state))


    def _click_save_spectrum(self):
        if callable(self.on_save_spectrum):
            self.on_save_spectrum()

    # ---------- public API ----------
    def set_index_range(self, lo: int, hi: int):
        self.scale.configure(from_=lo, to=hi)

    def update_spectrum(self, x: np.ndarray, y: np.ndarray, idx: int, n_total: int, path: str):
        # update plot
        if len(x) == 0:
            self.line.set_data([], [])
            self.ax.set_title('No Data', color='orange')
        else:
            self.ax.set_title("")
            self.line.set_data(x, y)

        # labels
        self.idx_label.config(text=f"Index: {idx+1} / {n_total}")
        name = os.path.basename(path) if path else ""
        self.file_var.set(self._shorten_middle(name, max_chars=60))
        self.scale.configure(value=idx)

        # rescale & remember wavelength limits
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

        # update slider range for cursor based on current x-limits (nm)
        self._update_x_limits_from_axes()
        self._sync_cursor_slider_range()

    # ---------- internal: index slider ----------
    def _on_scale(self, value):
        try:
            idx = int(round(float(value)))
            self.on_index_change(idx)
        except Exception:
            pass

    # ---------- internal: cursor helpers ----------
    def _on_cursor_move(self, value):
        """Handle cursor slider movement in current units (nm or eV)."""
        try:
            val = float(value)
        except Exception:
            return

        if self.cursor_unit.get() == "nm":
            lam_nm = val
        else:  # eV
            ev = val
            if ev <= 0:
                return
            lam_nm = self._ev_to_nm(ev)

        # move red vertical line
        if np.isfinite(lam_nm):
            self.vline.set_xdata([lam_nm, lam_nm])
            self.vline.set_visible(True)
        else:
            self.vline.set_visible(False)

        # update readout in both units
        if np.isfinite(lam_nm) and lam_nm > 0:
            ev = self._nm_to_ev(lam_nm)
            self.cursor_readout.config(text=f"{lam_nm:,.2f} nm  |  {ev:,.3f} eV")
        else:
            self.cursor_readout.config(text="—")

        self.canvas.draw_idle()

        # keep entry in sync with current units
        self.cursor_entry.delete(0, tk.END)
        self.cursor_entry.insert(0, f"{val:.4g}")

        # notify controller about cursor changes
        if self._on_cursor_change is not None:
            lam = float(lam_nm) if np.isfinite(lam_nm) else float("nan")
            ev  = float(ev) if ('ev' in locals() and np.isfinite(ev)) else float("nan")
            self._on_cursor_change(lam, ev)

    def current_cursor_nm(self):
        """Return current cursor wavelength in nm (nan if not set)."""
        try:
            x = self.vline.get_xdata()
            lam = float(x[0]) if x and np.isfinite(x[0]) else float("nan")
            return lam
        except Exception:
            return float("nan")


    def _update_x_limits_from_axes(self):
        """Capture current wavelength axis limits (in nm) to set cursor slider bounds."""
        try:
            xmin, xmax = self.ax.get_xlim()
            # normalize so xmin < xmax
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            self._x_nm_min = float(xmin)
            self._x_nm_max = float(xmax)
        except Exception:
            self._x_nm_min, self._x_nm_max = np.nan, np.nan

    def _sync_cursor_slider_range(self):
        """
        Match the cursor slider to the axes range.
        In nm: left→right increases wavelength.
        In eV: left→right DECREASES energy (high on left, low on right), to match top axis.
        """
        if not np.isfinite(self._x_nm_min) or not np.isfinite(self._x_nm_max) or self._x_nm_max <= 0:
            self.cursor_slider.configure(from_=0, to=1, value=0, state="disabled")
            return

        unit = self.cursor_unit.get()
        self.cursor_slider.configure(state="normal")

        if unit == "nm":
            # normal order: low nm (left) -> high nm (right)
            left = float(self._x_nm_min)
            right = float(self._x_nm_max)
        else:
            # reversed order for eV: HIGH eV (left) -> LOW eV (right)
            e_left  = float(self._nm_to_ev(self._x_nm_min))  # small nm -> high eV
            e_right = float(self._nm_to_ev(self._x_nm_max))  # large nm -> low eV
            left, right = e_left, e_right  # do NOT sort; keep reversed on purpose

        # clamp current value into [min(left,right), max(left,right)]
        cur = float(self.cursor_slider.get())
        lo_min, hi_max = (min(left, right), max(left, right))
        if not (lo_min <= cur <= hi_max):
            cur = float(np.clip(cur, lo_min, hi_max))

        # set reversed or normal range as requested
        self.cursor_slider.configure(from_=left, to=right, value=cur)
        self._on_cursor_move(cur)



    def _apply_cursor_entry(self):
        """
        Read the typed cursor value (in current units), clamp to slider range,
        update the slider, move vline, and refresh readout.
        """
        txt = self.cursor_entry.get().strip()
        if not txt:
            return
        try:
            val = float(txt)
        except Exception:
            # allow inputs like "2.95 eV" or "520 nm"
            parts = txt.lower().replace("\u00b5", "u").split()
            if len(parts) == 2:
                try:
                    val = float(parts[0])
                    unit = parts[1]
                    # If user typed a unit, temporarily switch the toggle to match.
                    if unit in ("nm", "nanometer", "nanometers"):
                        self.cursor_unit.set("nm")
                    elif unit in ("ev", "eV".lower()):
                        self.cursor_unit.set("eV")
                    # re-sync bounds if unit toggled
                    self._sync_cursor_slider_range()
                except Exception:
                    return
            else:
                return

        lo = float(self.cursor_slider.cget("from"))
        hi = float(self.cursor_slider.cget("to"))
        val = float(np.clip(val, min(lo, hi), max(lo, hi)))


        # set slider -> will call _on_cursor_move via command
        self.cursor_slider.configure(value=val)
        # Manually invoke to ensure immediate vline/readout update
        self._on_cursor_move(val)


    # ---------- utility ----------
    @staticmethod
    def _shorten_middle(text: str, max_chars: int = 60) -> str:
        if len(text) <= max_chars:
            return text
        keep = max_chars - 3
        left = keep // 2
        right = keep - left
        return text[:left] + "..." + text[-right:]

    @staticmethod
    def _nm_to_ev(nm):
        nm = np.asarray(nm, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return HC_EV_NM / nm

    @staticmethod
    def _ev_to_nm(ev):
        ev = np.asarray(ev, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return HC_EV_NM / ev
