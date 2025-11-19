import tkinter as tk
import numpy as np  # NEW
from ttkbootstrap import ttk
from tkinter import messagebox, filedialog
import csv
from .ui.series_panel import SeriesPanel
from .ui.controls_top import TopBar
from .ui.viewer2d import Spectrum2DView
from .ui.infopanel import InfoPanel
from .core.io import list_txt_files, read_spectrum_txt
from .core.metadata import parse_folder_metadata
from .ui.controls_filter import FilterBar          
from .core.filtering import FilterEngine           

HC_EV_NM = 1239.841984  # eV·nm

class CryoILApp:
    """
    Adds an InfoPanel outside the Notebook showing:
    Material, Temp (K), Interval, Index, Time (s), λ (nm), E (eV), Counts @ cursor.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cryo-IL Spectra Viewer")
        self.root.minsize(1100, 700)

        # App state
        self.files_all: list[str] = []
        self.files_active: list[str] = []
        self.active_index: int = 0
        self.meta = {"material": None, "temperature_K": None, "interval_s": None, "raw": ""}

        # NEW: hold current spectrum for interpolation
        self._cur_x = np.array([], dtype=float)
        self._cur_y = np.array([], dtype=float)

        # top-level grid on the root
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)  # TopBar
        self.root.grid_rowconfigure(1, weight=0)  # FilterBar
        self.root.grid_rowconfigure(2, weight=0)  # InfoPanel
        self.root.grid_rowconfigure(3, weight=1, minsize=200)  # Notebook (expands)


        # --- Top bar ---
        self.top = TopBar(self.root, on_pick=self._on_pick_folder)
        self.top.frame.grid(row=0, column=0, sticky="ew")

        # --- Filter bar (NEW) ---
        self.filter_engine = FilterEngine()  # NEW
        self.filterbar = FilterBar(
            self.root,
            on_preview=self._on_filter_preview,    # NEW
            on_apply=self._on_filter_apply,        # NEW
            on_reset=self._on_filter_reset,        # NEW
        )
        self.filterbar.frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 0))  # NEW

        # shift Info panel down one row (was row=1)
        self.info = InfoPanel(self.root)
        self.info.frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 0))

        # Notebook row index also shifts (was row=2)
        self.nb = ttk.Notebook(self.root, padding=(10, 10, 10, 10))
        self.nb.grid(row=3, column=0, sticky="nsew")


        # 2D tab container
        self.tab2d = ttk.Frame(self.nb)
        self.nb.add(self.tab2d, text="2D Viewer")
        self.tab2d.columnconfigure(0, weight=3)  # 2D plot wider
        self.tab2d.columnconfigure(1, weight=2)  # series panel
        self.tab2d.rowconfigure(0, weight=1)

        # left: 2D spectral viewer
        self.view2d = Spectrum2DView(
            self.tab2d,
            on_index_change=self._on_index_change,
            on_cursor_change=self._on_cursor_change,
            on_save_spectrum=self._on_save_spectrum_csv,
            on_toggle_include=self._on_toggle_include            
        )

        self.include_mask = []
        self.view2d.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # right: series panel
        self.series = SeriesPanel(
            self.tab2d, 
            on_generate=self._on_series_generate, 
            on_fit=self._on_series_fit, 
            on_save=self._on_save_series_csv
            )
        self.series.frame.grid(row=0, column=1, sticky="nsew")

        self.series.set_fluence_enabled(self.info.get_flux() is not None)

        # keyboard navigation
        self.root.bind("<Left>",  lambda e: self.nudge(-1))
        self.root.bind("<Right>", lambda e: self.nudge(+1))
        self.root.bind("<Home>",  lambda e: self.go_to(0))
        self.root.bind("<End>",   lambda e: self.go_to(len(self.files_active) - 1))

    def _on_save_spectrum_csv(self):
        """Save the current spectrum (Wavelength_nm, Energy_eV, Counts) to CSV."""
        if self._cur_x is None or self._cur_y is None or self._cur_x.size == 0:
            messagebox.showwarning("No spectrum", "No current spectrum to save.")
            return
        # suggest a filename from the current file or index
        sug = "spectrum.csv"
        if self.files_active and 0 <= self.active_index < len(self.files_active):
            import os
            sug = os.path.splitext(os.path.basename(self.files_active[self.active_index]))[0] + "_spectrum.csv"

        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save spectrum CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=sug
        )
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Wavelength_nm", "Energy_eV", "Counts"])
                for lam, c in zip(self._cur_x, self._cur_y):
                    ev = HC_EV_NM / lam if lam > 0 else float("nan")
                    w.writerow([f"{lam:.6g}", f"{ev:.6g}", f"{c:.6g}"])
            messagebox.showinfo("Saved", f"Spectrum CSV saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save CSV:\n{e}")

    def _on_save_series_csv(self):
        """Save the last generated series from the Series panel to CSV."""
        ser = self.series.last_series()
        if not ser:
            messagebox.showwarning("No series", "Generate a series first.")
            return

        x = ser["x"]; y = ser["y"]; kind = ser["kind"]; idxs = ser["idxs"]; lam = ser["lam_nm"]
        if x is None or y is None or x.size == 0:
            messagebox.showwarning("No series", "No data to save.")
            return

        # Compute the other axis when possible (time<->fluence) and include index column if available
        flux = self.info.get_flux()
        if kind == "fluence":
            phi = x
            time_s = (phi / float(flux)) if (flux is not None and flux > 0) else None
        else:  # time
            time_s = x
            phi = (time_s * float(flux)) if (flux is not None and flux > 0) else None

        # filename suggestion
        base = "series"
        if lam and lam == lam:
            base += f"_{lam:.1f}nm"
        base += f"_{kind}"
        sug = base + ".csv"

        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save series CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=sug
        )
        if not path:
            return

        # header columns: Index (if available), Time_s (if available), Fluence (if available), Counts, Wavelength_nm
        cols = []
        if idxs is not None: cols.append("Index")
        if time_s is not None: cols.append("Time_s")
        if phi is not None: cols.append("Fluence_(ions_cm^-2)")
        cols.append("Counts")
        cols.append("Wavelength_nm")

        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                n = len(y)
                for j in range(n):
                    row = []
                    if idxs is not None:
                        row.append(int(idxs[j]) if j < len(idxs) else "")
                    if time_s is not None:
                        row.append(f"{float(time_s[j]):.9g}")
                    if phi is not None:
                        row.append(f"{float(phi[j]):.9g}")
                    row.append(f"{float(y[j]):.9g}")
                    row.append(f"{lam:.6g}" if lam and lam == lam else "")
                    w.writerow(row)
            messagebox.showinfo("Saved", f"Series CSV saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save CSV:\n{e}")


    # ---------- model: C(phi) = 1 / (a + b * exp(-sigma * phi)) ----------
    @staticmethod
    def _model_counts_power(phi, C, A, phi0, n):
        # guard to keep phi0 positive
        phi = np.asarray(phi, float)
        phi0 = float(max(phi0, 1e-30))
        return C + A * np.power(1.0 + (phi / phi0), -n)

    def _fit_counts_vs_fluence_power(self, phi, counts):
        """
        Fit y(phi) = C + A * (1 + phi/phi0)^(-n) using scipy curve_fit (bounded).
        Returns (C, A, phi0, n, (phi_fit, yhat), R2) or None on failure.
        """
        phi = np.asarray(phi, float)
        y = np.asarray(counts, float)
        m = np.isfinite(phi) & np.isfinite(y)
        phi, y = phi[m], y[m]
        if phi.size < 4:
            return None

        # Initial guesses
        C0 = float(np.percentile(y, 5))                      # baseline from tail-ish
        A0 = float(max(np.max(y) - C0, 1e-9))                # amplitude
        span = float(max(np.max(phi) - np.min(phi), 1.0))
        phi0_0 = 0.2 * span                                  # characteristic fluence
        n0 = 1.0                                             # exponent

        p0 = (C0, A0, phi0_0, n0)

        # Bounds: keep positive A, phi0, n; allow C anywhere (can be small)
        lb = (-np.inf, 0.0, 1e-20, 1e-6)
        ub = ( np.inf, np.inf, np.inf, 10.0)

        try:
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(
                self._model_counts_power, phi, y, p0=p0, bounds=(lb, ub), maxfev=20000
            )
            C, A, phi0, n = map(float, popt)
        except Exception:
            return None

        # Evaluate fit and R^2
        yhat = self._model_counts_power(phi, C, A, phi0, n)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        return (C, A, phi0, n, (phi, yhat), r2)

    def _on_series_fit(self):
        """
        Fit the last generated series with the power-law model and overlay the curve.
        Works for either X kind: 'time' or 'fluence' (converts time -> fluence with flux).
        """
        x = self.series._xvals
        y = self.series._yvals
        if x is None or y is None or x.size == 0 or y.size == 0:
            from tkinter import messagebox
            messagebox.showwarning("No series", "Generate a series first.")
            return

        kind = getattr(self.series, "_last_kind", "time")  # 'time' or 'fluence'
        flux = self.info.get_flux()

        # Build fluence array used for fitting
        if kind == "fluence":
            phi = np.asarray(x, float)
        else:
            if flux is None or flux <= 0:
                from tkinter import messagebox
                messagebox.showwarning("Missing flux", "Set a positive flux in the Info panel to fit.")
                return
            phi = np.asarray(x, float) * float(flux)

        res = self._fit_counts_vs_fluence_power(phi, y)
        if res is None:
            self.series.set_status("Fit failed (power-law). Check data or flux.")
            return

        C, A, phi0, n, (phi_fit, yhat_phi), r2 = res

        # Convert fit X back to whichever axis the panel currently shows
        if kind == "fluence":
            xfit = phi_fit
            label = f"power: C={C:.3g}, A={A:.3g}, φ₀={phi0:.3g}, n={n:.3g}, R²={r2:.3f}"
        else:
            # time = phi / flux
            xfit = phi_fit / float(flux)
            label = f"power: C={C:.3g}, A={A:.3g}, φ₀={phi0:.3g} (fluence), n={n:.3g}, R²={r2:.3f}"

        # Sort for a clean line and overlay
        order = np.argsort(xfit)
        self.series.overlay_fit(xfit[order], yhat_phi[order], label=label)
        self.series.set_status(label)

        # Keep the orange marker visible
        self.series.highlight_by_index(self.active_index)


    # ---- folder selection ----
    def _on_pick_folder(self, folder: str):
        files = list_txt_files(folder)
        self.files_all = list(files)
        self.files_active = list(files)
        self.active_index = 0

        # --- initialize include mask (default ON) ---
        n = len(self.files_active)
        self.include_mask = [True] * n
        if n:
            self.view2d.set_include_state(True)


        self.top.set_folder(folder if folder else "No folder selected")
        self.top.set_counts(active=len(self.files_active), total=len(self.files_all))

        # parse static metadata from folder name
        self.meta = parse_folder_metadata(folder)
        self.info.set_static(self.meta["material"], self.meta["temperature_K"], self.meta["interval_s"])
        self.info.set_flux("")

        if self.files_active:
            self.view2d.set_index_range(0, len(self.files_active) - 1)
            self.go_to(0)
        else:
            self.view2d.set_index_range(0, 0)
            self.view2d.update_spectrum([], [], 0, 0, "")
            self.info.update(index=0, lam_nm=float("nan"), energy_ev=float("nan"), counts=float("nan"))

    def _on_series_generate(self, kind: str, step: int, style: str):
        """
        Build counts vs time/fluence at current cursor wavelength across active files.
        kind: "time" or "fluence"
        step: sample every Nth index (>=1)
        style: "scatter" or "line"
        """
        if not self.files_active:
            messagebox.showwarning("No data", "No active files to plot.")
            return

        # --- apply inclusion mask before building the series ---
        included = [i for i, ok in enumerate(self.include_mask) if ok]
        if not included:
            messagebox.showwarning("No points", "All points are excluded; nothing to plot.")
            return

        # Apply the step only to the included indices
        idxs = included[::max(1, int(step))]


        lam = self.view2d.current_cursor_nm()
        if not (np.isfinite(lam) and lam > 0):
            messagebox.showwarning("No cursor", "Place the cursor (nm/eV) first.")
            return

        # interval and optional flux
        interval_s = self.meta.get("interval_s", None)
        if interval_s is None or interval_s <= 0:
            messagebox.showwarning("Missing interval", "Interval (from folder name) is missing or invalid.")
            return

        flux = self.info.get_flux()  # may be None
        if kind == "fluence" and (flux is None or flux <= 0):
            messagebox.showwarning("Missing flux", "Set a positive flux in the Info panel to plot fluence.")
            self.series.set_fluence_enabled(False)
            return
        # keep UI in sync
        self.series.set_fluence_enabled(flux is not None and flux > 0)

        # iterate through active files with step
        idxs = list(range(0, len(self.files_active), max(1, int(step))))
        times = np.array([i * interval_s for i in idxs], dtype=float)

        # build counts via interpolation at lam (nm)
        counts = np.empty(len(idxs), dtype=float)
        for j, i in enumerate(idxs):
            x, y = read_spectrum_txt(self.files_active[i])
            x = np.asarray(x, float); y = np.asarray(y, float)
            if x.size and np.any(np.diff(x) <= 0):
                x, uniq = np.unique(x, return_index=True)
                y = y[uniq]
            if x.size == 0:
                counts[j] = np.nan
            else:
                try:
                    counts[j] = float(np.interp(lam, x, y))
                except Exception:
                    counts[j] = np.nan

        # choose x and label
        if kind == "fluence":
            xvals = times * float(flux)
        else:
            xvals = times

        # clean any all-nan case
        if not np.isfinite(counts).any():
            messagebox.showinfo("No data", "Interpolated counts are all NaN at this cursor.")
            return

        # plot on the panel
        self.series.show_plot(xvals, counts, kind=kind, style=style, lam_nm=lam, idxs=idxs)
        self.series.set_status(
            f"Plotted {len(idxs)} included points (step={step}) at λ={lam:.2f} nm "
            f"({sum(self.include_mask)} / {len(self.include_mask)} active)"
        )

        self.series.highlight_by_index(self.active_index)

    # ---- navigation ----
    def _on_index_change(self, idx: int):
        self.go_to(idx)

    def nudge(self, step: int):
        if not self.files_active:
            return
        self.go_to(self.active_index + step)

    def go_to(self, idx: int):
        if not self.files_active:
            return
        n = len(self.files_active)
        idx = max(0, min(n - 1, idx))
        self.active_index = idx

        path = self.files_active[idx]
        x, y = read_spectrum_txt(path)

        # NEW: keep a clean, monotonic copy for interpolation
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.size and np.any(np.diff(x) <= 0):
            x, uniq_idx = np.unique(x, return_index=True)
            y = y[uniq_idx]
        self._cur_x, self._cur_y = x, y

        self.view2d.update_spectrum(x, y, idx, n, path)

        # update info panel dynamic fields (including counts at cursor)
        lam = self.view2d.current_cursor_nm()
        ev = (HC_EV_NM / lam) if (np.isfinite(lam) and lam > 0) else float("nan")
        cts = self._interp_counts(lam)
        self.info.update(index=idx, lam_nm=lam, energy_ev=ev, counts=cts)

    # --- keep Include checkbox in sync ---
        try:
            self.view2d.set_include_state(self.include_mask[idx])
        except Exception:
            pass

        self.series.highlight_by_index(self.active_index)

        self.series.set_fluence_enabled(self.info.get_flux() is not None)

    def _on_toggle_include(self, state: bool):
        """Toggle inclusion of the current index for series/fits."""
        if not self.files_active:
            return
        i = int(getattr(self, "active_index", 0))
        if 0 <= i < len(self.include_mask):
            self.include_mask[i] = bool(state)
            # nudge the series marker (no auto-regenerate, per your preference)
            try:
                self.series.highlight_by_index(self.active_index)
            except Exception:
                pass

    # ---- cursor changes from the 2D viewer ----
    def _on_cursor_change(self, lam_nm: float, ev: float):
        cts = self._interp_counts(lam_nm)
        self.info.update(lam_nm=lam_nm, energy_ev=ev, counts=cts)
        self.series.set_fluence_enabled(self.info.get_flux() is not None)

        self.series.highlight_by_index(self.active_index)


    # ---- helpers ----
    def _interp_counts(self, lam_nm: float):
        """Interpolate counts at wavelength (nm) from current spectrum."""
        if not (np.isfinite(lam_nm) and lam_nm > 0):
            return float("nan")
        x, y = self._cur_x, self._cur_y
        if x.size == 0 or y.size == 0:
            return float("nan")
        try:
            return float(np.interp(lam_nm, x, y))
        except Exception:
            return float("nan")
        
    # ---- filtering actions (NEW) ----
    def _on_filter_preview(self):
        if not self.files_all:
            return
        params = self.filterbar.get_params()
        kept, removed = self.filter_engine.preview(self.files_all, params)
        from tkinter import messagebox
        messagebox.showinfo("Filter Preview", f"Would keep {kept} / {len(self.files_all)}\nRemove {removed}")

    def _on_filter_apply(self):
        if not self.files_all:
            return
        params = self.filterbar.get_params()
        self.files_active = self.filter_engine.apply(self.files_all, params)
        self.top.set_counts(active=len(self.files_active), total=len(self.files_all))
        if self.files_active:
            self.view2d.set_index_range(0, len(self.files_active) - 1)
            self.go_to(0)
        else:
            self.view2d.set_index_range(0, 0)
            self.view2d.update_spectrum([], [], 0, 0, "")
            self.info.update(index=0, lam_nm=float("nan"), energy_ev=float("nan"), counts=float("nan"))

    def _on_filter_reset(self):
        if not self.files_all:
            return
        self.files_active = list(self.files_all)
        self.top.set_counts(active=len(self.files_active), total=len(self.files_all))
        self.view2d.set_index_range(0, len(self.files_active) - 1)
        self.go_to(0)


    def on_close(self):
        self.root.destroy()
