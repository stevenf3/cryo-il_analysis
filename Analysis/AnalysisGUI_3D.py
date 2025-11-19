import os
import re
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog as tkfd
import tkinter.messagebox as mb

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class CryoILAnalysisGUI(tk.Tk):
    def __init__(self, theme='darkly'):
        super().__init__()
        self.protocol('WM_DELETE_WINDOW', self.onclose)
        self.title("Cryo-IL Spectra Viewer (GaN:O)")
        self.geometry("1200x900")
        self.minsize(1000, 750)
        self.style = ttk.Style(theme=theme)

        # Data containers
        self.files_all = []     # full list (never mutated)
        self.files = []         # active / filtered list
        self.cache = {}         # active-index -> (x,y)
        self.cache_order = []
        self.max_cache = 64
        self.metric_cache = {}  # path -> {"max":..., "area":..., "snr_mad":...}

        # 3D state
        self.cbar3d = None
        self._rotating = False

        # 2D playback state
        self._play_job = None
        self._is_playing = False
        self.speed_var = tk.DoubleVar(value=1.0)      # speed multiplier (×)
        self.use_rate_var = tk.BooleanVar(value=True) # parse rate from filename
        self.fallback_ms_var = tk.IntVar(value=200)   # ms if parsing fails
        self.direction_var = tk.StringVar(value="forward")
        self.loop_var = tk.BooleanVar(value=True)

        # --- λ-cursor state (for 2D vertical line and series) ---
        self.lambda_var = tk.DoubleVar(value=float('nan'))  # selected wavelength (nm)
        self._lambda_slider = None  # created in UI
        self._spectrum_has_data = False  # guard slider updates

        self._build_ui()

    # ---------------- Window lifecycle ----------------
    def onclose(self):
        self._stop_play()
        plt.close('all')
        self.destroy()

    # ---------------- Helpers (static) ----------------
    @staticmethod
    def natural_key(s: str):
        """Sort like humans: file2 < file10."""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    @staticmethod
    def read_spectrum_txt(path: str):
        """Robust reader: tab -> fallback whitespace; x = first col, y = last col."""
        try:
            df = pd.read_csv(path, sep=r'\t', comment='#', header=None, engine='python')
        except Exception:
            df = pd.read_csv(path, delim_whitespace=True, comment='#', header=None, engine='python')
        if df.shape[1] < 2:
            raise ValueError(f"Invalid format: {os.path.basename(path)}")
        x = df.iloc[:, 0].to_numpy(float)
        y = df.iloc[:, -1].to_numpy(float)
        return x, y

    @staticmethod
    def _snr_mad(y: np.ndarray) -> float:
        """Robust SNR: (max - median) / (1.4826 * MAD)."""
        med = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - med))
        denom = 1.4826 * mad if mad > 0 else 1e-12
        return float((np.nanmax(y) - med) / denom)

    @staticmethod
    def _parse_rate_from_filename(path: str) -> float:
        """
        Return seconds per frame parsed from filename.
        Matches: '200 ms', '0.5 s', '100us', '250 msec', '2sec', case-insensitive.
        Returns np.nan if not found.
        """
        name = os.path.basename(path)
        m = re.search(r'(\d+(?:\.\d+)?)\s*(µs|us|msec|ms|sec|s)\b', name, flags=re.IGNORECASE)
        if not m:
            return float('nan')
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ('µs', 'us'):
            return val * 1e-6
        if unit in ('msec', 'ms'):
            return val * 1e-3
        return val  # 's' or 'sec'

    def _frame_delay_ms(self, idx: int) -> int:
        """
        Compute the Tk after() delay in ms for current frame:
        - Use filename-derived rate if enabled and parse succeeds
        - Otherwise use fallback
        - Apply speed multiplier (speed>1 => faster => shorter delay)
        """
        delay_s = float(self.fallback_ms_var.get()) / 1000.0
        if self.use_rate_var.get():
            try:
                parsed = self._parse_rate_from_filename(self.files[idx])
                if not np.isnan(parsed) and parsed > 0:
                    delay_s = parsed
            except Exception:
                pass
        speed = max(0.01, float(self.speed_var.get()))
        delay_ms = int(max(10, min(5000, round((delay_s / speed) * 1000))))
        return delay_ms

    def _apply_roi_to_xy(self, x, y):
        """Apply current λ ROI to (x, y)."""
        mask = self._parse_roi(np.asarray(x))
        if not np.any(mask):
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.asarray(x)[mask], np.asarray(y)[mask]

    # ---------------- UI ----------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Row 0: folder + counts + choose
        top = ttk.Frame(self, padding=(10, 10, 10, 0))
        top.grid(row=0, column=0, sticky='ew')
        top.columnconfigure(0, weight=1)

        self.folder_label = ttk.Label(top, text="No folder selected")
        self.folder_label.grid(row=0, column=0, sticky='ew')

        self.count_label = ttk.Label(top, text="", bootstyle="info")
        self.count_label.grid(row=0, column=1, padx=10)

        self.btn_resel = ttk.Button(top, text="Choose Folder…",
                                    command=self._pick_folder_and_load, bootstyle='outline')
        self.btn_resel.grid(row=0, column=2)

        # Row 1: Filter controls (with λ-range ROI)
        filt = ttk.Labelframe(self, text="Noise / No-Signal Filter", padding=10)
        filt.grid(row=1, column=0, sticky='ew', padx=10, pady=(10, 0))
        for c in range(14):
            filt.columnconfigure(c, weight=0)
        filt.columnconfigure(13, weight=1)

        ttk.Label(filt, text="Metric:").grid(row=0, column=0, padx=(0, 6))
        self.metric_var = tk.StringVar(value="max")
        self.metric_combo = ttk.Combobox(filt, state="readonly", width=14,
                                         textvariable=self.metric_var,
                                         values=["max", "area", "snr_mad"])
        self.metric_combo.grid(row=0, column=1, padx=(0, 12))

        ttk.Label(filt, text="Threshold:").grid(row=0, column=2, padx=(0, 6))
        self.thresh_var = tk.DoubleVar(value=100.0)
        self.thresh_entry = ttk.Entry(filt, textvariable=self.thresh_var, width=10)
        self.thresh_entry.grid(row=0, column=3, padx=(0, 12))

        ttk.Label(filt, text="Rule:").grid(row=0, column=4, padx=(0, 6))
        self.rule_var = tk.StringVar(value=">=")
        self.rule_combo = ttk.Combobox(filt, state="readonly", width=6,
                                       textvariable=self.rule_var, values=[">=", "<="])
        self.rule_combo.grid(row=0, column=5, padx=(0, 12))

        ttk.Label(filt, text="λ min (nm):").grid(row=0, column=6, padx=(0, 6))
        self.lam_min_var = tk.StringVar(value="")
        self.lam_min_entry = ttk.Entry(filt, textvariable=self.lam_min_var, width=10)
        self.lam_min_entry.grid(row=0, column=7, padx=(0, 12))

        ttk.Label(filt, text="λ max (nm):").grid(row=0, column=8, padx=(0, 6))
        self.lam_max_var = tk.StringVar(value="")
        self.lam_max_entry = ttk.Entry(filt, textvariable=self.lam_max_var, width=10)
        self.lam_max_entry.grid(row=0, column=9, padx=(0, 12))

        self.btn_preview = ttk.Button(filt, text="Preview", bootstyle="secondary",
                                      command=self._preview_filter)
        self.btn_preview.grid(row=0, column=10, padx=(6, 6))

        self.btn_apply = ttk.Button(filt, text="Apply Filter", bootstyle="primary",
                                    command=self._apply_filter)
        self.btn_apply.grid(row=0, column=11, padx=(0, 6))

        self.btn_reset = ttk.Button(filt, text="Reset", bootstyle="warning",
                                    command=self._reset_filter)
        self.btn_reset.grid(row=0, column=12, padx=(10, 0))

        # Row 2: Notebook (2D + 3D)
        self.nb = ttk.Notebook(self, padding=(10, 10, 10, 10))
        self.nb.grid(row=2, column=0, sticky='nsew')

        # ---------- Tab 1: 2D viewer ----------
        self.tab2d = ttk.Frame(self.nb)
        self.nb.add(self.tab2d, text="2D Viewer")
        self.tab2d.rowconfigure(0, weight=1)   # plots
        self.tab2d.rowconfigure(1, weight=0)   # bottom2d
        self.tab2d.rowconfigure(2, weight=0)   # transport
        self.tab2d.rowconfigure(3, weight=0)   # λ controls
        self.tab2d.columnconfigure(0, weight=1)

        # ========== Two-panel plots area ==========
        plots2d = ttk.Frame(self.tab2d)
        plots2d.grid(row=0, column=0, sticky='nsew')
        plots2d.rowconfigure(0, weight=1)
        plots2d.columnconfigure(0, weight=3)  # main spectrum wider
        plots2d.columnconfigure(1, weight=2)  # time series panel

        # ----- Left: main spectrum plot -----
        left_frame = ttk.Frame(plots2d)
        left_frame.grid(row=0, column=0, sticky='nsew')
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.fig2d, self.ax2d = plt.subplots(figsize=(8, 5), dpi=100)
        self.ax2d.set_xlabel("Wavelength (nm)")
        self.ax2d.set_ylabel("Counts")
        self.ax2d.grid(True, alpha=0.25)
        self.line2d, = self.ax2d.plot([], [], lw=1)

        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=left_frame)
        self.canvas2d_widget = self.canvas2d.get_tk_widget()
        self.canvas2d_widget.grid(row=0, column=0, sticky='nsew')

        self.toolbar2d_frame = ttk.Frame(left_frame)
        self.toolbar2d_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar2d = NavigationToolbar2Tk(self.canvas2d, self.toolbar2d_frame)
        self.toolbar2d.update()

        # --- vertical λ cursor line on main 2D axes ---
        self.lambda_vline = self.ax2d.axvline(np.nan, color='tab:red',
                                              linestyle='--', lw=1.2, alpha=0.9)
        self.lambda_vline.set_visible(False)

        # ----- Right: I(λ,t) scatter panel -----
        right_frame = ttk.Frame(plots2d)
        right_frame.grid(row=0, column=1, sticky='nsew')
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self.fig_series, self.ax_series = plt.subplots(figsize=(6, 5), dpi=100)
        self.ax_series.set_title("I(λ,t)", fontsize=10)
        self.ax_series.set_xlabel("Time (s)")
        self.ax_series.set_ylabel("Counts")
        self.ax_series.grid(True, alpha=0.25)

        # Scatter artist we will update via set_offsets
        self.series_scatter = self.ax_series.scatter([], [], s=16, color='tab:purple', alpha=0.9)

        self.canvas_series = FigureCanvasTkAgg(self.fig_series, master=right_frame)
        self.canvas_series_widget = self.canvas_series.get_tk_widget()
        self.canvas_series_widget.grid(row=0, column=0, sticky='nsew')

        self.toolbar_series_frame = ttk.Frame(right_frame)
        self.toolbar_series_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar_series = NavigationToolbar2Tk(self.canvas_series, self.toolbar_series_frame)
        self.toolbar_series.update()

        # ----- Row under plots: file index slider + labels -----
        bottom2d = ttk.Frame(self.tab2d, padding=(0, 10, 0, 0))
        bottom2d.grid(row=1, column=0, sticky='ew')
        bottom2d.columnconfigure(2, weight=1)

        self.idx_label = ttk.Label(bottom2d, text="0 / 0", width=12)
        self.idx_label.grid(row=0, column=0, padx=(0, 8))

        self.file_label = ttk.Label(bottom2d, text="")
        self.file_label.grid(row=0, column=1, sticky='ew')

        self.scale = ttk.Scale(bottom2d, orient='horizontal', from_=0, to=0,
                               command=self._on_scale_change)
        self.scale.grid(row=0, column=2, sticky='ew', padx=(10, 0))

        # --- 2D playback transport row ---
        transport = ttk.Labelframe(self.tab2d, text="Playback", padding=8)
        transport.grid(row=2, column=0, sticky='ew')
        for i in range(12):
            transport.columnconfigure(i, weight=0)
        transport.columnconfigure(11, weight=1)

        self.btn_play = ttk.Button(transport, text="▶ Play", bootstyle="success",
                                   command=self._toggle_play)
        self.btn_play.grid(row=0, column=0, padx=(0, 8))

        self.btn_stop = ttk.Button(transport, text="■ Stop", bootstyle="danger",
                                   command=self._stop_play)
        self.btn_stop.grid(row=0, column=1, padx=(0, 16))

        ttk.Label(transport, text="Speed ×").grid(row=0, column=2)
        ttk.Spinbox(transport, from_=0.1, to=20.0, increment=0.1, width=6,
                    textvariable=self.speed_var).grid(row=0, column=3, padx=(0, 16))

        ttk.Label(transport, text="Direction").grid(row=0, column=4)
        ttk.Combobox(transport, state="readonly", width=10,
                     values=["forward", "backward"], textvariable=self.direction_var)\
            .grid(row=0, column=5, padx=(0, 16))

        ttk.Checkbutton(transport, text="Loop", variable=self.loop_var)\
            .grid(row=0, column=6, padx=(0, 16))

        ttk.Checkbutton(transport, text="Use rate in filename", variable=self.use_rate_var)\
            .grid(row=0, column=7, padx=(0, 12))

        ttk.Label(transport, text="Fallback (ms)").grid(row=0, column=8)
        ttk.Entry(transport, width=8, textvariable=self.fallback_ms_var)\
            .grid(row=0, column=9, padx=(0, 16))

        # Spacebar toggles play/pause
        self.bind("<space>", lambda e: self._toggle_play())

        # --- λ selection controls (slider + entry) ---
        lamctrl = ttk.Labelframe(self.tab2d, text="Pick wavelength (nm)", padding=8)
        lamctrl.grid(row=3, column=0, sticky='ew', padx=0, pady=(8, 0))
        for c in range(10):
            lamctrl.columnconfigure(c, weight=0)
        lamctrl.columnconfigure(5, weight=1)

        ttk.Label(lamctrl, text="λ:").grid(row=0, column=0, padx=(0, 6))
        self.lam_entry = ttk.Entry(lamctrl, width=10)
        self.lam_entry.grid(row=0, column=1, padx=(0, 10))
        ttk.Button(lamctrl, text="Go", command=self._apply_lambda_from_entry)\
            .grid(row=0, column=2, padx=(0, 12))

        ttk.Label(lamctrl, text="Slider:").grid(row=0, column=3, padx=(0, 6))

        # built after first spectrum is shown; set a dummy for now
        self._lambda_slider = ttk.Scale(lamctrl, orient='horizontal', from_=0, to=1,
                                        command=self._on_lambda_slider)
        self._lambda_slider.grid(row=0, column=4, sticky='ew', padx=(0, 10))

        # Keyboard navigation
        self.bind("<Left>", lambda e: self._nudge(-1))
        self.bind("<Right>", lambda e: self._nudge(+1))
        self.bind("<Home>", lambda e: self._go_to(0))
        self.bind("<End>", lambda e: self._go_to(len(self.files) - 1))

        # ---------- Tab 2: 3D view ----------
        self.tab3d = ttk.Frame(self.nb)
        self.nb.add(self.tab3d, text="3D: Spectra vs Time")
        self.tab3d.rowconfigure(2, weight=1)
        self.tab3d.columnconfigure(0, weight=1)

        # Row 0: data/plot options
        ctrl3d = ttk.Frame(self.tab3d, padding=(0, 10, 0, 6))
        ctrl3d.grid(row=0, column=0, sticky='ew')
        for c in range(14):
            ctrl3d.columnconfigure(c, weight=0)
        ctrl3d.columnconfigure(13, weight=1)

        ttk.Label(ctrl3d, text="Y axis:").grid(row=0, column=0, padx=(0, 6))
        self.y_mode = tk.StringVar(value="index")
        self.y_mode_combo = ttk.Combobox(ctrl3d, textvariable=self.y_mode,
                                         values=["index", "mtime_seconds"],
                                         state="readonly", width=16)
        self.y_mode_combo.grid(row=0, column=1, padx=(0, 15))

        ttk.Label(ctrl3d, text="Downsample (every N λ pts):").grid(row=0, column=2, padx=(0, 6))
        self.ds_var = tk.IntVar(value=2)
        self.ds_spin = ttk.Spinbox(ctrl3d, from_=1, to=50, increment=1,
                                   textvariable=self.ds_var, width=6)
        self.ds_spin.grid(row=0, column=3, padx=(0, 15))

        self.norm3d_var = tk.BooleanVar(value=False)
        self.norm3d_chk = ttk.Checkbutton(ctrl3d, text="Normalize each spectrum",
                                          variable=self.norm3d_var)
        self.norm3d_chk.grid(row=0, column=4, padx=(0, 15))

        ttk.Label(ctrl3d, text="Colormap:").grid(row=0, column=5, padx=(0, 6))
        self.cmap_var = tk.StringVar(value="viridis")
        self.cmap_combo = ttk.Combobox(ctrl3d, textvariable=self.cmap_var,
                                       state="readonly", width=12,
                                       values=["viridis", "plasma", "inferno", "magma",
                                               "cividis", "turbo"])
        self.cmap_combo.grid(row=0, column=6, padx=(0, 15))

        # Toggle for line stacks vs surface + stride
        self.lines_mode_var = tk.BooleanVar(value=False)
        self.lines_mode_chk = ttk.Checkbutton(ctrl3d, text="Render as colored line stacks",
                                              variable=self.lines_mode_var)
        self.lines_mode_chk.grid(row=0, column=7, padx=(0, 15))

        ttk.Label(ctrl3d, text="Every Nth spectrum:").grid(row=0, column=8, padx=(0, 6))
        self.spec_stride_var = tk.IntVar(value=1)
        self.spec_stride_spin = ttk.Spinbox(ctrl3d, from_=1, to=100, increment=1,
                                            width=6, textvariable=self.spec_stride_var)
        self.spec_stride_spin.grid(row=0, column=9, padx=(0, 15))

        self.btn_build3d = ttk.Button(ctrl3d, text="Build 3D Plot",
                                      bootstyle=PRIMARY, command=self._build_3d_plot)
        self.btn_build3d.grid(row=0, column=10)

        # Row 1: rotation toolbar
        view3d = ttk.Labelframe(self.tab3d, text="3D View Controls", padding=(10, 8))
        view3d.grid(row=1, column=0, sticky='ew', padx=0, pady=(0, 6))
        for c in range(10):
            view3d.columnconfigure(c, weight=0)
        view3d.columnconfigure(8, weight=1)

        ttk.Label(view3d, text="Elev").grid(row=0, column=0, padx=(0, 6))
        self.elev_var = tk.DoubleVar(value=30.0)
        elev_slider = ttk.Scale(view3d, from_=-10, to=90, orient='horizontal',
                                command=lambda v: self._update_view_from_sliders(),
                                variable=self.elev_var)
        elev_slider.grid(row=0, column=1, sticky='ew', padx=(0, 12))

        ttk.Label(view3d, text="Azim").grid(row=0, column=2, padx=(0, 6))
        self.azim_var = tk.DoubleVar(value=-60.0)
        azim_slider = ttk.Scale(view3d, from_=-180, to=180, orient='horizontal',
                                command=lambda v: self._update_view_from_sliders(),
                                variable=self.azim_var)
        azim_slider.grid(row=0, column=3, sticky='ew', padx=(0, 12))

        self.btn_iso = ttk.Button(view3d, text="Iso", command=lambda: self._set_view(30, -60))
        self.btn_iso.grid(row=0, column=4, padx=(0, 6))
        self.btn_top = ttk.Button(view3d, text="Top", command=lambda: self._set_view(90, -90))
        self.btn_top.grid(row=0, column=5, padx=(0, 6))
        self.btn_side = ttk.Button(view3d, text="Side", command=lambda: self._set_view(0, 0))
        self.btn_side.grid(row=0, column=6, padx=(0, 6))

        self._rot_speed = tk.DoubleVar(value=20.0)  # deg/sec
        self.btn_rotate = ttk.Button(view3d, text="▶ Rotate", command=self._toggle_rotate)
        self.btn_rotate.grid(row=0, column=7, padx=(6, 6))
        ttk.Label(view3d, text="Speed (°/s)").grid(row=0, column=8, padx=(6, 6))
        self.spin_speed = ttk.Spinbox(view3d, from_=1, to=180, increment=1, width=6,
                                      textvariable=self._rot_speed)
        self.spin_speed.grid(row=0, column=9, padx=(0, 0))

        # Row 2: 3D plot area + toolbar
        plot3d_frame = ttk.Frame(self.tab3d)
        plot3d_frame.grid(row=2, column=0, sticky='nsew')
        plot3d_frame.rowconfigure(0, weight=1)
        plot3d_frame.columnconfigure(0, weight=1)

        self.fig3d = plt.figure(figsize=(8, 5), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.ax3d.set_xlabel("Wavelength (nm)")
        self.ax3d.set_ylabel("Index")
        self.ax3d.set_zlabel("Counts")

        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=plot3d_frame)
        self.canvas3d_widget = self.canvas3d.get_tk_widget()
        self.canvas3d_widget.grid(row=0, column=0, sticky='nsew')

        self.toolbar3d_frame = ttk.Frame(plot3d_frame)
        self.toolbar3d_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar3d = NavigationToolbar2Tk(self.canvas3d, self.toolbar3d_frame)
        self.toolbar3d.update()

    # ---------------- Folder / files ----------------
    def _pick_folder_and_load(self):
        folder = tkfd.askdirectory(title="Select Data Folder")
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder: str):
        # case-insensitive .txt list (Windows-safe), no duplicates
        found = [e.path for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith('.txt')]
        if not found:
            mb.showerror("Error", "No .txt files found in folder.")
            return
        found.sort(key=lambda p: self.natural_key(os.path.basename(p)))

        self.files_all = list(found)
        self.files = list(found)
        self.folder_label.config(text=f"{folder}")
        self._update_counts()
        self._clear_active_cache()

        self.scale.configure(from_=0, to=len(self.files) - 1)
        self._go_to(0)
        self._stop_play()

    def _update_counts(self):
        self.count_label.config(text=f"{len(self.files)} / {len(self.files_all)} active")

    def _clear_active_cache(self):
        self.cache.clear()
        self.cache_order.clear()
        # metric_cache is per-path; keep it

    # ---------------- 2D navigation ----------------
    def _nudge(self, step: int):
        if not self.files:
            return
        idx = int(round(float(self.scale.get())))
        idx = max(0, min(len(self.files) - 1, idx + step))
        self._go_to(idx)

    def _go_to(self, idx: int):
        if not self.files:
            self.idx_label.config(text="0 / 0")
            self.file_label.config(text="")
            self.line2d.set_data([], [])
            self.ax2d.relim(); self.ax2d.autoscale_view()
            self.canvas2d.draw_idle()
            return
        idx = max(0, min(len(self.files) - 1, int(idx)))
        self.scale.configure(value=idx)
        self._show_index(idx)

    def _on_scale_change(self, value):
        if not self.files:
            return
        idx = int(round(float(value)))
        self._show_index(idx)

    # ---------------- 2D plot update ----------------
    def _show_index(self, idx: int):
        n = len(self.files)
        path = self.files[idx]
        self.idx_label.config(text=f"{idx+1} / {n}")
        self.file_label.config(text=os.path.basename(path))

        x, y = self._get_xy(idx, path)

        # Apply current λ ROI if present
        try:
            xr, yr = self._apply_roi_to_xy(x, y)
        except Exception:
            xr, yr = np.asarray(x), np.asarray(y)

        if xr.size == 0:
            self.line2d.set_data([], [])
            self.ax2d.set_title("No data in current λ range", color="orange")
        else:
            self.ax2d.set_title("")
            self.line2d.set_data(xr, yr)

        self.ax2d.relim()
        self.ax2d.autoscale_view()
        self.ax2d.set_xlabel("Wavelength (nm)")
        self.ax2d.set_ylabel("Counts")

        # Update λ slider range and cursor/series
        self._update_lambda_widgets_range()

        self.canvas2d.draw_idle()

    def _get_xy(self, active_idx: int, path: str):
        if active_idx in self.cache:
            return self.cache[active_idx]
        try:
            x, y = self.read_spectrum_txt(path)
        except Exception as e:
            x = np.array([0, 1], float)
            y = np.array([0, 0], float)
            self.ax2d.set_title(f"Error: {e}", color='red')
        else:
            self.ax2d.set_title("")
        self.cache[active_idx] = (x, y)
        self.cache_order.append(active_idx)
        if len(self.cache_order) > self.max_cache:
            evict = self.cache_order.pop(0)
            self.cache.pop(evict, None)
        return x, y

    # ---------------- Metrics & filtering ----------------
    def _parse_roi(self, x: np.ndarray):
        """Return a boolean mask for λ ROI using entries; empty -> full range."""
        lam_min_str = self.lam_min_var.get().strip()
        lam_max_str = self.lam_max_var.get().strip()
        if lam_min_str == "" and lam_max_str == "":
            return np.ones_like(x, dtype=bool)  # full
        try:
            lam_min = float(lam_min_str) if lam_min_str != "" else np.nanmin(x)
            lam_max = float(lam_max_str) if lam_max_str != "" else np.nanmax(x)
        except ValueError:
            return np.ones_like(x, dtype=bool)
        if lam_min > lam_max:
            lam_min, lam_max = lam_max, lam_min
        return (x >= lam_min) & (x <= lam_max)

    def _compute_metrics_for_path_roi(self, path: str):
        """Compute metrics within λ ROI for a single file (reads file once)."""
        try:
            x, y = self.read_spectrum_txt(path)
        except Exception:
            x = np.array([0, 1], float); y = np.array([0, 0], float)

        mask = self._parse_roi(x)
        if not np.any(mask):
            return {"max": 0.0, "area": 0.0, "snr_mad": 0.0}

        xr = x[mask]
        yr = y[mask]

        m = {}
        m["max"] = float(np.nanmax(yr))
        try:
            m["area"] = float(np.trapezoid(yr, xr))
        except Exception:
            m["area"] = float(np.nansum(yr))
        m["snr_mad"] = self._snr_mad(yr)
        return m

    def _collect_metric_array(self, file_list):
        metric = self.metric_var.get()
        vals = np.array([self._compute_metrics_for_path_roi(p)[metric] for p in file_list],
                        dtype=float)
        return vals

    def _preview_filter(self):
        if not self.files:
            return
        thresh = float(self.thresh_var.get())
        rule = self.rule_var.get()
        vals = self._collect_metric_array(self.files)
        keep = vals >= thresh if rule == ">=" else vals <= thresh
        kept = int(np.count_nonzero(keep))
        removed = len(vals) - kept
        mb.showinfo("Preview",
                    f"Metric: {self.metric_var.get()}\n"
                    f"λ ROI: [{self.lam_min_var.get() or 'min'}, {self.lam_max_var.get() or 'max'}]\n"
                    f"Threshold: {thresh} ({rule})\n\n"
                    f"Would keep {kept} / {len(vals)} files\n"
                    f"Remove {removed}.")

    def _apply_filter(self):
        if not self.files:
            return
        thresh = float(self.thresh_var.get())
        rule = self.rule_var.get()
        vals = self._collect_metric_array(self.files)
        keep_mask = (vals >= thresh) if rule == ">=" else (vals <= thresh)
        self.files = [p for p, k in zip(self.files, keep_mask) if k]
        self._update_counts()
        self._clear_active_cache()
        if self.files:
            self.scale.configure(from_=0, to=len(self.files) - 1)
            self._go_to(0)
        else:
            self.scale.configure(from_=0, to=0)
            self._go_to(0)

    def _reset_filter(self):
        if not self.files_all:
            return
        self.files = list(self.files_all)
        self._update_counts()
        self._clear_active_cache()
        self.scale.configure(from_=0, to=len(self.files) - 1)
        self._go_to(0)

    # ---------- λ selection handlers ----------
    def _on_lambda_slider(self, value):
        """Slider moved (value in nm)."""
        if not self._spectrum_has_data:
            return
        try:
            lam = float(value)
        except Exception:
            return
        self.lambda_var.set(lam)
        # reflect in entry
        self.lam_entry.delete(0, tk.END)
        self.lam_entry.insert(0, f"{lam:.2f}")
        self._update_lambda_visuals_and_series()

    def _apply_lambda_from_entry(self):
        """User types a wavelength (nm) and clicks Go."""
        if not self._spectrum_has_data:
            return
        txt = self.lam_entry.get().strip()
        try:
            lam = float(txt)
        except Exception:
            return
        xmin, xmax = self._current_xminmax()
        lam = min(max(lam, xmin), xmax)
        self.lambda_var.set(lam)
        self._lambda_slider.configure(value=lam)
        self._update_lambda_visuals_and_series()

    def _current_xminmax(self):
        """Return (xmin, xmax) of the currently shown 2D spectrum (after ROI)."""
        xdat = self.line2d.get_xdata()
        if xdat is None or len(xdat) == 0:
            return (0.0, 1.0)
        return (float(np.nanmin(xdat)), float(np.nanmax(xdat)))

    def _update_lambda_widgets_range(self):
        """Update the λ slider min/max to match the currently shown spectrum range."""
        xmin, xmax = self._current_xminmax()
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            self._spectrum_has_data = False
            self._lambda_slider.configure(from_=0.0, to=1.0, value=0.0, state='disabled')
            self.lambda_vline.set_visible(False)
            self.canvas2d.draw_idle()
            return
        self._spectrum_has_data = True
        self._lambda_slider.configure(state='normal', from_=xmin, to=xmax)
        # initialize λ at spectral peak if nan
        if not np.isfinite(self.lambda_var.get()):
            try:
                y = self.line2d.get_ydata()
                x = self.line2d.get_xdata()
                if len(x) and len(y):
                    lam0 = float(x[np.nanargmax(y)])
                else:
                    lam0 = (xmin + xmax) * 0.5
            except Exception:
                lam0 = (xmin + xmax) * 0.5
            self.lambda_var.set(lam0)
            self._lambda_slider.configure(value=lam0)
            self.lam_entry.delete(0, tk.END); self.lam_entry.insert(0, f"{lam0:.2f}")
        else:
            lam = min(max(self.lambda_var.get(), xmin), xmax)
            self.lambda_var.set(lam)
            self._lambda_slider.configure(value=lam)
            self.lam_entry.delete(0, tk.END); self.lam_entry.insert(0, f"{lam:.2f}")
        self._update_lambda_visuals_and_series()

    def _update_lambda_visuals_and_series(self):
        """Move the red vline and rebuild the I(λ,t) scatter panel."""
        lam = float(self.lambda_var.get())
        # Handle vline on the main spectrum
        if not np.isfinite(lam):
            self.lambda_vline.set_visible(False)
            # Clear scatter
            self.series_scatter.set_offsets(np.empty((0, 2)))
            self.ax_series.relim(); self.ax_series.autoscale_view()
            self.canvas2d.draw_idle()
            self.canvas_series.draw_idle()
            return

        # move vertical line on the spectrum
        self.lambda_vline.set_xdata([lam, lam])
        self.lambda_vline.set_visible(True)

        # rebuild time series across all files
        t, I = self._build_series_for_lambda(lam)

        # update scatter offsets (Nx2 array)
        if len(t) and len(I):
            pts = np.column_stack([np.asarray(t, dtype=float), np.asarray(I, dtype=float)])
            self.series_scatter.set_offsets(pts)
            # set reasonable xlim even if all t equal
            if np.nanmin(pts[:, 0]) == np.nanmax(pts[:, 0]):
                self.ax_series.set_xlim(pts[0, 0] - 1.0, pts[0, 0] + 1.0)
        else:
            self.series_scatter.set_offsets(np.empty((0, 2)))

        # rescale series axes
        self.ax_series.relim(); self.ax_series.autoscale_view()
        self.ax_series.set_title(f"I(λ={lam:.2f} nm, t)", fontsize=10)

        # redraw both canvases
        self.canvas2d.draw_idle()
        self.canvas_series.draw_idle()

    def _build_series_for_lambda(self, lam_nm: float):
        """
        Compute I(λ=lam_nm) across all active files.
        Returns (time_seconds_from_first, intensity_array).
        """
        if not self.files:
            return [], []

        # time base: modification time in seconds since first
        try:
            mtimes = np.array([os.path.getmtime(p) for p in self.files], dtype=float)
            t0 = float(np.nanmin(mtimes))
            t = (mtimes - t0).tolist()
        except Exception:
            # fallback to index
            t = list(range(len(self.files)))

        Ivals = []
        for path in self.files:
            try:
                x, y = self.read_spectrum_txt(path)
                x = np.asarray(x, float); y = np.asarray(y, float)
                if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
                    Ivals.append(np.nan); continue
                # ensure strictly increasing x for interp
                if np.any(np.diff(x) <= 0):
                    x, uniq_idx = np.unique(x, return_index=True)
                    y = y[uniq_idx]
                    if x.size == 0:
                        Ivals.append(np.nan); continue
                Ivals.append(float(np.interp(lam_nm, x, y)))
            except Exception:
                Ivals.append(np.nan)

        if not np.any(np.isfinite(Ivals)):
            return (t, [np.nan] * len(Ivals))
        return (t, Ivals)

    # ---------------- 3D builder ----------------
    def _update_3d_ylabel(self):
        mode = self.y_mode.get()
        self.ax3d.set_ylabel("Time (s from first)" if mode == "mtime_seconds" else "Index")

    def _build_3d_plot(self):
        if not self.files:
            return
        self._update_3d_ylabel()
        ds = max(1, int(self.ds_var.get()))
        normalize = bool(self.norm3d_var.get())
        mode = self.y_mode.get()

        # Reference wavelength grid from first active file
        ref_x, _ = self.read_spectrum_txt(self.files[0])
        ref_x = np.asarray(ref_x, float)
        x_idx = np.arange(0, ref_x.size, ds, dtype=int)
        x_ds = ref_x[x_idx]

        # Y values (index or seconds since first mtime)
        if mode == "mtime_seconds":
            mtimes = np.array([os.path.getmtime(p) for p in self.files], dtype=float)
            t0 = mtimes.min()
            Y_vals = mtimes - t0
        else:
            Y_vals = np.arange(len(self.files), dtype=float)

        # Build Z rows
        Z_rows = []
        for path in self.files:
            x, y = self.read_spectrum_txt(path)
            x = np.asarray(x, float); y = np.asarray(y, float)
            # Align via interpolation if needed
            if x.shape != ref_x.shape or not np.allclose(x, ref_x, rtol=0, atol=1e-9):
                if np.any(np.diff(x) <= 0):
                    x, uniq_idx = np.unique(x, return_index=True)
                    y = y[uniq_idx]
                y_interp = np.interp(ref_x, x, y)
            else:
                y_interp = y
            y_ds = y_interp[x_idx]
            if normalize:
                vmax = np.nanmax(y_ds); vmin = np.nanmin(y_ds)
                rng = vmax - vmin
                y_ds = (y_ds - vmin) / rng if rng > 0 else np.zeros_like(y_ds)
            Z_rows.append(y_ds)

        Z = np.vstack(Z_rows)          # (n_files, n_lambda_ds)
        Y = Y_vals                     # (n_files,)
        Xg, Yg = np.meshgrid(x_ds, Y)  # grids for surface

        # Clear and relabel
        self.ax3d.cla()
        self.ax3d.set_xlabel("Wavelength (nm)")
        self._update_3d_ylabel()
        self.ax3d.set_zlabel("Counts" + (" (normalized)" if normalize else ""))

        # Colormap by Y (index or seconds)
        cmap = cm.get_cmap(self.cmap_var.get())
        norm = Normalize(vmin=float(Y.min()), vmax=float(Y.max()))

        if self.lines_mode_var.get():
            # ---- Colored line stacks (faster for huge sets) ----
            stride = max(1, int(self.spec_stride_var.get()))
            for i in range(0, len(Y), stride):
                color = cmap(norm(Y[i]))
                self.ax3d.plot(x_ds, np.full_like(x_ds, Y[i], dtype=float), Z[i, :],
                               lw=0.8, color=color, alpha=0.95)
        else:
            # ---- Surface with facecolors tied to Y ----
            facecolors = cmap(norm(Yg))
            self.ax3d.plot_surface(
                Xg, Yg, Z,
                rstride=max(1, len(Y)//200 + 1),
                cstride=max(1, Z.shape[1]//400 + 1),
                linewidth=0, antialiased=False,
                facecolors=facecolors, shade=False
            )

        # Colorbar
        if self.cbar3d is not None:
            self.cbar3d.remove(); self.cbar3d = None
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        label = "Index" if mode == "index" else "Seconds since first"
        self.cbar3d = self.fig3d.colorbar(mappable, ax=self.ax3d, pad=0.12)
        self.cbar3d.set_label(label)

        # Apply current view
        self.ax3d.view_init(elev=float(self.elev_var.get()), azim=float(self.azim_var.get()))
        self.canvas3d.draw_idle()

    # ---------------- Playback core ----------------
    def _toggle_play(self):
        if self._is_playing:
            self._stop_play()
            return
        if not self.files:
            return
        self._is_playing = True
        self.btn_play.config(text="⏸ Pause", bootstyle="secondary")
        self._schedule_next_step()

    def _stop_play(self):
        self._is_playing = False
        self.btn_play.config(text="▶ Play", bootstyle="success")
        if self._play_job is not None:
            try:
                self.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None

    def _schedule_next_step(self):
        if not self._is_playing or not self.files:
            return
        idx = int(round(float(self.scale.get())))
        self._play_job = self.after(self._frame_delay_ms(idx), self._play_step)

    def _play_step(self):
        if not self._is_playing or not self.files:
            return
        n = len(self.files)
        idx = int(round(float(self.scale.get())))
        step = 1 if self.direction_var.get() == "forward" else -1
        new_idx = idx + step

        if new_idx < 0 or new_idx >= n:
            if self.loop_var.get():
                new_idx = 0 if step > 0 else n - 1
            else:
                self._stop_play()
                return

        self._go_to(new_idx)           # moves slider and redraws
        self._schedule_next_step()     # schedule based on new index

    # ---------------- 3D view controls ----------------
    def _set_view(self, elev, azim):
        self.elev_var.set(float(elev))
        self.azim_var.set(float(azim))
        self._update_view_from_sliders()

    def _update_view_from_sliders(self):
        if not hasattr(self, "ax3d"):
            return
        self.ax3d.view_init(elev=float(self.elev_var.get()), azim=float(self.azim_var.get()))
        self.canvas3d.draw_idle()

    def _toggle_rotate(self):
        self._rotating = not self._rotating
        self.btn_rotate.config(text="⏸ Pause" if self._rotating else "▶ Rotate")
        if self._rotating:
            self._do_rotate_step()

    def _do_rotate_step(self):
        if not self._rotating:
            return
        step = float(self._rot_speed.get()) * 0.03  # ~30 ms tick
        new_az = float(self.azim_var.get()) + step
        if new_az > 180:
            new_az -= 360
        self.azim_var.set(new_az)
        self._update_view_from_sliders()
        self.after(30, self._do_rotate_step)

# ---------- Run standalone ----------
if __name__ == "__main__":
    app = CryoILAnalysisGUI(theme="darkly")  # try "flatly", "cosmo", etc.
    app.mainloop()
