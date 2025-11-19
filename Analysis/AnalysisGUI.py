import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter.filedialog as tkfd
import ttkbootstrap as ttk
import numpy as np
import pandas as pd
from icecream import ic
import glob
import os

def natural_key(s: str):
    """Sort filenames like 'file2' < 'file10'."""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_spectrum_txt(path: str):
    """Read tab or whitespace-delimited spectrum file."""
    try:
        df = pd.read_csv(path, sep=r'\t', comment='#', header=None, engine='python')
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment='#', header=None, engine='python')
    if df.shape[1] < 2:
        raise ValueError(f"Invalid format: {path}")
    x = df.iloc[:, 0].to_numpy(float)
    y = df.iloc[:, -1].to_numpy(float)
    return x, y

theme = 'darkly'  # 'flatly', 'darkly', 'cosmo', etc.
class CryoILAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.protocol('WM_DELETE_WINDOW', self.onclose)

        self.title("Cryo-IL Spectra Viewer (GaN:O)")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.style = ttk.Style(theme=theme)

        # Data attributes
        self.files = []
        self.cache = {}
        self.cache_order = []
        self.max_cache = 64

        self._build_ui()
        

        #self.plot_data()

    def onclose(self):
        plt.close('all')
        self.destroy()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # plot row expands

        # ----- Row 0: Top Controls -----
        ctrl = ttk.Frame(self, padding=10)
        ctrl.grid(row=0, column=0, sticky='ew')
        ctrl.columnconfigure(0, weight=1)

        self.folder_label = ttk.Label(ctrl, text="No folder selected")
        self.folder_label.grid(row=0, column=0, sticky='ew')

        self.btn_resel = ttk.Button(ctrl, text="Choose Folder…", command=self._pick_folder_and_load)
        self.btn_resel.grid(row=0, column=1, padx=(10, 0))

        # ----- Row 1: Plot Area -----
        plot_frame = ttk.Frame(self, padding=10)
        plot_frame.grid(row=1, column=0, sticky='nsew')
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Counts")
        self.ax.grid(True, alpha=0.25)
        self.line, = self.ax.plot([], [], lw=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky='nsew')

        self.toolbar_frame = ttk.Frame(plot_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky='ew')  # spans width under plot

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # ----- Row 2: Slider + Info -----
        bottom = ttk.Frame(self, padding=10)
        bottom.grid(row=2, column=0, sticky='ew')
        bottom.columnconfigure(2, weight=1)

        self.idx_label = ttk.Label(bottom, text="0 / 0", width=12)
        self.idx_label.grid(row=0, column=0, padx=(0, 8))

        self.file_label = ttk.Label(bottom, text="")
        self.file_label.grid(row=0, column=1, sticky='ew')

        self.scale = ttk.Scale(
            bottom, orient='horizontal', from_=0, to=0, command=self._on_scale_change
        )
        self.scale.grid(row=0, column=2, sticky='ew', padx=(10, 0))

        # Key bindings
        self.bind("<Left>", lambda e: self._nudge(-1))
        self.bind("<Right>", lambda e: self._nudge(+1))
        self.bind("<Home>", lambda e: self._go_to(0))
        self.bind("<End>", lambda e: self._go_to(len(self.files) - 1))

    def _pick_folder_and_load(self):
        folder = tkfd.askdirectory(title="Select Data Folder")
        if not folder:
            return
        self._load_folder(folder)

    def _load_folder(self, folder: str):
        patterns = ["*.txt"]
        found = []
        for p in patterns:
            found.extend(glob.glob(os.path.join(folder, p)))
        if not found:
            ttk.dialogs.Messagebox.show_error("No .txt files found in folder.", "Error")
            return
        found.sort(key=natural_key)
        self.files = found
        self.folder_label.config(text=folder)
        self.cache.clear()
        self.cache_order.clear()

        self.scale.configure(from_=0, to=len(found) - 1)
        self._go_to(0)

    def _nudge(self, step: int):
        if not self.files:
            return
        idx = int(round(float(self.scale.get())))
        idx = max(0, min(len(self.files) - 1, idx + step))
        self._go_to(idx)

    def _go_to(self, idx: int):
        if not self.files:
            return
        idx = max(0, min(len(self.files) - 1, int(idx)))
        self.scale.configure(value=idx)
        self._show_index(idx)

    def _on_scale_change(self, value):
        if not self.files:
            return
        idx = int(round(float(value)))
        self._show_index(idx)

    # ---------- Plot Update ----------
    def _show_index(self, idx: int):
        n = len(self.files)
        path = self.files[idx]
        self.idx_label.config(text=f"{idx+1} / {n}")
        self.file_label.config(text=os.path.basename(path))

        x, y = self._get_xy(idx, path)

        self.line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Counts")
        self.canvas.draw_idle()

    def _get_xy(self, idx: int, path: str):
        if idx in self.cache:
            return self.cache[idx]
        try:
            x, y = read_spectrum_txt(path)
        except Exception as e:
            ic(f"Read error: {path} -> {e}")
            x = np.array([0, 1])
            y = np.array([0, 0])
            self.ax.set_title(f"Error: {e}", color='red')
        else:
            self.ax.set_title("")
        self.cache[idx] = (x, y)
        self.cache_order.append(idx)
        if len(self.cache_order) > self.max_cache:
            evict = self.cache_order.pop(0)
            self.cache.pop(evict, None)
        return x, y

if __name__ == "__main__":
    app = CryoILAnalysisGUI()
    app.mainloop()