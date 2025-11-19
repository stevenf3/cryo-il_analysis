import tkinter as tk
from ttkbootstrap import Style
from .app import CryoILApp

def main():
    root = tk.Tk()
    fontfamily_global = 'Segoe UI'
    fontsize_global = 12
    style = Style(theme="darkly")  # try "flatly", "cosmo", etc.
    style.configure('.', font=(fontfamily_global, fontsize_global))   # <— change font family & base size
    style.configure('TLabel', font=(fontfamily_global, fontsize_global))
    style.configure('TButton', font=(fontfamily_global, fontsize_global))
    style.configure('TEntry', font=(fontfamily_global, fontsize_global))
    style.configure('TRadiobutton', font=(fontfamily_global, fontsize_global))
    style.configure('TCheckbutton', font=(fontfamily_global, fontsize_global))
    app = CryoILApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
