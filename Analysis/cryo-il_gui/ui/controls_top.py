import tkinter as tk
import tkinter.filedialog as tkfd
from ttkbootstrap import ttk

class TopBar:
    '''
    Top Panel Controls with:

    -folder label
    -active/total count of loaded files
    -buttons to load folder

    '''

    def __init__(self, parent, on_pick):
        self.frame = ttk.Frame(parent, padding=(10, 10, 10, 0))
        self.frame.columnconfigure(0, weight=1)
        self.on_pick = on_pick

        self.folder_label = ttk.Label(self.frame, text='No folder selected')
        self.folder_label.grid(row=0, column=0, sticky='ew')

        self.count_label = ttk.Label(self.frame, text='0 files loaded', bootstyle='info')
        self.count_label.grid(row=0, column=1, sticky='ew', padx=10)

        self.ChooseFolderButton = ttk.Button(self.frame, text='Choose Folder', bootstyle='outline', command=self.choose_folder)
        self.ChooseFolderButton.grid(row=0, column=2, sticky='ew')

    def choose_folder(self):
        folder  = tkfd.askdirectory(title='Select Folder')
        if folder:
            self.on_pick(folder)

    def set_folder(self, text: str):
        self.folder_label.config(text=text)

    def set_counts(self, active: int, total: int):
        if total > 0:
            self.count_label.config(text=f'{active} / {total} active')
        else:
            self.count_label.config(text='')                                