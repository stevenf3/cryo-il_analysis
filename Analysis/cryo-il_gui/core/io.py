import os
import re
import numpy as np

def natural_key(s:str):
    '''
    Sort a string using natural sorting

    i.e. "file2" comes before "file10"
    '''

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_txt_files(folder: str) -> list[str]:
    '''
    List all .txt files in a folder, sorted using natural sorting
    '''
    if not folder or not os.path.isdir(folder):
        return []
    
    files = [
        e.path for e in os.scandir(folder)
        if e.is_file() and e.name.lower().endswith('.txt')
    ]

    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    
    return files

def read_spectrum_txt(path: str):
    """
    Read a spectrum where the first column is wavelength (nm)
    and the last column is counts. Supports tab or whitespace.
    Returns (x, y) as float numpy arrays.
    """
    import pandas as pd
    try:
        df = pd.read_csv(path, sep=r'\t', comment='#', header=None, engine='python')
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment='#', header=None, engine='python')
    if df.shape[1] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)  # invalid format
    x = df.iloc[:, 0].to_numpy(float)
    y = df.iloc[:, -1].to_numpy(float)
    return x, y