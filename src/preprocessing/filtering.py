"""Zero-phase Butterworth bandpass filtering."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

def filter_signal(signal, fs, config):
    x=np.asarray(signal,dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError("Signal must be a finite [samples, channels] array")
    f=config.raw["filter"]
    sos=butter(f["prototype_order"],[f["low_hz"],f["high_hz"]],btype="bandpass",fs=fs,output="sos")
    return sosfiltfilt(sos,x,axis=0,padtype=f["padtype"])
