"""Per-segment, per-channel z-score normalization."""
import numpy as np

def normalize_segments(segments, config):
    x=np.asarray(segments,dtype=np.float64)
    if x.ndim!=3 or not np.isfinite(x).all(): raise ValueError("Segments must be finite [batch,time,channels]")
    settings=config.raw["normalization"]
    mean=x.mean(axis=1,keepdims=True)
    std=x.std(axis=1,ddof=settings["ddof"],keepdims=True)
    if np.any(std < settings["epsilon"]): raise ValueError("Constant/near-constant segment channel cannot be normalized")
    out=(x-mean)/std
    if not np.isfinite(out).all(): raise ValueError("Normalization produced non-finite values")
    return out.astype(settings["output_dtype"])
