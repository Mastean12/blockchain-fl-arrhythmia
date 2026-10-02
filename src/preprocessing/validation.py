"""Assertions for record arrays and labeled segment provenance."""
import numpy as np

def validate_batch(batch, config):
    n=len(batch.labels)
    arrays=[batch.symbols,batch.record_ids,batch.patient_groups,batch.annotation_samples,batch.starts,batch.ends]
    if any(len(a)!=n for a in arrays): raise ValueError("Label/segment metadata length mismatch")
    expected=(n,config.segment_length,config.n_channels)
    if batch.segments.shape!=expected: raise ValueError(f"Incorrect segment shape {batch.segments.shape}; expected {expected}")
    if not np.isfinite(batch.segments).all(): raise ValueError("NaN/Inf in segments")
    if not set(batch.labels.tolist()).issubset(set(config.labels.values())): raise ValueError("Invalid class label")
    for i,sym in enumerate(batch.symbols):
        if config.labels.get(str(sym)) != batch.labels[i]: raise ValueError("Label does not match source symbol")
        if batch.starts[i]+config.pre != batch.annotation_samples[i] or batch.ends[i]-config.post != batch.annotation_samples[i]: raise ValueError("Segment/annotation alignment mismatch")
    if n:
        means=batch.segments.astype(np.float64).mean(axis=1)
        stds=batch.segments.astype(np.float64).std(axis=1)
        if not np.allclose(means,0,atol=2e-5) or not np.allclose(stds,1,atol=2e-5): raise ValueError("Normalization check failed")
    return True
