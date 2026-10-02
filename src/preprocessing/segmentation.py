"""Beat-centered extraction preserving source annotation provenance."""
from dataclasses import dataclass
import numpy as np

@dataclass
class SegmentBatch:
    segments: np.ndarray
    labels: np.ndarray
    symbols: np.ndarray
    record_ids: np.ndarray
    patient_groups: np.ndarray
    annotation_samples: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    edge_dropped: int
    nonbeat_skipped: int

def segment_record(signal, samples, symbols, record_id, patient_group, config):
    x=np.asarray(signal)
    if x.ndim!=2 or x.shape[1]!=config.n_channels: raise ValueError("Signal shape does not match configured channels")
    samples=np.asarray(samples,dtype=np.int64); symbols=list(symbols)
    if len(samples)!=len(symbols): raise ValueError("Annotation sample/symbol mismatch")
    before,after=config.pre,config.post
    cuts=[]; labels=[]; kept_symbols=[]; ann_kept=[]; starts=[]; ends=[]
    edge=nonbeat=0
    mapping=config.labels; nonbeats=set(config.raw["labels"]["nonbeat_symbols"])
    for pos,sym in zip(samples,symbols):
        if sym not in mapping:
            if sym in nonbeats: nonbeat+=1; continue
            raise ValueError(f"Unmapped annotation symbol {sym!r} in {record_id}")
        start=int(pos)-before; end=int(pos)+after
        if start<0 or end>len(x): edge+=1; continue
        cuts.append(x[start:end]); labels.append(mapping[sym]); kept_symbols.append(sym)
        ann_kept.append(pos); starts.append(start); ends.append(end)
    shape=(0,config.segment_length,config.n_channels)
    seg=np.stack(cuts) if cuts else np.empty(shape,dtype=x.dtype)
    n=len(labels)
    return SegmentBatch(seg,np.asarray(labels,dtype=str),np.asarray(kept_symbols,dtype=str),np.full(n,record_id,dtype=str),np.full(n,patient_group,dtype=str),np.asarray(ann_kept,dtype=np.int64),np.asarray(starts,dtype=np.int64),np.asarray(ends,dtype=np.int64),edge,nonbeat)
