"""Load WFDB records after structural and source-integrity checks."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import wfdb

@dataclass
class RawRecord:
    record_id: str
    signal: np.ndarray
    fs: float
    channel_names: list
    ann_samples: np.ndarray
    ann_symbols: list
    patient_group: str

def listed_records(raw_dir):
    p=Path(raw_dir)/"RECORDS"
    ids=[x.strip() for x in p.read_text().splitlines() if x.strip()]
    if not ids or any(not x.isdigit() for x in ids): raise ValueError("Invalid or empty RECORDS file")
    return ids

def verify_record_files(raw_dir, record_id, checksums=True):
    root=Path(raw_dir)
    for ext in ("hea","dat","atr"):
        p=root/f"{record_id}.{ext}"
        if not p.is_file() or p.stat().st_size == 0: raise FileNotFoundError(f"Missing/empty source: {p}")
    manifest=root/"SHA256SUMS.txt"
    if checksums and manifest.exists():
        entries={line.split()[1].lstrip("*"):line.split()[0].lower() for line in manifest.read_text().splitlines() if len(line.split())>=2}
        for ext in ("hea","dat","atr"):
            name=f"{record_id}.{ext}"
            if name in entries:
                digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
                if digest != entries[name]: raise ValueError(f"SHA-256 mismatch: {name}")

def load_record(raw_dir, record_id, config, checksums=True):
    verify_record_files(raw_dir,record_id,checksums)
    rec=wfdb.rdrecord(str(Path(raw_dir)/record_id), physical=True)
    ann=wfdb.rdann(str(Path(raw_dir)/record_id),"atr")
    sig=np.asarray(rec.p_signal,dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != config.n_channels: raise ValueError(f"Bad signal shape for {record_id}: {sig.shape}")
    if not np.isfinite(sig).all(): raise ValueError(f"Non-finite signal in {record_id}")
    if rec.fs != config.fs: raise ValueError(f"Unexpected sampling frequency for {record_id}: {rec.fs}")
    samples=np.asarray(ann.sample,dtype=np.int64)
    symbols=list(ann.symbol)
    if samples.ndim != 1 or len(samples)!=len(symbols) or np.any(samples<0) or np.any(samples>=len(sig)): raise ValueError(f"Invalid annotations in {record_id}")
    group=config.raw["labels"].get("patient_group_overrides",{}).get(record_id,f"record_{record_id}")
    return RawRecord(record_id,sig,float(rec.fs),list(rec.sig_name),samples,symbols,group)
