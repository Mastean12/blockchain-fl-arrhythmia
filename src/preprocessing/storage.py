"""Safe NPZ and tabular serialization for derived segments."""
import csv
import json
from pathlib import Path
import numpy as np

def save_record(path, batch, fs, channel_names):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,segments=batch.segments.astype(np.float32),labels=batch.labels.astype(str),source_symbols=batch.symbols.astype(str),record_ids=batch.record_ids.astype(str),patient_groups=batch.patient_groups.astype(str),annotation_samples=batch.annotation_samples,starts=batch.starts,ends=batch.ends,sampling_frequency_hz=np.asarray(fs),channel_names=np.asarray(channel_names,dtype=str))

def load_record(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}

def write_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def write_csv(path, rows, fields):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
