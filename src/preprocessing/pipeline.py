"""End-to-end configurable MIT-BIH preprocessing; no splitting or balancing."""
from collections import Counter
from pathlib import Path
import json
import os
import platform
import shutil
import tempfile
import numpy as np
import scipy
import wfdb
from .config import load_config
from .ecg_io import listed_records, load_record
from .filtering import filter_signal
from .segmentation import segment_record
from .normalization import normalize_segments
from .validation import validate_batch
from .storage import save_record,write_csv,write_json

def process_dataset(raw_dir, output_dir, config_path, record_ids=None, checksums=True):
    raw_dir=Path(raw_dir); output_dir=Path(output_dir); config=load_config(config_path)
    ids=listed_records(raw_dir)
    chosen=list(record_ids) if record_ids is not None else ids
    if not chosen or len(set(chosen))!=len(chosen) or not set(chosen).issubset(ids): raise ValueError("Invalid requested record subset")
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    stage=Path(tempfile.mkdtemp(prefix=output_dir.name+".staging-",dir=output_dir.parent))
    dist=Counter(); summaries=[]; total=0
    try:
        for rid in chosen:
            raw=load_record(raw_dir,rid,config,checksums)
            filtered=filter_signal(raw.signal,raw.fs,config)
            batch=segment_record(filtered,raw.ann_samples,raw.ann_symbols,rid,raw.patient_group,config)
            batch.segments=normalize_segments(batch.segments,config)
            validate_batch(batch,config)
            save_record(stage/f"{rid}.npz",batch,raw.fs,raw.channel_names)
            counts=Counter(batch.labels.tolist()); dist.update(counts); total+=len(batch.labels)
            summaries.append({"record_id":rid,"patient_group":raw.patient_group,"samples":len(raw.signal),"annotations":len(raw.ann_samples),"segments":len(batch.labels),"edge_dropped":batch.edge_dropped,"nonbeat_skipped":batch.nonbeat_skipped})
        manifest={"dataset":"MIT-BIH Arrhythmia Database","dataset_version":"PhysioNet 1.0.0","config":json.loads(config.source.read_text()),"config_path":"configs/preprocessing/mitbih_v1.json","records":chosen,"record_count":len(chosen),"segment_count":total,"segment_shape":[config.segment_length,config.n_channels],"class_distribution":dict(sorted(dist.items())),"sampling_frequency_hz":config.fs,"no_split_created":True,"no_class_balancing":True,"software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"wfdb":wfdb.__version__}}
        write_json(stage/"manifest.json",manifest)
        write_csv(stage/"class_distribution.csv",[{"label":k,"count":v} for k,v in sorted(dist.items())],["label","count"])
        write_csv(stage/"record_summary.csv",summaries,["record_id","patient_group","samples","annotations","segments","edge_dropped","nonbeat_skipped"])
        shutil.copy2(config.source,stage/"preprocessing_config.json")
        os.replace(stage,output_dir)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return manifest

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",default="data/raw/mitbih")
    parser.add_argument("--output",default="data/processed/mitbih_v1")
    parser.add_argument("--config",default="configs/preprocessing/mitbih_v1.json")
    parser.add_argument("--records",nargs="*")
    args=parser.parse_args()
    result=process_dataset(args.raw,args.output,args.config,args.records)
    print(json.dumps({k:result[k] for k in ("record_count","segment_count","class_distribution")},indent=2))
