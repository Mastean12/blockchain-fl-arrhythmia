"""Load and validate the versioned preprocessing configuration."""
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class PreprocessConfig:
    raw: dict
    source: Path

    @property
    def fs(self): return int(self.raw["dataset"]["sampling_frequency_hz"])
    @property
    def n_channels(self): return int(self.raw["dataset"]["channels"])
    @property
    def pre(self): return round(float(self.raw["segmentation"]["pre_seconds"])*self.fs)
    @property
    def post(self): return round(float(self.raw["segmentation"]["post_seconds"])*self.fs)
    @property
    def segment_length(self): return self.pre+self.post
    @property
    def labels(self): return self.raw["labels"]["beat_symbols"]

def load_config(path):
    p=Path(path)
    raw=json.loads(p.read_text(encoding="utf-8"))
    fs=raw["dataset"]["sampling_frequency_hz"]
    f=raw["filter"]
    if not 0 < f["low_hz"] < f["high_hz"] < fs/2: raise ValueError("Invalid filter cutoffs for sampling frequency")
    if f["prototype_order"] < 1: raise ValueError("Filter order must be positive")
    if len(set(raw["labels"]["beat_symbols"].values())) != len(raw["labels"]["beat_symbols"]): raise ValueError("Label mapping merges source symbols")
    if set(raw["labels"]["beat_symbols"]) & set(raw["labels"]["nonbeat_symbols"]): raise ValueError("Beat/nonbeat symbol overlap")
    c=PreprocessConfig(raw,p.resolve())
    if c.segment_length <= 0: raise ValueError("Segment length must be positive")
    return c
