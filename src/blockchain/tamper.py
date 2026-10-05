"""Seeded tamper scenarios for the FL round ledger.

Each trial deep-copies a valid ledger, applies one manipulation at a random
position, and records whether ``validate()`` flags it on its own ("internal") and
when checked against the anchor taken before tampering ("with_anchor").
"""
import copy
import random
import tempfile
from pathlib import Path

from .ledger import Ledger, ValidationError, sha256_hex


def _rehash_from(ledger, start):
    """Consistently recompute hashes and links from `start` onwards (an attacker with full write access)."""
    for i in range(start, len(ledger.blocks)):
        if i > 0:
            ledger.blocks[i].previous_hash = ledger.blocks[i - 1].block_hash
        ledger.blocks[i].block_hash = ledger.blocks[i].compute_hash()


def _fake_hash(rng):
    return sha256_hex(rng.getrandbits(256).to_bytes(32, "big"))


def _round_block(ledger, rng):
    return rng.randrange(1, len(ledger.blocks))


def edit_update_hash(ledger, rng):
    block = ledger.blocks[_round_block(ledger, rng)]
    rng.choice(block.participants)["update_hash"] = _fake_hash(rng)


def edit_update_hash_rehash_block(ledger, rng):
    i = _round_block(ledger, rng)
    rng.choice(ledger.blocks[i].participants)["update_hash"] = _fake_hash(rng)
    ledger.blocks[i].block_hash = ledger.blocks[i].compute_hash()


def edit_num_samples(ledger, rng):
    record = rng.choice(ledger.blocks[_round_block(ledger, rng)].participants)
    record["num_samples"] += rng.randrange(1, 1000)


def remove_participant(ledger, rng):
    block = ledger.blocks[_round_block(ledger, rng)]
    block.participants.pop(rng.randrange(len(block.participants)))


def edit_aggregation_hash(ledger, rng):
    ledger.blocks[rng.randrange(0, len(ledger.blocks))].aggregation_hash = _fake_hash(rng)


def edit_timestamp(ledger, rng):
    block = ledger.blocks[_round_block(ledger, rng)]
    block.timestamp = block.timestamp[:-1] + ("1" if block.timestamp[-1] != "1" else "2")


def edit_round(ledger, rng):
    ledger.blocks[_round_block(ledger, rng)].round += 1000


def delete_block(ledger, rng):
    ledger.blocks.pop(rng.randrange(1, len(ledger.blocks) - 1))


def swap_adjacent_blocks(ledger, rng):
    i = rng.randrange(1, len(ledger.blocks) - 1)
    ledger.blocks[i], ledger.blocks[i + 1] = ledger.blocks[i + 1], ledger.blocks[i]


def insert_forged_block(ledger, rng):
    i = rng.randrange(1, len(ledger.blocks))
    forged = copy.deepcopy(ledger.blocks[i - 1])
    forged.index, forged.previous_hash = i, ledger.blocks[i - 1].block_hash
    forged.aggregation_hash = _fake_hash(rng)
    forged.block_hash = forged.compute_hash()
    ledger.blocks.insert(i, forged)


def full_rewrite_from_block(ledger, rng):
    i = _round_block(ledger, rng)
    rng.choice(ledger.blocks[i].participants)["update_hash"] = _fake_hash(rng)
    _rehash_from(ledger, i)


def truncate_tail(ledger, rng):
    k = rng.randrange(1, len(ledger.blocks) - 1)
    del ledger.blocks[-k:]


SCENARIOS = {
    "edit_update_hash": edit_update_hash,
    "edit_update_hash_and_rehash_block": edit_update_hash_rehash_block,
    "edit_num_samples": edit_num_samples,
    "remove_participant_record": remove_participant,
    "edit_aggregation_hash": edit_aggregation_hash,
    "edit_timestamp": edit_timestamp,
    "edit_round_number": edit_round,
    "delete_block": delete_block,
    "swap_adjacent_blocks": swap_adjacent_blocks,
    "insert_forged_block": insert_forged_block,
    "full_rewrite_with_recomputed_hashes": full_rewrite_from_block,
    "truncate_newest_blocks": truncate_tail,
}


def file_byte_flip_trial(ledger, rng, directory):
    """Flip one hex/digit character inside a saved block line; loading or validation must fail."""
    path = Path(directory) / "ledger.jsonl"
    ledger.save(path)
    lines = path.read_bytes().splitlines(keepends=True)
    line_no = rng.randrange(1, len(lines))  # skip the header line
    line = bytearray(lines[line_no])
    candidates = [j for j, c in enumerate(line) if chr(c) in "0123456789abcdef"]
    j = rng.choice(candidates)
    line[j] = ord(rng.choice([c for c in "0123456789abcdef" if c != chr(line[j])]))
    lines[line_no] = bytes(line)
    path.write_bytes(b"".join(lines))
    try:
        loaded = Ledger.load(path)
    except (ValidationError, ValueError, TypeError, KeyError):
        return True, True  # unreadable file counts as detected
    ok, _ = loaded.validate()
    ok_anchor, _ = loaded.validate(anchor=ledger.anchor())
    return not ok, not ok_anchor


def run_tamper_trials(ledger, trials, seed):
    if not ledger.validate()[0]:
        raise ValueError("Tamper trials need a valid starting ledger")
    anchor = ledger.anchor()
    rows = []
    for name, scenario in SCENARIOS.items():
        rng = random.Random(f"{seed}:{name}")
        internal = anchored = 0
        for _ in range(trials):
            copy_ = copy.deepcopy(ledger)
            scenario(copy_, rng)
            internal += not copy_.validate()[0]
            anchored += not copy_.validate(anchor=anchor)[0]
        rows.append({"scenario": name, "trials": trials, "detected_internal": internal,
                     "detected_with_anchor": anchored, "internal_rate": internal / trials,
                     "anchored_rate": anchored / trials})
    rng = random.Random(f"{seed}:file_byte_flip")
    internal = anchored = 0
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(trials):
            a, b = file_byte_flip_trial(ledger, rng, tmp)
            internal += a
            anchored += b
    rows.append({"scenario": "file_byte_flip_on_disk", "trials": trials, "detected_internal": internal,
                 "detected_with_anchor": anchored, "internal_rate": internal / trials,
                 "anchored_rate": anchored / trials})
    return rows
