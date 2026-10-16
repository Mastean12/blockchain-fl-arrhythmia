"""Export and verify ledger anchors through an independently retained file."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .ledger import HEX64, Ledger, ValidationError


def export_anchor(ledger_path, anchor_path):
    """Validate a ledger and create a new anchor file without overwriting one."""
    ledger = Ledger.load(ledger_path)
    valid, problems = ledger.validate()
    if not valid:
        raise ValidationError(f"cannot anchor an invalid ledger: {problems[:3]}")
    destination = Path(anchor_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing anchor: {destination}")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(ledger.anchor(), stream, indent=2)
            stream.write("\n")
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return ledger.anchor()


def verify_anchor(ledger_path, anchor_path):
    """Return the ledger's anchor-validation result and diagnostics."""
    ledger = Ledger.load(ledger_path)
    anchor = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    if (not isinstance(anchor, dict) or set(anchor) != {"ledger_id", "length", "tip_hash"}
            or not isinstance(anchor["ledger_id"], str)
            or not isinstance(anchor["length"], int) or isinstance(anchor["length"], bool)
            or not isinstance(anchor["tip_hash"], str) or not HEX64.fullmatch(anchor["tip_hash"])):
        return False, ["anchor must contain a ledger_id, integer length, and lowercase SHA-256 tip_hash"]
    return ledger.validate(anchor=anchor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="validate a ledger and export its current tip")
    export_parser.add_argument("--ledger", required=True)
    export_parser.add_argument("--anchor", required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a ledger against a separately retained anchor")
    verify_parser.add_argument("--ledger", required=True)
    verify_parser.add_argument("--anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_anchor(args.ledger, args.anchor)
        print(json.dumps({"status": "exported", **result}, sort_keys=True))
        return 0
    valid, problems = verify_anchor(args.ledger, args.anchor)
    print(json.dumps({"status": "valid" if valid else "invalid", "problems": problems}, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
