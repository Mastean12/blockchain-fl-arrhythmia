import unittest
from pathlib import Path

from src.blockchain.ganache_benchmark import run_ganache_benchmark


class GanacheBenchmarkTests(unittest.TestCase):
    def test_run_ganache_benchmark_writes_json(self):
        output_root = Path(__file__).resolve().parents[1] / "results" / "blockchain" / "ganache_test"
        try:
            summary = run_ganache_benchmark(output_root=output_root, rounds=4, warmup=1, port=8555)
        except (FileNotFoundError, TimeoutError, RuntimeError, OSError) as exc:
            self.skipTest(f"Ganache benchmark unavailable in this environment: {exc}")

        self.assertIn("summary", summary)
        self.assertIn("samples", summary)
        self.assertGreater(len(summary["samples"]), 0)
        self.assertGreater(summary["summary"]["mean_gas_used"], 0)
        self.assertGreater(summary["summary"]["mean_latency_ms"], 0)
        self.assertTrue((output_root / "ganache_benchmark.json").exists())
