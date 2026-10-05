import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from src.federated.config import load_federated_config
from src.federated.fedavg import (batchnorm_state_keys, client_start_state, client_update, fedavg_aggregate,
                                  get_model_state)
from src.federated.robustness import load_robustness_config, planned_runs, run_overrides
from src.federated.run_fedavg import apply_overrides
from src.models.cnn1d import ECG1DCNN

ROOT = Path(__file__).resolve().parents[1]


class BatchNormKeyTests(unittest.TestCase):
    def test_batchnorm_keys_cover_only_batchnorm_layers(self):
        keys = batchnorm_state_keys(ECG1DCNN())
        self.assertEqual(len(keys), 15)  # 3 BatchNorm1d layers x (weight, bias, running_mean, running_var, counter)
        self.assertTrue(all(k.startswith(("features.1.", "features.5.", "features.9.")) for k in keys))
        self.assertNotIn("features.0.weight", keys)
        self.assertNotIn("classifier.1.weight", keys)

    def test_start_state_default_is_global_and_local_substitutes_batchnorm(self):
        global_state = get_model_state(ECG1DCNN())
        self.assertIs(client_start_state(global_state, None), global_state)
        keys = batchnorm_state_keys(ECG1DCNN())
        local = {k: torch.full_like(global_state[k], 7) for k in keys}
        start = client_start_state(global_state, local)
        self.assertEqual(list(start), list(global_state))
        for k in global_state:
            torch.testing.assert_close(start[k], local[k] if k in local else global_state[k])
        with self.assertRaises(ValueError):
            client_start_state(global_state, {"not.a.key": torch.zeros(1)})


class FedBNRoundTests(unittest.TestCase):
    def test_one_round_shares_non_bn_and_keeps_client_bn_local(self):
        torch.use_deterministic_algorithms(True)
        gen = torch.Generator().manual_seed(0)
        clients = [TensorDataset(torch.randn(24, 2, 216, generator=gen) + shift, torch.randint(0, 15, (24,), generator=gen))
                   for shift in (0.0, 3.0)]
        torch.manual_seed(1)
        model = ECG1DCNN()
        global_state = get_model_state(model)
        keys = batchnorm_state_keys(model)
        kwargs = dict(local_epochs=1, batch_size=8, learning_rate=1e-3, optimizer_name="Adam")
        updates = [client_update(ECG1DCNN(), global_state, data, seed=10 + i, **kwargs) for i, data in enumerate(clients)]
        local_bn = {i: {k: u["state"][k] for k in keys} for i, u in enumerate(updates)}
        new_global = fedavg_aggregate([u["state"] for u in updates], [24, 24])
        for i in range(2):
            start = client_start_state(new_global, local_bn[i])
            for k in keys:  # BatchNorm: the client's own state, not the average
                torch.testing.assert_close(start[k], updates[i]["state"][k], rtol=0, atol=0)
            torch.testing.assert_close(start["features.0.weight"], new_global["features.0.weight"], rtol=0, atol=0)
        self.assertFalse(torch.equal(local_bn[0]["features.1.running_mean"], local_bn[1]["features.1.running_mean"]))


class FedBNDesignTests(unittest.TestCase):
    def test_config_validation_rejects_unknown_batchnorm_mode(self):
        base = json.loads((ROOT / "configs/federated/fedavg_v1.json").read_text(encoding="utf-8"))
        base["aggregation"]["batchnorm"] = "partial"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_federated_config(path)

    def test_diagnostic_matches_day11_design_except_batchnorm(self):
        fedbn = load_robustness_config(ROOT / "configs/federated/fedbn_diagnostic_v1.json")
        day11 = load_robustness_config(ROOT / "configs/federated/fedavg_v1_robustness.json")
        self.assertEqual(fedbn["seeds"], day11["seeds"])
        self.assertEqual(fedbn["primary_partition"], day11["primary_partition"])
        self.assertEqual(fedbn["base_config"], day11["base_config"])
        self.assertEqual(fedbn["output_root"], "results/diagnostics/fedbn")
        base = load_federated_config(ROOT / fedbn["base_config"])
        for run_id, strategy, _, seed in planned_runs(fedbn):
            fedbn_cfg = apply_overrides(base, run_overrides(run_id, strategy, seed, fedbn["extra_overrides"]))
            fedavg_cfg = apply_overrides(base, run_overrides(run_id, strategy, seed))
            self.assertEqual(fedbn_cfg["aggregation"]["batchnorm"], "local")
            fedbn_cfg["aggregation"].pop("batchnorm")
            self.assertEqual(fedbn_cfg, fedavg_cfg)  # the only difference is the BatchNorm mode

    def test_extra_overrides_cannot_change_per_run_fields(self):
        with self.assertRaises(ValueError):
            run_overrides("x", "random_group_equal_count", 1, {"random_seed": 5})


if __name__ == "__main__":
    unittest.main()
