"""Correctness of CKKS homomorphic aggregation against plaintext aggregation.

Documented tolerance: max |HE - plaintext| <= HE_TOLERANCE * max(1, max |plaintext|).
With N = 8192, scale 2^40 the observed relative error is about 1e-7, so the
tolerance of 1e-5 leaves roughly two orders of magnitude of margin.
"""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import tenseal as ts
import torch

from src.federated.config import load_federated_config
from src.federated.fedavg import fedavg_aggregate, get_model_state
from src.federated.robustness import load_robustness_config, planned_runs, run_overrides
from src.federated.run_fedavg import apply_overrides, run_experiment
from src.he.aggregation import he_fedavg_aggregate
from src.he.ckks import CKKSParameters, CKKSSession, encrypted_weighted_average
from src.he.config import validate_he_config
from src.models.cnn1d import ECG1DCNN

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/he/he_fedavg_v1.json"
HE_TOLERANCE = 1e-5


def _tol(reference):
    return HE_TOLERANCE * max(1.0, float(np.max(np.abs(reference))))


class HomomorphicAggregationCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = CKKSSession(CKKSParameters())

    def test_small_weighted_average_matches_plaintext(self):
        vectors = [np.array([1.5, -2.25, 3.0, 0.0]), np.array([0.5, 0.25, -1.0, 4.0]),
                   np.array([-3.0, 1.0, 2.0, -0.5])]
        weights = np.array([10, 20, 30]) / 60
        result, info = encrypted_weighted_average(self.session, vectors, weights)
        expected = np.tensordot(weights, np.stack(vectors), axes=1)
        np.testing.assert_allclose(result, expected, rtol=0, atol=_tol(expected))
        self.assertLessEqual(info["max_abs_error"], _tol(expected))

    def test_model_sized_multi_chunk_aggregation_matches_plaintext(self):
        rng = np.random.default_rng(0)
        length = 10351  # floating-point state entries of ECG1DCNN: 3 ciphertext chunks of 4096 slots
        vectors = [rng.uniform(-10, 10, length) for _ in range(5)]
        weights = np.array([16406, 18698, 16563, 13946, 11937], dtype=float)
        weights /= weights.sum()
        result, info = encrypted_weighted_average(self.session, vectors, weights)
        expected = np.tensordot(weights, np.stack(vectors), axes=1)
        self.assertEqual(info["ciphertext_chunks_per_client"], 3)
        self.assertEqual(result.shape, (length,))
        self.assertLessEqual(info["max_abs_error"], _tol(expected))

    def test_server_context_has_no_secret_key_and_cannot_decrypt(self):
        self.assertFalse(self.session.server_context.has_secret_key())
        blob = self.session.encrypt(np.array([1.0, 2.0]))[0]
        server_vector = ts.ckks_vector_from(self.session.server_context, blob)
        with self.assertRaises(Exception):
            server_vector.decrypt()

    def test_ciphertexts_are_randomized_and_not_plaintext(self):
        x = np.array([0.25, -0.5, 1.0])
        a, b = self.session.encrypt(x)[0], self.session.encrypt(x)[0]
        self.assertNotEqual(a, b)  # probabilistic encryption
        self.assertNotIn(np.asarray(x, dtype=np.float64).tobytes(), a)

    def test_parameter_validation_enforces_128_bit_security(self):
        with self.assertRaises(ValueError):
            CKKSParameters(8192, (60, 60, 60, 60), 60).validate()  # 240 > 218 bits
        with self.assertRaises(ValueError):
            CKKSParameters(8192, (60, 60), 40).validate()  # no rescaling level
        with self.assertRaises(ValueError):
            CKKSParameters(8192, (60, 30, 40, 60), 40).validate()  # middle prime != scale
        self.assertEqual(CKKSParameters().slots, 4096)


class HEFedAvgIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = CKKSSession(CKKSParameters())

    def test_he_fedavg_matches_plaintext_fedavg_on_model_states(self):
        torch.manual_seed(0)
        global_state = get_model_state(ECG1DCNN())
        clients = []
        for i in range(5):
            state = copy.deepcopy(global_state)
            for key, value in state.items():
                if value.is_floating_point():
                    state[key] = value + 0.05 * torch.randn_like(value)
                else:
                    state[key] = value + 60 + i
            clients.append(state)
        counts = [16406, 18698, 16563, 13946, 11937]
        he_state, diag = he_fedavg_aggregate(global_state, clients, counts, self.session)
        plain = fedavg_aggregate(clients, counts)
        self.assertEqual(list(he_state), list(plain))
        for key in plain:
            self.assertEqual(he_state[key].dtype, plain[key].dtype)
            if plain[key].is_floating_point():
                torch.testing.assert_close(he_state[key], plain[key], rtol=0, atol=_tol(plain[key].numpy()))
            else:
                self.assertTrue(torch.equal(he_state[key], plain[key]))  # counters aggregated in plaintext
        self.assertLessEqual(diag["aggregate_max_abs_error"], HE_TOLERANCE * max(1.0, diag["aggregate_max_abs_value"]))
        self.assertEqual(diag["ciphertext_chunks_per_client"], 3)
        self.assertGreater(min(diag["uplink_ciphertext_bytes"]), 4 * diag["dimension"])  # ciphertexts exceed float32

    def test_he_config_validation(self):
        he = copy.deepcopy(json.loads(STUDY.read_text(encoding="utf-8"))["extra_overrides"]["homomorphic_encryption"])
        self.assertEqual(validate_he_config(he).slots, 4096)
        for key, value in [("scheme", "BFV"), ("library", "pyfhel"), ("encrypted_quantity", "model"),
                           ("aggregation", "median"), ("coeff_mod_bit_sizes", [60, 60, 60, 60])]:
            bad = copy.deepcopy(he)
            bad[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_he_config(bad)

    def test_he_study_differs_from_fedavg_only_by_he_section(self):
        study = load_robustness_config(STUDY)
        self.assertEqual(study["seeds"], [42, 123, 2024])
        self.assertEqual(study["output_root"], "results/he")
        base = load_federated_config(ROOT / study["base_config"])
        for run_id, strategy, _, seed in planned_runs(study):
            he = apply_overrides(base, run_overrides(run_id, strategy, seed, study["extra_overrides"]))
            plain = apply_overrides(base, run_overrides(run_id, strategy, seed))
            he.pop("homomorphic_encryption")
            self.assertEqual(he, plain)

    def test_he_cannot_be_combined_with_dp(self):
        dp = json.loads((ROOT / "configs/privacy/dp_fedavg_v1.json").read_text(encoding="utf-8"))["extra_overrides"]
        he = json.loads(STUDY.read_text(encoding="utf-8"))["extra_overrides"]
        with self.assertRaises(ValueError):
            run_experiment(ROOT / "configs/federated/fedavg_v1.json", evaluate_test=False,
                           output_root=ROOT / "results/__never_written__", overrides={**dp, **he})
        self.assertFalse((ROOT / "results/__never_written__").exists())


if __name__ == "__main__":
    unittest.main()
