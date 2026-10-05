"""CKKS encryption, homomorphic weighted aggregation, and decryption (TenSEAL / Microsoft SEAL).

Roles in the simulation
-----------------------
* Key holders (the clients) share one CKKS secret key. They encrypt with the
  public key and decrypt the aggregate.
* The server receives only a *public* context (no secret key) and serialized
  ciphertexts. It computes sum_k p_k * Enc(x_k) using plaintext-scalar
  multiplication (public weights p_k) and ciphertext addition, then returns the
  encrypted aggregate. It never sees an individual client vector in plaintext.

Every ciphertext crosses the client/server boundary as serialized bytes, so the
reported sizes are measured, not estimated. CKKS is approximate: decrypted
results carry a small numerical error that is measured, not assumed.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
import tenseal as ts

# HomomorphicEncryption.org standard: max total coefficient-modulus bits for 128-bit
# classical security with a uniform ternary secret (as enforced by Microsoft SEAL).
MAX_COEFF_BITS_128 = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}


@dataclass(frozen=True)
class CKKSParameters:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple = (60, 40, 40, 60)
    global_scale_bits: int = 40
    security_level_bits: int = 128

    @property
    def slots(self):
        return self.poly_modulus_degree // 2

    def validate(self):
        n, bits = self.poly_modulus_degree, list(self.coeff_mod_bit_sizes)
        if n not in MAX_COEFF_BITS_128:
            raise ValueError(f"Unsupported poly_modulus_degree {n}")
        if self.security_level_bits != 128:
            raise ValueError("Only the 128-bit security level is supported")
        if sum(bits) > MAX_COEFF_BITS_128[n]:
            raise ValueError(f"Coefficient modulus {sum(bits)} bits exceeds the 128-bit limit "
                             f"{MAX_COEFF_BITS_128[n]} for N={n}")
        if len(bits) < 3:
            raise ValueError("Need at least one rescaling level for plaintext-scalar multiplication")
        if not all(b == self.global_scale_bits for b in bits[1:-1]):
            raise ValueError("Intermediate primes must equal the scale bits so rescaling preserves the scale")
        return self

    @classmethod
    def from_config(cls, cfg):
        return cls(int(cfg["poly_modulus_degree"]), tuple(int(b) for b in cfg["coeff_mod_bit_sizes"]),
                   int(cfg["global_scale_bits"]), int(cfg.get("security_level_bits", 128))).validate()


class CKKSSession:
    """Holds the secret-key context (key holders) and the public context bytes (server)."""

    def __init__(self, params=CKKSParameters()):
        self.params = params.validate()
        self.secret_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=params.poly_modulus_degree,
                                         coeff_mod_bit_sizes=list(params.coeff_mod_bit_sizes))
        self.secret_context.global_scale = 2 ** params.global_scale_bits
        public = self.secret_context.copy()
        public.make_context_public()  # drops the secret key
        self.public_context_bytes = public.serialize()
        self.server_context = ts.context_from(self.public_context_bytes)
        if self.server_context.has_secret_key():
            raise RuntimeError("Server context must not contain the secret key")

    def chunks(self, length):
        return math.ceil(length / self.params.slots)

    # ------------------------------------------------------------------ key holders (clients)
    def encrypt(self, vector):
        """Encrypt a flat float vector into serialized CKKS ciphertext chunks."""
        vector = np.asarray(vector, dtype=np.float64)
        slots = self.params.slots
        return [ts.ckks_vector(self.secret_context, vector[i:i + slots].tolist()).serialize()
                for i in range(0, len(vector), slots)]

    def decrypt(self, serialized_chunks, length):
        values = []
        for blob in serialized_chunks:
            values.extend(ts.ckks_vector_from(self.secret_context, blob).decrypt())
        out = np.asarray(values, dtype=np.float64)
        if len(out) != length:
            raise ValueError(f"Decrypted length {len(out)} != expected {length}")
        return out

    # ------------------------------------------------------------------ server (public context only)
    def aggregate(self, client_chunks, weights):
        """Server side: sum_k weights[k] * Enc(x_k), chunk by chunk, without the secret key."""
        if not client_chunks or len(client_chunks) != len(weights):
            raise ValueError("Need one weight per client")
        n_chunks = len(client_chunks[0])
        if any(len(c) != n_chunks for c in client_chunks):
            raise ValueError("All clients must send the same number of ciphertext chunks")
        result = []
        for j in range(n_chunks):
            total = None
            for blob, weight in zip((c[j] for c in client_chunks), weights):
                term = ts.ckks_vector_from(self.server_context, blob) * float(weight)
                total = term if total is None else total + term
            result.append(total.serialize())
        return result


def encrypted_weighted_average(session, vectors, weights):
    """Full round trip with timings and sizes: encrypt -> server aggregate -> decrypt."""
    length = len(vectors[0])
    t0 = time.perf_counter()
    encrypted = [session.encrypt(v) for v in vectors]
    t1 = time.perf_counter()
    aggregated = session.aggregate(encrypted, weights)
    t2 = time.perf_counter()
    result = session.decrypt(aggregated, length)
    t3 = time.perf_counter()
    plaintext = np.tensordot(np.asarray(weights, dtype=np.float64), np.asarray(vectors, dtype=np.float64), axes=1)
    error = result - plaintext
    return result, {
        "encrypt_seconds": t1 - t0, "aggregate_seconds": t2 - t1, "decrypt_seconds": t3 - t2,
        "ciphertext_chunks_per_client": len(encrypted[0]),
        "uplink_ciphertext_bytes_per_client": [sum(len(b) for b in chunks) for chunks in encrypted],
        "downlink_ciphertext_bytes": sum(len(b) for b in aggregated),
        "max_abs_error": float(np.max(np.abs(error))), "rms_error": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_value": float(np.max(np.abs(plaintext))),
    }
