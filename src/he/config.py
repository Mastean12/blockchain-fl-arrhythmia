"""Validation of the `homomorphic_encryption` experiment section."""
from .ckks import CKKSParameters


def validate_he_config(he):
    required = {"enabled", "scheme", "library", "poly_modulus_degree", "coeff_mod_bit_sizes", "global_scale_bits",
                "security_level_bits", "encrypted_quantity", "aggregation"}
    missing = required - set(he)
    if missing:
        raise ValueError(f"homomorphic_encryption config missing {sorted(missing)}")
    if he["enabled"] is not True:
        raise ValueError("homomorphic_encryption.enabled must be true when the section is present")
    if he["scheme"] != "CKKS" or he["library"] != "tenseal":
        raise ValueError("Only CKKS via TenSEAL is implemented")
    if he["encrypted_quantity"] != "client_update":
        raise ValueError("encrypted_quantity must be 'client_update'")
    if he["aggregation"] != "sample_weighted_fedavg":
        raise ValueError("HE aggregation must reproduce sample-weighted FedAvg")
    return CKKSParameters.from_config(he)
