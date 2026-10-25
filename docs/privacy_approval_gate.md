# Privacy approval gate for the HE+DP pipeline

## Status

This document records the minimum approval gate required before any new HE+DP utility claim is made.

The repository already preserves the historical HE+DP v1 and v2 artifacts and must not replace or reinterpret them. Any new privacy experiment must use a new output directory and an explicit, versioned config.

## Required approval items

Before running a new HE+DP experiment, the following must be explicitly documented and approved:

1. Privacy unit and adjacency relation
2. Threat model and trust assumptions for the decryptor, server, and clients
3. Key ownership and key-distribution assumptions
4. Whether the same public roster and public weights are fixed across rounds
5. Exact noise placement (before or after encryption, and by whom)
6. Accountant assumptions and release set
7. The final evaluation protocol and whether the test split is excluded from tuning

## Current repo evidence

The current v2 protocol already records these assumptions in [`he_dp_v2_protocol.md`](he_dp_v2_protocol.md) and [`configs/he_dp_fedavg_v2.json`](../configs/he_dp_fedavg_v2.json):

- fixed public weights `[0.2, 0.2, 0.2, 0.2, 0.2]`
- client-slot adjacency with a trusted decryptor
- DP noise added after decryption of the weighted aggregate
- CKKS encryption for the clipped client update
a- GDP accounting with `epsilon ≈ 7.98` and `delta = 1e-5`

These are documented assumptions, not a deployment-grade certification. They remain open research evidence and should not be used as an unqualified privacy guarantee.

## Approval decision

The approval gate remains open until the researcher or supervisor confirms the above items in writing. Until then, the repo should describe the privacy protocol as a conditional, protocol-specific evidence statement rather than a guaranteed privacy claim.
