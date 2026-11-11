# SSDD float/common-support downstream-seed 5202 FIT qualification

The first float/common-support E24 repeat recovered 3.2334367 mask-AP points against its paired floating-label native arm. The same result was 0.2203454 points above the previously fixed 70.4732456 historical integer-trained native anchor when both used the official floating-label DEV187 scorer. The paired difference and controlling difference answer distinct comparisons. The controlling difference is below the fixed 1-point criterion. The prospective decision permits one additional downstream seed, 5202.

## Controlled FIT qualification

- Both arms retain the same verified Q100 initialization produced with seed 5201 from official COCO weights. Downstream training, sampler, Lightning and CPU/CUDA random streams use seed 5202 in both arms.
- Reuse the identical SSDD FIT741 partition, 1,661 retained annotations, official continuous-vertex-to-mask label path, common-support fallback and Windows stock training entry. Native and center differ only in the already frozen support-preserving mask transformation.
- The qualification has eight actual optimizer updates and eight EMA updates over 32 image visits per arm. Both use the unchanged Q100 100-epoch schedule and 558-update warmup. This short qualification covers training transport and pairing.
- Accept only a closed original owner and both zero-exit workers. Independently verify exact clock, visit order, shared initialization, non-mask targets, common support, mask-difference witness and actual optimizer/EMA counters.
- Evaluation is outside this operation. It opens FIT only, without DEV187 or official TEST input, scoring or AP. If it qualifies, a separate original run may repeat the locked 24-epoch budget. This note supplies no third-seed authorization.

The checked-in adapter composes the prior floating-label short runner and the independently verified downstream seed binding. Runtime paths and exact protected input requests are provided by the local execution custody. The checked-in test runs with ordinary Python and no GPU. An actual short run requires the original local runtime and guards.
