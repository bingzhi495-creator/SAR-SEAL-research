# Supplemental Umbra native intake

The [technical protocol](PRELOCK.md) defines the original cohort dependencies and eligibility. The original [ranged extraction and verification cores](../UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/README.md) remain unchanged. The supplemental workers check those exact original core hashes before load. No broad dataset export is required for an authorized replay.

In a fresh location use the verified Python environment and the four content pins for the published cohort. All flags below apply to both workers.

~~~text
--manifest COHORT_PRIVATE.json --manifest-sha256 SHA
--metadata-result METADATA/RESULT_SAFE.json --metadata-result-sha256 SHA
--metadata-terminal OBSERVED_METADATA/TERMINAL.json --metadata-terminal-sha256 SHA
--published-lock COHORT_LOCK.md --published-lock-sha256 SHA --published-commit COMMIT
~~~

The operator verifies the remote cohort lock before the first source pixel read. Then use the common flags plus the following options.

~~~text
python -I -B -X utf8 observe.py --phase native-window-intake --worker extract.py
  --result RUN/intake/RESULT_SAFE.json --observer RUN/observed_intake --
  python -I -B -X utf8 extract.py [common flags]
  --native-core ORIGINAL/native_windows.py --output RUN/intake

python -I -B -X utf8 observe.py --phase native-window-readback --worker verify.py
  --result RUN/readback/RESULT_SAFE.json --observer RUN/observed_readback --
  python -I -B -X utf8 verify.py [common flags]
  --intake RUN/intake --extraction-terminal RUN/observed_intake/TERMINAL.json
  --extraction-terminal-sha256 SHA --extraction-result-sha256 SHA
  --readback-core ORIGINAL/verify_native_windows.py --output RUN/readback
~~~

These wrapped phase commands record natural exits. An intake exit of 1 with a verified partial result footer retains all unsuccessful original positions. Run independent verification once a complete or partial result and observer terminal are sealed. The readback result identifies whether at least 18 unique original windows are available with four raw matches per window. Subsequent evaluation may use those windows with the original gaps explicitly identified by the coordinator.

Run `python -B test_admission.py` for the metadata lock checks. Use `python -B test_native_readback.py --original-core-dir ORIGINAL` to verify the original extraction and independent verification on generated GeoTIFFs and simulated S3 responses.
