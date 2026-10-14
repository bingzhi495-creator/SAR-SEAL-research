# UNBOUND_DRAFT — N16 CPU qualifier body v03

Only revision from preserved02: raw_target_mapping accepts native tuple or valid list, exact Python INT elements (bool/float reject), full G-length permutation. Native labelweights remain INT64, maskweights/importance FP32; all PR/GT/sampling checks unchanged.

SYNTHETIC_ONLY test_N16_RAW_TARGET_MAPPING_SYNTHETIC_ONLY_20261014_03.py first RED on02 (native tuple rejected; bool/float accepted),03 two tests/all subcases GREEN CPU2/coldCUDA0.310s. Both sources AST_ONLY_OK. No actual science/cache/weights/source/GT/Git read or execution. ROOT validates actual canonical mapping against closed original parser proof before binding.
