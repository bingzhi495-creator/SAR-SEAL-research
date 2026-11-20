# Closed stock serialization readback correction

The first independent reader stopped when it compared reconstructed stock JSON bytes against captured stock bytes. The continuous query stream sorts nested JSON object fields. The original native stock encoder preserves insertion order. The36 copied CPU cache rows already used sorted RLE object fields, while the73 new native rows used the original author RLE insertion order.

An exact closed readback checked all109 original stock records against the reconstructed query fields. Every image ID, candidate count, native rank, score, category, binary compressed RLE and native canvas field was identical. Serialized ordering differed for exactly rows37–109. No scientific payload or original execution changed.

The separate corrected reader compares these exact parsed fields after verifying the unchanged sealed stock byte digest. Its native candidate linkage, original closure, probability-file hashes, independent raw-logit raster replay, image identity, full population census and all original gates remain. DEV mode additionally retains its exact historical whole-stream byte digest requirement. The first original failed reader and its source remain frozen.

