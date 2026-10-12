"""Unused parameter-count control; injected Torch and immutable readout class.

SOURCE_IMPLEMENTED_NUMERICAL_VALIDATION_PENDING. No training/qualification I/O.
The second 256->16 projection uses spatial F, not the final mask embedding e0.
This supplies 32 local feature dimensions, adds a direct F-gradient route and
extra projection/render compute. Equal 8 tensors/9441 scalars do not establish
equal effective capacity or compute. Native L/p/B/contrast remain query-dependent.
"""

VALIDATION_STATUS = 'SOURCE_IMPLEMENTED_NUMERICAL_VALIDATION_PENDING'


def build_adapter(torch, readout_class, *, seed, kappa=1.0, eps=1e-6):
    """Inherit original CPU/private-RNG constructor, names and pointwise MLP.

    Caller must bind the original QueryPixelPosteriorReadout SOURCE. No native
    model, source loading, matcher, sampling, backward or optimizer lives here.
    The mathematical projection helper permits positive low-resolution grids;
    the real native caller remains responsible for its exact 200x200 geometry.
    """
    functional = torch.nn.functional

    class PixelCapacityControl(readout_class):
        def project_queries(self, features, e0):
            """Reused query_projection parameters -> spatial [B,16,Hf,Wf].

            e0 is retained and validated, but has no direct computational path.
            Its indirect query role through caller-provided native M is retained.
            """
            reference = self._reference()
            self._tensor('features', features, 4, reference)
            self._tensor('e0', e0, 3, reference)
            if features.shape[1] != 256 or any(v <= 0 for v in features.shape):
                raise ValueError('Positive [B,256,Hf,Wf] features required')
            if tuple(e0.shape) != (features.shape[0], 100, 256):
                raise ValueError('Original e0[B,100,256] required')
            result = self.query_projection(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            self._finite('second spatial projection', result)
            return result

        def project_native_fields(self, F256, e0, M200, actual_img_shape, original_shape, renderer):
            """Render two learned F16 maps and native M; preserve ambient graph.

            renderer(value,identity) returns [C,Horig,Worig] for native batch1.
            It must be the caller's same linear native renderer for both maps.
            No F/e0/M detach or in-place modification is performed.
            """
            reference = self._reference()
            self._tensor('F256', F256, 4, reference)
            self._tensor('M200', M200, 4, reference)
            if F256.shape[0] != 1 or tuple(M200.shape) != (1, 100, *F256.shape[-2:]):
                raise ValueError('Aligned batch1 native F/M grids required')
            for label, dims in (('actual_img_shape', actual_img_shape), ('original_shape', original_shape)):
                if not isinstance(dims, (tuple, list)) or len(dims) != 2 or any(type(v) is not int or v <= 0 for v in dims):
                    raise ValueError(label+' must contain two positive integers')
            if any(v > 800 for v in actual_img_shape) or not callable(renderer):
                raise ValueError('Native crop <=800 and caller renderer required')
            identity = dict(actual_img_shape=list(actual_img_shape), original_shape=list(original_shape))
            first = renderer(self.project_features(F256), identity).unsqueeze(0)
            second = renderer(self.project_queries(F256, e0), identity).unsqueeze(0)
            native = renderer(M200, identity).unsqueeze(0)
            for name, value, channels in (('first', first, 16), ('second', second, 16), ('native', native, 100)):
                self._tensor(name, value, 4, reference)
                if tuple(value.shape) != (1, channels, *original_shape):
                    raise ValueError('Renderer original shape differs: '+name)
            probability = native.sigmoid()
            mean = functional.avg_pool2d(probability[0, :, None], 3, stride=1, padding=1,
                                         count_include_pad=False)[:, 0]
            contrast = probability-mean.unsqueeze(0)
            self._finite('contrast', contrast)
            return dict(projected_pixel=first, projected_second_pixel=second,
                        native_logits=native, contrast=contrast)

        def point_outputs(self, pixel, second_pixel, native_logits, contrast):
            """Both [S,16] blocks come from the same original integer pixels."""
            return super().point_outputs(pixel, second_pixel, native_logits, contrast)

        def matched_point_loss(self, fields, query_ids, gt_masks, point_indices):
            """BCE(L+u), 12544 replacement points/GT, then GT mean per image."""
            if not isinstance(fields, dict) or set(fields) != {'projected_pixel', 'projected_second_pixel', 'native_logits', 'contrast'}:
                raise ValueError('Exact spatial-control fields required')
            reference = self._reference()
            first, second = fields['projected_pixel'], fields['projected_second_pixel']
            native, contrast = fields['native_logits'], fields['contrast']
            if not isinstance(first, torch.Tensor) or first.ndim != 4 or any(v <= 0 for v in first.shape):
                raise ValueError('Positive original pixel grid required')
            h, w = first.shape[-2:]
            for name, value, shape in (('first', first, (1,16,h,w)), ('second', second, (1,16,h,w)),
                                       ('native', native, (1,100,h,w)), ('contrast', contrast, (1,100,h,w))):
                self._tensor(name, value, 4, reference)
                if tuple(value.shape) != shape: raise ValueError('Original field shape differs: '+name)
            if not isinstance(query_ids, torch.Tensor) or query_ids.layout != torch.strided or query_ids.dtype != torch.int64 or query_ids.ndim != 1 or query_ids.device != reference.device:
                raise ValueError('Shared-device int64[G] query IDs required')
            count = query_ids.numel()
            if count > 100 or bool(((query_ids < 0)|(query_ids >= 100)).any()) or torch.unique(query_ids).numel() != count:
                raise ValueError('Unique native query IDs in [0,99] required')
            if not isinstance(gt_masks, torch.Tensor) or gt_masks.layout != torch.strided or gt_masks.dtype != torch.bool or gt_masks.device != reference.device or tuple(gt_masks.shape) != (count,h,w):
                raise ValueError('Aligned original bool[G,H,W] GT required')
            if not isinstance(point_indices, torch.Tensor) or point_indices.layout != torch.strided or point_indices.dtype != torch.int64 or point_indices.device != reference.device or tuple(point_indices.shape) != (count,12544):
                raise ValueError('Original int64[G,12544] points required')
            if bool(((point_indices < 0)|(point_indices >= h*w)).any()):
                raise ValueError('Points must index complete original raster')
            if count == 0:
                return sum((v[..., :0].sum() for v in (*fields.values(), *self.parameters())), reference.new_zeros(()))
            indices = point_indices.reshape(-1)
            first_points = first[0].flatten(1).t()[indices]
            second_points = second[0].flatten(1).t()[indices]
            native_points = native[0,query_ids].flatten(1).gather(1, point_indices).reshape(-1)
            contrast_points = contrast[0,query_ids].flatten(1).gather(1, point_indices).reshape(-1)
            targets = gt_masks.flatten(1).gather(1, point_indices).to(reference.dtype)
            logits = self.point_outputs(first_points, second_points, native_points, contrast_points)['posterior_logits'].reshape(count,12544)
            return functional.binary_cross_entropy_with_logits(logits, targets, reduction='none').mean(1).mean()

        @torch.no_grad()
        def dense_outputs(self, pixel, second_pixel, native_logits, query_chunk_size=4):
            """Inference: second spatial term, never a per-query broadcast vector.

            Same three dense fields plus corrected-support posterior budget as
            the original module. Factorization changes FP operation grouping;
            agreement with direct concatenation requires numerical validation.
            """
            reference = self._reference()
            for name, value in (('pixel', pixel), ('second_pixel', second_pixel), ('native_logits', native_logits)):
                self._tensor(name, value, 4, reference)
            if type(query_chunk_size) is not int or query_chunk_size <= 0:
                raise ValueError('Positive non-bool query chunk required')
            batch, channels, h, w = pixel.shape
            if channels != 16 or any(v <= 0 for v in pixel.shape) or tuple(second_pixel.shape) != tuple(pixel.shape) or tuple(native_logits.shape) != (batch,100,h,w):
                raise ValueError('Two [B,16,H,W] maps and native [B,100,H,W] required')
            weight = self.hidden.weight
            first = functional.linear(pixel.permute(0,2,3,1), weight[:,:16], self.hidden.bias).permute(0,3,1,2)
            second = functional.linear(second_pixel.permute(0,2,3,1), weight[:,16:32]).permute(0,3,1,2)
            fields = {name:torch.empty_like(native_logits) for name in ('posterior','corrected_logits','bounded_delta')}
            budget = native_logits.new_empty((batch,100))
            for start in range(0,100,query_chunk_size):
                stop = min(start+query_chunk_size,100); native = native_logits[:,start:stop]
                probability = native.sigmoid(); binary = (native > 0).to(reference.dtype)
                mean = functional.avg_pool2d(probability.reshape(batch*(stop-start),1,h,w),3,stride=1,padding=1,
                                             count_include_pad=False).reshape(batch,stop-start,h,w)
                hidden = (first[:,None]+second[:,None]).expand(batch,stop-start,32,h,w).clone()
                for i, scalar in enumerate((native,probability,binary,probability-mean)):
                    hidden.addcmul_(scalar[:,:,None], weight[:,32+i].reshape(1,1,32,1,1))
                u = functional.linear(functional.silu(hidden).permute(0,1,3,4,2), self.final.weight, self.final.bias).squeeze(-1)
                block = self._outputs(native,u)
                support = (block['corrected_logits'] > 0).to(reference.dtype)
                quality = (support*block['posterior']).sum((-2,-1))/(support.sum((-2,-1))+((1-support)*block['posterior']).sum((-2,-1))+self.eps)
                self._finite('budget', quality); budget[:,start:stop] = quality
                for name in fields: fields[name][:,start:stop] = block[name]
            fields['budget'] = budget
            return fields

    result = PixelCapacityControl(seed=seed, kappa=kappa, eps=eps)
    if len(tuple(result.parameters())) != 8 or sum(p.numel() for p in result.parameters()) != 9441 or list(result.buffers()):
        raise ValueError('Original eight tensors/9441/no-buffer constructor required')
    return result
