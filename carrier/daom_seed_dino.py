# -*- coding: utf-8 -*-
"""DAOM-Seed: Density-Adaptive One-to-Many seeded auxiliary queries for DINO.

Training-only, non-displacing extension of SeedDINO. The proven defect of
SeedDINO is that ``pre_decoder`` REPLACES the last-k matching reference points
with evidence seed boxes (``ref[b, -k:] = seeds``), which DISPLACES good
learnable queries and caused a recall@0.5 -3.5 regression. DAOM-Seed fixes this
by adding a SEPARATE auxiliary query group that is appended AFTER the existing
``[dn ‖ matching]`` queries (so total = ``[dn ‖ matching ‖ aux]``), injecting the
seeds NON-DISPLACINGLY into the AUX reference points only, supervising the aux
group with a one-to-many (Group-DETR / H-DETR style) loss, and DROPPING the aux
group entirely at inference. The test-time computation graph is therefore
byte-identical to the baseline DINO (strongest possible capacity control) and no
new learnable parameters are introduced (the aux content / position embeddings
reuse the matching ``query_embedding`` and the encoder ``memory`` exactly as the
matching group does).

Everything is gated on ``self.training``:
  * eval / predict / _forward -> pure parent DINO path, no seeds, no aux.
  * loss (train)              -> aux group built, o2m aux loss added.

The primary (matching) loss and the dn loss are left byte-identical to baseline
DINO: the aux columns are sliced off the decoder outputs BEFORE the standard
head loss runs, so there is zero PSeg-regression risk on the primary objective.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Union

import torch
from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList

from carrier.seed_dino import SeedDINO


@MODELS.register_module()
class DAOMSeedDINO(SeedDINO):
    """DINO + density-adaptive one-to-many seeded auxiliary queries (train-only).

    Args:
        aux_num_queries (int): Number of auxiliary queries appended after the
            ``[dn ‖ matching]`` block during training. Defaults to 300.
        k_o2m (int): One-to-many factor. Each GT is repeated ``k_o2m`` times
            before Hungarian assignment so up to ``k_o2m`` aux queries can match
            one GT (Group-DETR / H-DETR trick). Defaults to 6.
        lambda_aux (float): Scalar weight applied to every aux loss term.
            Defaults to 1.0.
        density_alpha (float): Density-adaptive seed budget multiplier;
            ``k_seed = clip(round(density_alpha * n_evid), k_min, k_max)``.
            Defaults to 1.0.
        k_min (int): Lower clip for the per-image seed budget. Defaults to 8.
        k_max (int): Upper clip for the per-image seed budget. Defaults to 200.
    """

    def __init__(self,
                 *args,
                 aux_num_queries: int = 300,
                 k_o2m: int = 6,
                 lambda_aux: float = 1.0,
                 density_alpha: float = 1.0,
                 k_min: int = 8,
                 k_max: int = 200,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_num_queries = int(aux_num_queries)
        self.k_o2m = int(k_o2m)
        self.lambda_aux = float(lambda_aux)
        self.density_alpha = float(density_alpha)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        # number of matching queries; aux reuses the same learnable embedding.
        assert self.aux_num_queries <= self.num_queries, (
            'aux_num_queries must be <= num_queries because the aux group '
            'reuses the matching query_embedding rows (no new parameters).')
        # Per-forward scratch state, populated in pre_decoder (train only).
        self._aux_meta: Dict = {}

    # ------------------------------------------------------------------ #
    #  pre_decoder: build [dn ‖ matching ‖ aux] and the block-diag mask    #
    # ------------------------------------------------------------------ #
    def pre_decoder(self, memory, memory_mask, spatial_shapes,
                    batch_data_samples=None):
        # NOTE: deliberately bypass SeedDINO.pre_decoder (which DISPLACES the
        # matching tail with seeds). We call DINO.pre_decoder (grandparent) so
        # the matching + dn groups are exactly the baseline ones, then append a
        # fresh aux group ourselves. This is what makes eval byte-identical.
        decoder_inputs_dict, head_inputs_dict = DINO.pre_decoder(
            self, memory, memory_mask, spatial_shapes, batch_data_samples)

        self._aux_meta = {}
        if not self.training or self.aux_num_queries <= 0 \
                or batch_data_samples is None:
            # eval / predict: pure parent path, no aux group, no seeds.
            return decoder_inputs_dict, head_inputs_dict

        query = decoder_inputs_dict['query']                 # (bs, n_dn+nq, d)
        reference_points = decoder_inputs_dict['reference_points']  # sigmoided
        dn_mask = decoder_inputs_dict['dn_mask']             # (L0, L0) bool
        bs, n_existing, d = query.shape
        n_aux = self.aux_num_queries
        device = query.device

        # --- aux content queries: reuse the matching query_embedding rows --- #
        # (no new nn.Parameter; identical init source as the matching group)
        aux_content = self.query_embedding.weight[:n_aux, None, :]
        aux_content = aux_content.repeat(1, bs, 1).transpose(0, 1)  # (bs,n_aux,d)

        # --- aux reference points: standard learnable init, then seeds --- #
        # Baseline learnable matching ref points come from encoder topk
        # (already in `reference_points`'s matching slice). For aux we reuse the
        # *matching* learnable reference points (top-k encoder proposals) as the
        # standard init, then overwrite the first k_seed with evidence seeds.
        n_matching = self.num_queries
        matching_ref = reference_points[:, -n_matching:, :]        # (bs,nq,4)
        aux_ref = matching_ref[:, :n_aux, :].clone()               # (bs,n_aux,4)

        n_evid_list: List[int] = []
        for b in range(bs):
            seeds = self._seed_boxes_norm(batch_data_samples[b], device)
            n_evid = 0 if seeds is None else int(seeds.shape[0])
            n_evid_list.append(n_evid)
            if seeds is None or n_evid == 0:
                continue  # density off / no evidence -> pure learnable aux init
            # density-adaptive per-image seed budget
            k_seed = int(round(self.density_alpha * n_evid))
            k_seed = max(self.k_min, min(self.k_max, k_seed))
            k_seed = min(k_seed, n_aux, n_evid)
            if k_seed > 0:
                aux_ref[b, :k_seed, :] = seeds[:k_seed]

        # --- concatenate aux group AFTER [dn ‖ matching] --- #
        query = torch.cat([query, aux_content], dim=1)
        reference_points = torch.cat([reference_points, aux_ref], dim=1)

        # --- extend the self-attention mask to be block-diagonal --- #
        attn_mask = self._extend_attn_mask(dn_mask, n_existing, n_aux, device)

        decoder_inputs_dict['query'] = query
        decoder_inputs_dict['reference_points'] = reference_points
        decoder_inputs_dict['dn_mask'] = attn_mask

        # stash for the loss split. `n_existing` = num_dn + num_matching.
        self._aux_meta = dict(
            n_existing=n_existing,
            n_aux=n_aux,
            n_seed_per_img=n_evid_list)
        return decoder_inputs_dict, head_inputs_dict

    @staticmethod
    def _extend_attn_mask(dn_mask, n_existing: int, n_aux: int,
                          device) -> Tensor:
        """Build the (L, L) block-diagonal self-attention mask.

        Convention (verified against mmdet ``generate_dn_mask``):
        ``True`` == masked (cannot attend), ``False`` == can attend.

        Layout, L = n_existing + n_aux where n_existing = num_dn + num_matching::

            cols ->        [ dn | matching | aux ]
            [ dn       ]   [    dn_mask    |  T  ]
            [ matching ]   [    (existing) |  T  ]
            [ aux      ]   [   T      T    |  F* ]

        The top-left (n_existing x n_existing) block is the ORIGINAL dn_mask,
        copied verbatim (preserves dn<->matching blocking exactly). The aux
        block-row is masked against dn+matching (aux cannot see them) and the
        existing rows are masked against aux (dn+matching cannot see aux). The
        aux<->aux block is fully visible (False), i.e. aux attends only within
        aux -- a single one-to-many group.
        """
        L = n_existing + n_aux
        attn_mask = torch.zeros(L, L, dtype=torch.bool, device=device)
        if dn_mask is not None:
            # copy the original dn<->matching mask into the top-left block
            attn_mask[:n_existing, :n_existing] = dn_mask.to(device=device,
                                                             dtype=torch.bool)
        # existing (dn+matching) rows must NOT attend to aux columns
        attn_mask[:n_existing, n_existing:] = True
        # aux rows must NOT attend to dn+matching columns
        attn_mask[n_existing:, :n_existing] = True
        # aux<->aux stays False (visible) -> aux attends only within aux
        return attn_mask

    # ------------------------------------------------------------------ #
    #  loss: standard primary+dn loss (aux-free) + separate o2m aux loss   #
    # ------------------------------------------------------------------ #
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)

        aux_meta = self._aux_meta
        if not aux_meta:
            # no aux group was built (aux disabled) -> exact parent loss.
            return self.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples)

        n_existing = aux_meta['n_existing']

        # ---- split the decoder outputs: [existing | aux] ---- #
        # hidden_states: (num_layers, bs, L, d); references: list of (bs, L, 4)
        hidden_states = head_inputs_dict['hidden_states']
        references = head_inputs_dict['references']

        primary_hs = hidden_states[:, :, :n_existing, :]
        aux_hs = hidden_states[:, :, n_existing:, :]
        primary_refs = [r[:, :n_existing, :] for r in references]
        aux_refs = [r[:, n_existing:, :] for r in references]

        # ---- primary + dn loss: byte-identical to baseline DINO ---- #
        primary_inputs = dict(head_inputs_dict)
        primary_inputs['hidden_states'] = primary_hs
        primary_inputs['references'] = primary_refs
        losses = self.bbox_head.loss(
            **primary_inputs, batch_data_samples=batch_data_samples)

        # ---- separate one-to-many aux loss ---- #
        aux_losses = self._loss_aux(aux_hs, aux_refs, batch_data_samples)
        for k, v in aux_losses.items():
            losses[f'aux.{k}'] = v
        return losses

    def _loss_aux(self, aux_hidden_states: Tensor, aux_references: List[Tensor],
                  batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """One-to-many auxiliary loss on the aux decoder outputs.

        Reuses the head's existing matching machinery (forward + DETRHead
        ``loss_by_feat`` via the HungarianAssigner). The one-to-many behaviour is
        obtained by REPEATING each GT ``k_o2m`` times before assignment, so the
        assigner can match up to ``k_o2m`` aux queries to one GT (Group-DETR /
        H-DETR). No new loss function is written. The aux group carries no dn and
        no encoder loss (``dn_meta=None``, ``enc_*=None``), so the head runs only
        the matching branch on the aux outputs.
        """
        # build o2m-repeated gt instances and matching img metas
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt = data_sample.gt_instances
            if self.k_o2m > 1 and len(gt) > 0:
                rep = InstanceListRepeat(gt, self.k_o2m)
                batch_gt_instances.append(rep)
            else:
                batch_gt_instances.append(gt)

        # head forward on aux outputs -> per-layer cls scores + bbox preds
        cls_scores, bbox_preds = self.bbox_head(aux_hidden_states, aux_references)

        # Call DETRHead.loss_by_feat directly (matching-only) -- the SAME call
        # DINOHead makes internally via super(DeformableDETRHead, self). This
        # bypasses split_outputs / dn / enc logic entirely (DINOHead.loss_by_feat
        # would dereference dn_meta['...'] before its None-check and crash). The
        # aux group thus gets a pure Hungarian one-to-one loss over the
        # k_o2m-repeated GT set == one-to-many over the original GTs.
        loss_dict = DETRHead.loss_by_feat(
            self.bbox_head,
            cls_scores,
            bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        # weight by lambda_aux
        if self.lambda_aux != 1.0:
            loss_dict = {k: v * self.lambda_aux for k, v in loss_dict.items()}
        return loss_dict


def InstanceListRepeat(gt_instances, k: int):
    """Return a new InstanceData with every field repeated ``k`` times.

    Used to build the one-to-many target set: with each GT repeated ``k`` times,
    the HungarianAssigner produces a one-to-one matching over the repeated set,
    which is effectively a one-to-(<=k) matching over the original GTs.
    """
    new = copy.copy(gt_instances)  # shallow clone of InstanceData container
    out = gt_instances.__class__()
    for key in gt_instances.keys():
        val = gt_instances[key]
        if isinstance(val, Tensor):
            # repeat along dim 0: (n, ...) -> (n*k, ...)
            reps = [k] + [1] * (val.dim() - 1)
            out[key] = val.repeat(*reps)
        else:
            # bbox structures (e.g. HorizontalBoxes) expose .tensor / cat
            try:
                out[key] = val.cat([val for _ in range(k)])
            except Exception:
                out[key] = val
    del new
    return out
