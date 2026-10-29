"""Concerto base official encoder with stage-level PARTIAL unfreezing + binary head.

Same design as the verified frozen probe, extended for partial fine-tuning:

- Encoder: SAME official checkpoint loader (concerto.model.load) with the
  SAME custom_config as the frozen probe — the encoder itself is not
  re-implemented.
- Unfreeze rule is derived from the official PointTransformerV3 structure
  (Concerto-main/concerto/model.py): stage modules are named
  `enc.enc{s}` (s = 0..4); a stage holds `down` (GridPooling, s>0) and
  `block{i}`. We select parameters by the name prefix `enc.enc{s}.` for
  s in config_finetune.UNFREEZE_STAGES. Everything else (embedding,
  enc.enc0..3) keeps requires_grad=False.
- Autograd note: with frozen params requires_grad=False and input tensors
  requires_grad=False, frozen stages are NOT recorded in the autograd graph,
  so their activations are not retained (same memory behavior as the frozen
  probe's no_grad, but gradients flow through the unfrozen stage).
- Frozen stages are forced to eval() even in training mode (see train()
  below): they contain LayerNorm (no running stats to corrupt) but also
  DropPath / attention dropout, which must stay OFF to keep frozen features
  identical to the frozen-probe behavior. Unfrozen stages follow the
  wrapper's training mode.
- Head: identical MLP to the frozen probe, randomly re-initialized.
- Upcast: identical to the frozen probe (official README two-concat +
  replace unpools).
"""
import torch
import torch.nn as nn

import concerto  # made importable by bootstrap.py before this module loads

import config as C
import config_finetune as CF


class ConcertoPartialFinetuneSeg(nn.Module):
    def __init__(self, unfreeze_stages=None):
        super().__init__()
        stages = tuple(unfreeze_stages if unfreeze_stages is not None
                       else CF.UNFREEZE_STAGES)
        assert all(isinstance(s, int) and 0 <= s <= 4 for s in stages), stages
        self.unfreeze_stages = stages
        self.unfreeze_prefixes = tuple(f"enc.enc{s}." for s in stages)

        self.encoder = concerto.model.load(
            C.CONCERTO_CKPT, custom_config=C.CONCERTO_CUSTOM_CONFIG)

        # ---- freeze / unfreeze by verified module naming ------------------
        n_enc_total = 0
        self.frozen_param_names = []
        self.trainable_enc_param_names = []
        for name, p in self.encoder.named_parameters():
            n_enc_total += p.numel()
            if name.startswith(self.unfreeze_prefixes):
                p.requires_grad_(True)
                self.trainable_enc_param_names.append(name)
            else:
                p.requires_grad_(False)
                self.frozen_param_names.append(name)
        assert self.trainable_enc_param_names, "unfreeze rule matched nothing"
        self.n_encoder = n_enc_total
        ps = dict(self.encoder.named_parameters())
        self.n_frozen = sum(ps[n].numel() for n in self.frozen_param_names)

        # frozen stage submodules (for the eval() override in train())
        self._frozen_stage_mods = [self.encoder.embedding]
        for s in range(self.encoder.num_stages):
            if s not in stages:
                self._frozen_stage_mods.append(getattr(self.encoder.enc,
                                                       f"enc{s}"))

        assert CF.HEAD_INIT == "random"
        self.head = nn.Sequential(
            nn.Linear(C.UPCAST_DIM, C.HEAD_HIDDEN), nn.GELU(),
            nn.Dropout(C.HEAD_DROPOUT), nn.Linear(C.HEAD_HIDDEN, C.NUM_CLASSES),
        )

    # ------------------------------------------------------------------ io --
    def train(self, mode=True):
        super().train(mode)
        # frozen parts stay eval(): DropPath / attn dropout off, features
        # identical to the frozen probe; unfrozen stages follow `mode`
        for m in self._frozen_stage_mods:
            m.eval()
        return self

    def trainable_encoder_parameters(self):
        ps = dict(self.encoder.named_parameters())
        return [ps[n] for n in self.trainable_enc_param_names]

    def unfrozen_state_dict(self):
        """State of the unfrozen encoder params only (for checkpoints)."""
        ps = dict(self.encoder.named_parameters())
        return {n: ps[n].detach().cpu() for n in self.trainable_enc_param_names}

    def load_unfrozen_state_dict(self, sd):
        ps = dict(self.encoder.named_parameters())
        for n, v in sd.items():
            assert n in ps, f"unexpected param {n}"
            ps[n].data.copy_(v)

    def forward(self, point):
        # NO torch.no_grad here: gradients must reach the unfrozen stage.
        # Frozen stages are excluded from the autograd graph automatically
        # (no input requires grad), so no memory regression vs the probe.
        point = self.encoder(point)
        # official point-feature upcast (Sonata README), same as the probe
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        logits = self.head(point.feat)
        return logits, point
