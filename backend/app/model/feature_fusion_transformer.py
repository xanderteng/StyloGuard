"""
FeatureFusionTransformer (DualChannelIndoBERT)

Port of the Kaggle training notebook's ``DualChannelIndoBERT`` model.  The
architecture fuses:

1. **Text branch** – CLS token + masked-mean pooling from an IndoBERT backbone,
   projected through a ``text_fusion`` layer.
2. **Stylometry branch** – 52 hand-crafted stylometric features projected through
   a small FFNN (128 → 64).

Both branches are concatenated and fed into a final classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class FeatureFusionTransformer(nn.Module):
    """
    Dual-channel classifier:
      • IndoBERT text encoder  →  text_fusion (hidden*2 → 256)
      • Stylometric FFNN       →  sty_branch  (N_sty → 128 → 64)
      • Concatenation           →  classifier  (320 → 128 → num_classes)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_sty_features: int,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden = self.backbone.config.hidden_size  # 768 for indobert-base-p1

        self.sty_branch = nn.Sequential(
            nn.Linear(num_sty_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stylometry: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if return_attentions:
            self.backbone.config.output_attentions = True
            
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
        )
        last_hidden = outputs.last_hidden_state

        cls_vec = last_hidden[:, 0, :]
        sty_vec = self.sty_branch(stylometry)

        fused = torch.cat([cls_vec, sty_vec], dim=1)
        logits = self.classifier(fused)

        if return_attentions:
            return logits, outputs.attentions
        return logits
