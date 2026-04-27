"""Leave-one-out word-importance scoring shared by both attacks.

The score for position ``i`` is

    (orig_prob - new_prob_for_orig_label)
    + (label_changed) * (new_top_prob - orig_prob_for_new_top)

which rewards positions whose removal both lowers the original-label
confidence and pushes the prediction away from the original label.

This consolidates the two near-duplicate copies that previously lived in
``textfooler/main.py`` and ``bert_attack/main.py``.
"""

from __future__ import annotations

import numpy as np
import torch


def importance_scores_from_probs(
    orig_prob: torch.Tensor,
    orig_label: int | torch.Tensor,
    orig_probs: torch.Tensor,
    leave_1_probs: torch.Tensor,
) -> np.ndarray:
    """Compute per-candidate importance from leave-one-out predictions.

    Args:
        orig_prob:    scalar tensor — orig_probs.max().
        orig_label:   index tensor / int — argmax of orig_probs.
        orig_probs:   1-D tensor of original-text class probabilities.
        leave_1_probs: 2-D tensor (n_positions, n_classes) of probabilities
                       after leaving each candidate token out.

    Returns:
        numpy array of shape (n_positions,).
    """
    leave_1_argmax = torch.argmax(leave_1_probs, dim=-1)
    scores = (
        orig_prob
        - leave_1_probs[:, orig_label]
        + (leave_1_argmax != orig_label).float()
        * (
            leave_1_probs.max(dim=-1)[0]
            - torch.index_select(orig_probs, 0, leave_1_argmax)
        )
    )
    return scores.detach().cpu().numpy()
