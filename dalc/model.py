"""
DaLC GNN architecture and training loop.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from dalc.core import set_seed

# Fixed training-time constants (not user-tunable, not described in the paper).
_PATIENCE = 30                              # early-stopping patience (epochs)
_EDGE_DROPOUT_RATE = 0.02                   # per-edge dropout during training
_EDGE_DROPOUT_TARGET = "object_only"        # which edge type the dropout applies to
_TRUTH_HEAD_MODE = "easy_vs_ambiguous_vs_hard"


class DifficultyAwareCompletionGNN(nn.Module):
    def __init__(
        self,
        object_in_dim: int,
        object_numeric_dim: int,
        object_nominal_cardinalities: List[int],
        worker_in_dim: int,
        object_edge_dim: int,
        vote_edge_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        truth_head_mode: str,
        num_classes: int,
        rev_vote_edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.truth_head_mode = truth_head_mode
        self.num_classes = num_classes
        self.object_in_dim = int(object_in_dim)
        if self.object_in_dim > 0:
            self.object_main_proj = nn.Linear(object_in_dim, hidden_dim)
        else:
            self.object_main_proj = None
        self.object_numeric_dim = int(object_numeric_dim)
        self.object_nominal_cardinalities = [int(v) for v in object_nominal_cardinalities]
        branch_count = 1 if self.object_main_proj is not None else 0
        if self.object_numeric_dim > 0:
            self.object_numeric_proj = nn.Linear(self.object_numeric_dim, hidden_dim)
            branch_count += 1
        else:
            self.object_numeric_proj = None
        if self.object_nominal_cardinalities:
            self.object_nominal_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality + 1, min(16, max(4, int(round(cardinality ** 0.5)) + 1)), padding_idx=0)
                    for cardinality in self.object_nominal_cardinalities
                ]
            )
            nominal_dim = sum(embedding.embedding_dim for embedding in self.object_nominal_embeddings)
            self.object_nominal_proj = nn.Linear(nominal_dim, hidden_dim)
            branch_count += 1
        else:
            self.object_nominal_embeddings = nn.ModuleList()
            self.object_nominal_proj = None
        if branch_count == 0:
            raise ValueError("DifficultyAwareCompletionGNN requires at least one object feature branch.")
        self.object_fusion = nn.Linear(hidden_dim * branch_count, hidden_dim) if branch_count > 1 else None
        self.worker_proj = nn.Linear(worker_in_dim, hidden_dim)
        _rev_vote_edge_dim = rev_vote_edge_dim if rev_vote_edge_dim is not None else vote_edge_dim

        def make_relations() -> Dict[Tuple[str, str, str], GATv2Conv]:
            return {
                ("object", "similar", "object"): GATv2Conv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=object_edge_dim,
                    add_self_loops=False,
                    dropout=dropout,
                ),
                ("worker", "votes", "object"): GATv2Conv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=vote_edge_dim,
                    add_self_loops=False,
                    dropout=dropout,
                ),
                ("object", "rev_votes", "worker"): GATv2Conv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=_rev_vote_edge_dim,
                    add_self_loops=False,
                    dropout=dropout,
                ),
            }

        self.conv1 = HeteroConv(make_relations(), aggr="sum")
        self.conv2 = HeteroConv(make_relations(), aggr="sum")
        self.norm1 = nn.ModuleDict({"object": nn.LayerNorm(hidden_dim), "worker": nn.LayerNorm(hidden_dim)})
        self.norm2 = nn.ModuleDict({"object": nn.LayerNorm(hidden_dim), "worker": nn.LayerNorm(hidden_dim)})

        def make_truth_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        if truth_head_mode == "easy_vs_rest":
            self.truth_head_easy = make_truth_head()
            self.truth_head_rest = make_truth_head()
        elif truth_head_mode == "easy_vs_ambiguous_vs_hard":
            self.truth_head_easy = make_truth_head()
            self.truth_head_ambiguous = make_truth_head()
            self.truth_head_hard = make_truth_head()
        else:
            self.truth_head = make_truth_head()

        self.mv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.completion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        object_parts = []
        if self.object_main_proj is not None:
            object_parts.append(F.gelu(self.object_main_proj(x_dict["object"])))
        if self.object_numeric_proj is not None and "object_numeric" in x_dict:
            object_parts.append(F.gelu(self.object_numeric_proj(x_dict["object_numeric"])))
        if self.object_nominal_proj is not None and "object_nominal" in x_dict:
            nominal_inputs = x_dict["object_nominal"].long()
            nominal_embeds = [
                embedding(nominal_inputs[:, idx])
                for idx, embedding in enumerate(self.object_nominal_embeddings)
            ]
            nominal_cat = torch.cat(nominal_embeds, dim=-1) if nominal_embeds else None
            if nominal_cat is not None:
                object_parts.append(F.gelu(self.object_nominal_proj(nominal_cat)))
        object_hidden = (
            object_parts[0]
            if len(object_parts) == 1
            else F.gelu(self.object_fusion(torch.cat(object_parts, dim=-1)))
        )
        h = {
            "object": F.dropout(object_hidden, p=self.dropout, training=self.training),
            "worker": F.dropout(F.gelu(self.worker_proj(x_dict["worker"])), p=self.dropout, training=self.training),
        }
        out1 = self.conv1(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h = {
            node_type: self.norm1[node_type](
                h[node_type] + F.dropout(F.gelu(out1[node_type]), p=self.dropout, training=self.training)
            )
            for node_type in h
        }
        out2 = self.conv2(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h = {
            node_type: self.norm2[node_type](
                h[node_type] + F.dropout(F.gelu(out2[node_type]), p=self.dropout, training=self.training)
            )
            for node_type in h
        }
        return h

    def decode_truth(self, object_h: torch.Tensor, object_bucket_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.truth_head_mode == "easy_vs_rest":
            if object_bucket_id is None:
                raise ValueError("object_bucket_id is required when truth_head_mode='easy_vs_rest'")
            easy_logit = self.truth_head_easy(object_h)
            rest_logit = self.truth_head_rest(object_h)
            return torch.where((object_bucket_id == 0).unsqueeze(-1), easy_logit, rest_logit)
        if self.truth_head_mode == "easy_vs_ambiguous_vs_hard":
            if object_bucket_id is None:
                raise ValueError("object_bucket_id is required when truth_head_mode='easy_vs_ambiguous_vs_hard'")
            easy_logit = self.truth_head_easy(object_h)
            amb_logit = self.truth_head_ambiguous(object_h)
            hard_logit = self.truth_head_hard(object_h)
            return torch.where(
                (object_bucket_id == 0).unsqueeze(-1),
                easy_logit,
                torch.where((object_bucket_id == 1).unsqueeze(-1), amb_logit, hard_logit),
            )
        return self.truth_head(object_h)

    def decode_completion(
        self,
        object_h: torch.Tensor,
        worker_h: torch.Tensor,
        worker_idx: torch.Tensor,
        object_idx: torch.Tensor,
    ) -> torch.Tensor:
        pair_h = torch.cat([worker_h[worker_idx], object_h[object_idx]], dim=-1)
        return self.completion_head(pair_h)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
        object_bucket_id: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        truth_logit = self.decode_truth(h["object"], object_bucket_id=object_bucket_id)
        mv_logit = self.mv_head(h["object"])
        return truth_logit, mv_logit, h


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return -(target * log_prob).sum(dim=-1)


def safe_mean(loss: torch.Tensor) -> torch.Tensor:
    if loss.numel() == 0:
        return loss.new_tensor(0.0)
    return loss.mean()


def maybe_apply_edge_dropout(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    edge_dropout_rate: float,
    edge_dropout_target: str = "shared",
) -> Tuple[Dict[Tuple[str, str, str], torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
    if edge_dropout_rate <= 0.0:
        return edge_index_dict, edge_attr_dict

    def apply_mask(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.shape[1] == 0:
            return edge_index, edge_attr
        if not bool(mask.any()):
            keep_idx = torch.randint(edge_index.shape[1], (1,), device=edge_index.device)
            mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
            mask[keep_idx] = True
        return edge_index[:, mask], edge_attr[mask]

    out_edge_index = dict(edge_index_dict)
    out_edge_attr = dict(edge_attr_dict)

    obj_key = ("object", "similar", "object")
    obj_edges = edge_index_dict[obj_key]
    apply_object_dropout = edge_dropout_target in {"shared", "object_only"}
    if apply_object_dropout and obj_edges.shape[1] > 0:
        obj_mask = torch.rand(obj_edges.shape[1], device=obj_edges.device) >= edge_dropout_rate
        out_edge_index[obj_key], out_edge_attr[obj_key] = apply_mask(edge_index_dict[obj_key], edge_attr_dict[obj_key], obj_mask)

    vote_key = ("worker", "votes", "object")
    rev_vote_key = ("object", "rev_votes", "worker")
    vote_edges = edge_index_dict[vote_key]
    apply_vote_dropout = edge_dropout_target in {"shared", "vote_only"}
    if apply_vote_dropout and vote_edges.shape[1] > 0:
        vote_mask = torch.rand(vote_edges.shape[1], device=vote_edges.device) >= edge_dropout_rate
        out_edge_index[vote_key], out_edge_attr[vote_key] = apply_mask(edge_index_dict[vote_key], edge_attr_dict[vote_key], vote_mask)
        out_edge_index[rev_vote_key], out_edge_attr[rev_vote_key] = apply_mask(
            edge_index_dict[rev_vote_key],
            edge_attr_dict[rev_vote_key],
            vote_mask,
        )
    return out_edge_index, out_edge_attr


def train(
    data: HeteroData,
    args,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    # DaLC objective: the MV auxiliary head + the observed/missing completion
    # losses. (Truth-pseudo-target and consistency terms were tested during
    # development but did not help on the paper datasets, so they are not part
    # of the objective.) All ablation variants share this loss structure --
    # they differ only in the NCS / worker-reliability / edge-feature
    # mechanisms above this layer.
    object_x = data["object"].x.to(device)
    object_numeric_x = data["object"].x_numeric.to(device) if hasattr(data["object"], "x_numeric") else object_x.new_zeros((object_x.shape[0], 0))
    object_nominal_x = data["object"].x_nominal.to(device) if hasattr(data["object"], "x_nominal") else torch.zeros((object_x.shape[0], 0), dtype=torch.long, device=device)
    worker_x = data["worker"].x.to(device)
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    edge_attr_dict = {
        ("object", "similar", "object"): data["object", "similar", "object"].edge_attr.to(device),
        ("worker", "votes", "object"): data["worker", "votes", "object"].edge_attr.to(device),
        ("object", "rev_votes", "worker"): data["object", "rev_votes", "worker"].edge_attr.to(device),
    }
    vote_edge_dim = edge_attr_dict[("worker", "votes", "object")].shape[1]
    rev_vote_edge_dim = edge_attr_dict[("object", "rev_votes", "worker")].shape[1]
    model = DifficultyAwareCompletionGNN(
        object_in_dim=object_x.shape[1],
        object_numeric_dim=object_numeric_x.shape[1],
        object_nominal_cardinalities=data.object_nominal_cardinalities.cpu().tolist() if hasattr(data, "object_nominal_cardinalities") else [],
        worker_in_dim=worker_x.shape[1],
        object_edge_dim=edge_attr_dict[("object", "similar", "object")].shape[1],
        vote_edge_dim=vote_edge_dim,
        rev_vote_edge_dim=rev_vote_edge_dim if rev_vote_edge_dim != vote_edge_dim else None,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        truth_head_mode=_TRUTH_HEAD_MODE,
        num_classes=int(data.num_classes),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    aux_loss_fn = nn.CrossEntropyLoss()
    observed_ce = nn.CrossEntropyLoss()

    train_mask = data["object"].train_mask.to(device)
    val_mask = data["object"].val_mask.to(device)
    y_majority = data["object"].y_majority.to(device)
    bucket_id = data["object"].bucket_id.to(device)

    observed_worker_idx = data.observed_worker_idx.to(device)
    observed_object_idx = data.observed_object_idx.to(device)
    observed_label = data.observed_label.to(device)
    observed_train_mask = train_mask[observed_object_idx]
    observed_val_mask = val_mask[observed_object_idx]

    missing_worker_idx = data.missing_worker_idx.to(device)
    missing_object_idx = data.missing_object_idx.to(device)
    missing_q_init = data.missing_q_init.to(device)
    missing_train_mask = train_mask[missing_object_idx]
    missing_val_mask = val_mask[missing_object_idx]

    best_val = float("inf")
    best_state = None
    patience_left = _PATIENCE
    x_dict = {
        "object": object_x,
        "object_numeric": object_numeric_x,
        "object_nominal": object_nominal_x,
        "worker": worker_x,
    }

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_edge_index_dict, train_edge_attr_dict = maybe_apply_edge_dropout(
            edge_index_dict,
            edge_attr_dict,
            edge_dropout_rate=_EDGE_DROPOUT_RATE,
            edge_dropout_target=_EDGE_DROPOUT_TARGET,
        )
        truth_logit, mv_logit, hidden = model(
            x_dict,
            train_edge_index_dict,
            train_edge_attr_dict,
            object_bucket_id=bucket_id,
        )
        obs_logit = model.decode_completion(hidden["object"], hidden["worker"], observed_worker_idx, observed_object_idx)
        missing_logit = model.decode_completion(hidden["object"], hidden["worker"], missing_worker_idx, missing_object_idx)

        loss_aux = aux_loss_fn(mv_logit[train_mask], y_majority[train_mask])
        if bool(observed_train_mask.any()):
            loss_observed = observed_ce(obs_logit[observed_train_mask], observed_label[observed_train_mask])
        else:
            loss_observed = truth_logit.new_tensor(0.0)
        loss_prior = safe_mean(soft_cross_entropy(missing_logit[missing_train_mask], missing_q_init[missing_train_mask]))
        loss = (
            args.aux_weight * loss_aux
            + args.completion_observed_weight * loss_observed
            + args.completion_prior_weight * loss_prior
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _truth_logit, mv_logit, hidden = model(x_dict, edge_index_dict, edge_attr_dict, object_bucket_id=bucket_id)
            obs_logit = model.decode_completion(hidden["object"], hidden["worker"], observed_worker_idx, observed_object_idx)
            missing_logit = model.decode_completion(hidden["object"], hidden["worker"], missing_worker_idx, missing_object_idx)
            val_aux = aux_loss_fn(mv_logit[val_mask], y_majority[val_mask])
            if bool(observed_val_mask.any()):
                val_observed = observed_ce(obs_logit[observed_val_mask], observed_label[observed_val_mask])
            else:
                val_observed = _truth_logit.new_tensor(0.0)
            val_prior = safe_mean(soft_cross_entropy(missing_logit[missing_val_mask], missing_q_init[missing_val_mask]))
            val_loss_tensor = (
                args.aux_weight * val_aux
                + args.completion_observed_weight * val_observed
                + args.completion_prior_weight * val_prior
            )
            val_loss = float(val_loss_tensor.item())

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = _PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        truth_logit, _mv_logit, hidden = model(x_dict, edge_index_dict, edge_attr_dict, object_bucket_id=bucket_id)
        truth_prob = torch.softmax(truth_logit, dim=-1)
        missing_logit = model.decode_completion(hidden["object"], hidden["worker"], missing_worker_idx, missing_object_idx)
        missing_prob = torch.softmax(missing_logit, dim=-1)

    return (
        truth_prob.cpu().numpy().astype(np.float32),
        missing_prob.cpu().numpy().astype(np.float32),
        data.missing_object_idx.cpu().numpy().astype(np.int64),
    )
