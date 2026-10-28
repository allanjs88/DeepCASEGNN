"""Training and scoring for the standalone GRU + GraphSAGE branch."""

import random

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             auc, precision_score, recall_score, f1_score)
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from .context_builder import ContextBuilder


class GraphContextBuilder:
    """Self-supervised event scorer using host histories and graph neighbors.

    This is deliberately separate from ``deepcase.DeepCASE``. It predicts an
    event category from historical host/edge context and thresholds its
    surprise score. Red Team labels are never used in model fitting.
    """

    def __init__(self, features, hidden_size=32, embedding_size=32,
                 gnn_layers=2, seed=7, device="cpu"):
        self.config = dict(features=int(features), hidden_size=hidden_size,
                           embedding_size=embedding_size, gnn_layers=gnn_layers,
                           seed=seed)
        self.device = torch.device(device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.network = ContextBuilder(features, hidden_size, embedding_size,
                                      gnn_layers).to(self.device)
        self.threshold = None
        self.loss_history = []

    def fit(self, dataset, epochs=10, batch_size=128, learning_rate=0.003,
            verbose=True):
        if not dataset or epochs < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("Training data and positive hyperparameters are required.")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(self.config["seed"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator)
        for epoch in range(epochs):
            self.network.train()
            total_loss = total_events = 0
            for graph in loader:
                graph = graph.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(graph)
                loss = F.cross_entropy(logits, graph.y - 1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * graph.y.numel()
                total_events += graph.y.numel()
            mean_loss = total_loss / total_events
            self.loss_history.append(mean_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: event-category NLL={mean_loss:.6f}",
                      flush=True)
        return self

    def score(self, dataset, batch_size=128):
        self.network.eval()
        scores = np.empty(len(dataset), dtype=np.float64)
        with torch.no_grad():
            for graph in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                graph = graph.to(self.device)
                losses = F.cross_entropy(self.network(graph), graph.y - 1,
                                         reduction="none")
                scores[graph.sample_position.cpu().numpy()] = losses.cpu().numpy()
        return scores

    def calibrate(self, validation_scores, quantile=0.99):
        values = np.asarray(validation_scores, dtype=float)
        if values.size == 0 or not np.isfinite(values).all() or not 0 < quantile < 1:
            raise ValueError("Calibration needs finite scores and a quantile in (0, 1).")
        self.threshold = float(np.quantile(values, quantile, method="higher"))
        return self.threshold

    def predict(self, scores):
        if self.threshold is None:
            raise RuntimeError("Calibrate a threshold on validation data first.")
        return (np.asarray(scores) > self.threshold).astype(np.int8)

    def save(self, path, vocabulary):
        torch.save({"format_version": 1, "config": self.config,
                    "state_dict": self.network.state_dict(),
                    "threshold": self.threshold,
                    "loss_history": self.loss_history,
                    "vocabulary": vocabulary}, path)


def binary_metrics(labels, predictions, scores=None):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    if scores is not None and labels.sum():
        scores = np.asarray(scores, dtype=float)
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(recall_curve, precision_curve))
        average_precision = float(average_precision_score(labels, scores))
    else:
        pr_auc = average_precision = None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "false_alert_rate": fp / (fp + tn) if fp + tn else None,
            "pr_auc": pr_auc, "average_precision": average_precision}
