"""Phase 1: stream the complete LANL auth file through GRU + Interpreter."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from deepcase import DeepCASE
from deepcase.preprocessing.lanl import read_full_auth

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "DeepCASE/example/data/lanl"


def confusion(y, pred):
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "false_alert_rate": fp / (fp + tn) if fp + tn else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--auth", type=Path, default=DATA / "auth.txt")
    p.add_argument("--redteam", type=Path, default=DATA / "redteam.txt")
    p.add_argument("--output", type=Path, default=ROOT / "runs/phase1_full_auth")
    p.add_argument("--split-time", type=int, default=147600)
    p.add_argument("--context", type=int, default=10)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train-limit", type=int, default=100000)
    p.add_argument("--reference-limit", type=int, default=20000)
    p.add_argument("--test-negatives", type=int, default=100000)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--progress-every", type=int, default=10000000)
    args = p.parse_args()
    out = args.output.resolve()
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Run already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    print(f"Streaming complete auth file {args.auth}; split={args.split_time}", flush=True)
    vocab, counts, train, reference, attacks, negatives = read_full_auth(
        args.auth, args.redteam, split_time=args.split_time,
        context_length=args.context, train_limit=args.train_limit,
        reference_limit=args.reference_limit, test_negative_limit=args.test_negatives,
        seed=args.seed, progress_every=args.progress_every)
    if counts["train_redteam_matches"]:
        raise ValueError("Training split contains Red Team events; move split earlier")
    if not attacks:
        raise ValueError("No held-out Red Team authentication events found")

    tx = torch.tensor([x for x, _ in train], dtype=torch.long)
    ty = torch.tensor([y for _, y in train], dtype=torch.long)
    rx = torch.tensor([x for x, _ in reference], dtype=torch.long)
    ry = torch.tensor([y for _, y in reference], dtype=torch.long)
    model = DeepCASE(features=len(vocab), max_length=args.context,
                     hidden_size=args.hidden, eps=0.1, min_samples=5,
                     threshold=0.2)
    print(f"Training GRU on {len(train):,} pre-attack windows", flush=True)
    model.context_builder.fit(tx, ty, epochs=args.epochs, batch_size=128,
                              learning_rate=0.003, optimizer=torch.optim.Adam,
                              verbose=False)
    print(f"Building Interpreter clusters from {len(reference):,} pre-attack references",
          flush=True)
    model.interpreter.fit(rx, ry,
                          scores=np.zeros(len(reference), dtype=np.int64),
                          iterations=args.iterations, batch_size=512,
                          strategy="max", verbose=False)

    test = attacks + negatives
    test_x = torch.tensor([x for (x, _y), _meta in test], dtype=torch.long)
    test_y = torch.tensor([y for (_x, y), _meta in test], dtype=torch.long)
    truth = np.asarray([1] * len(attacks) + [0] * len(negatives), dtype=np.int64)
    print(f"Scoring {len(attacks):,} Red Team events and {len(negatives):,} sampled nonmatches",
          flush=True)
    scores = model.predict(test_x, test_y, iterations=args.iterations,
                           batch_size=512, verbose=False)
    # Phase 1 review rule: a negative Interpreter code means no confident match
    # to a known pre-attack context. It requests human review; it is not an
    # automatic claim that the event is malicious.
    alerts = (scores < 0).astype(np.int64)
    metrics = confusion(truth, alerts)
    score_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(scores, return_counts=True))}
    group_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(model.interpreter.clusters,
                                                                    return_counts=True))}
    report = {
        "phase": 1,
        "objective": "Train original DeepCASE GRU Context Builder and Interpreter on full-timeline LANL authentication; group/flag contexts for analyst review.",
        "data": {"auth_path": str(args.auth.resolve()),
                 "auth_bytes": args.auth.stat().st_size,
                 "redteam_path": str(args.redteam.resolve()),
                 "complete_auth_file_streamed": True,
                 "filter": "remote LogOn events with distinct, known source/destination hosts",
                 "event_features": ["authentication_type", "logon_type", "orientation", "result", "user_mismatch"],
                 "identities_and_redteam_labels_are_model_features": False},
        "protocol": {"chronological_train_cutoff": args.split_time,
                     "train": "before cutoff; no Red Team matches allowed",
                     "interpreter_reference": "reservoir sample from pre-attack train, all treated as normal",
                     "test": "all held-out Red Team auth matches plus reservoir-sampled nonmatches",
                     "interpretation": "negative score is novelty/review request, not a maliciousness probability"},
        "config": {"context_length": args.context, "hidden_size": args.hidden,
                   "epochs": args.epochs, "seed": args.seed,
                   "training_reservoir": args.train_limit,
                   "reference_reservoir": args.reference_limit,
                   "test_negative_reservoir": args.test_negatives,
                   "attention_query_iterations": args.iterations,
                   "interpreter_eps": model.interpreter.eps,
                   "interpreter_min_samples": model.interpreter.min_samples,
                   "confidence_threshold": model.interpreter.threshold},
        "audit": dict(counts),
        "sample": {"test_attacks": len(attacks), "test_nonmatches": len(negatives),
                   "test_rows": len(test), "vocabulary_size": len(vocab),
                   "reference_cluster_counts": group_counts,
                   "test_interpreter_score_counts": score_counts},
        "review_metrics_on_attack_complete_sample": metrics,
        "training_next_event_loss": [float(x) for x in model.context_builder.loss_history],
        "limitations": [
            "Every physical auth row is scanned, but training and Interpreter references use bounded reservoirs because materializing hundreds of millions of windows is impractical.",
            "Evaluation retains every known Red Team auth event and samples nonmatches; precision and F1 are case-control sample metrics, not full-population estimates.",
            "A novelty review request or cluster match is not an analyst attack classification.",
            "Phase 1 uses authentication only and has no GNN."
        ]
    }
    model.save(str(out / "deepcase.pt"))
    (out / "vocabulary.json").write_text(json.dumps(vocab, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "predictions_sample.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_line", "timestamp", "source_computer", "destination_computer",
                    "redteam_match_for_evaluation", "interpreter_score", "review_request"])
        metadata = [x[1] for x in attacks] + [x[1] for x in negatives]
        for meta, label, score, alert in zip(metadata, truth, scores, alerts):
            w.writerow([*meta, int(label), int(score), int(alert)])
    print(json.dumps({"physical_rows": counts["physical_rows"],
                      "remote_logon_rows": counts["remote_logon_rows"],
                      "train_rows": counts["train_rows"], "test_rows": counts["test_rows"],
                      "redteam_test": len(attacks), "reference_clusters": group_counts,
                      "metrics": metrics, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
