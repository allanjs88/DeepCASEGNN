"""Phase 2: full-auth stream into a bounded GRU + GraphSAGE experiment."""

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from deepcase_gnn.model import GraphContextBuilder, binary_metrics


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "DeepCASE/example/data/lanl"
PHASE1 = ROOT / "runs/phase1_full_auth"
WINDOW_SECONDS = 300
AUTH_FIELDS = 9


def event_category(row):
    return "|".join((row[5], row[6], row[7], row[8],
                     str(int(row[1] != row[2]))))


def load_phase1_test(path, negative_limit, seed):
    """Choose a reproducible negative subset and retain every Phase 1 attack."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    attacks = [row for row in rows
               if int(row["redteam_match_for_evaluation"]) == 1]
    negatives = [row for row in rows
                 if int(row["redteam_match_for_evaluation"]) == 0]
    if negative_limit > len(negatives):
        raise ValueError(f"Requested {negative_limit:,} negatives, only "
                         f"{len(negatives):,} are available in Phase 1 sample.")
    chosen = random.Random(seed).sample(negatives, negative_limit)
    selected = attacks + chosen
    by_line = {int(row["input_line"]): row for row in selected}
    return by_line, attacks, chosen


class RollingGraph:
    """Bounded 300-second causal host graph used to snapshot selected events."""

    def __init__(self, max_length=10, max_neighbors=4, max_edges=64):
        self.max_length = max_length
        self.max_neighbors = max_neighbors
        self.max_edges = max_edges
        self.active = deque()
        self.history = defaultdict(lambda: deque(maxlen=max_length))
        self.pairs = {}
        self.out_count = Counter()
        self.in_count = Counter()
        self.out_neighbors = defaultdict(set)
        self.in_neighbors = defaultdict(set)

    def clear(self):
        """Start a disjoint time range with no events before its left edge."""
        self.active.clear()
        self.history.clear()
        self.pairs.clear()
        self.out_count.clear()
        self.in_count.clear()
        self.out_neighbors.clear()
        self.in_neighbors.clear()

    def expire(self, timestamp):
        while self.active and timestamp - self.active[0][0] > WINDOW_SECONDS:
            t, source, destination, user, result = self.active.popleft()
            pair = (source, destination)
            state = self.pairs[pair]
            state["times"].popleft()
            state["users"][user] -= 1
            if state["users"][user] <= 0:
                del state["users"][user]
            if result == "Success":
                state["success"] -= 1
            elif result == "Fail":
                state["fail"] -= 1
            self.out_count[source] -= 1
            self.in_count[destination] -= 1
            if not state["times"]:
                del self.pairs[pair]
                self.out_neighbors[source].discard(destination)
                self.in_neighbors[destination].discard(source)

    def add(self, timestamp, source, destination, user, result, token):
        pair = (source, destination)
        state = self.pairs.get(pair)
        if state is None:
            state = {"times": deque(), "users": Counter(),
                     "success": 0, "fail": 0}
            self.pairs[pair] = state
            self.out_neighbors[source].add(destination)
            self.in_neighbors[destination].add(source)
        state["times"].append(timestamp)
        state["users"][user] += 1
        if result == "Success":
            state["success"] += 1
        elif result == "Fail":
            state["fail"] += 1
        self.out_count[source] += 1
        self.in_count[destination] += 1
        self.active.append((timestamp, source, destination, user, result))
        self.history[source].append((timestamp, token))

    @staticmethod
    def _pair_attr(state, timestamp):
        if state is None:
            return [0.0, 0.0, 0.0, 0.0]
        count = len(state["times"])
        failures = state["fail"]
        return [math.log1p(count), math.log1p(len(state["users"])),
                failures / count if count else 0.0,
                max(0.0, timestamp - state["times"][-1]) / WINDOW_SECONDS]

    def _ranked_neighbors(self, host, incoming, timestamp):
        neighbors = self.in_neighbors[host] if incoming else self.out_neighbors[host]
        result = []
        for neighbor in neighbors:
            key = (neighbor, host) if incoming else (host, neighbor)
            state = self.pairs[key]
            result.append((len(state["times"]), state["times"][-1], neighbor))
        result.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [host_id for _, _, host_id in result[:self.max_neighbors]]

    def snapshot(self, timestamp, source, destination, token, sample_position):
        hosts = {source, destination}
        for endpoint in (source, destination):
            hosts.update(self._ranked_neighbors(endpoint, False, timestamp))
            hosts.update(self._ranked_neighbors(endpoint, True, timestamp))
        hosts = sorted(hosts)
        host_id = {host: index for index, host in enumerate(hosts)}
        sequence = torch.zeros((len(hosts), self.max_length), dtype=torch.long)
        lengths = torch.zeros(len(hosts), dtype=torch.long)
        features = torch.zeros((len(hosts), 4), dtype=torch.float32)
        for host, index in host_id.items():
            events = self.history[host]
            while events and timestamp - events[0][0] > WINDOW_SECONDS:
                events.popleft()
            tokens = [event_token for _, event_token in events]
            if tokens:
                sequence[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                lengths[index] = len(tokens)
            features[index] = torch.tensor(
                [math.log1p(self.out_count[host]), math.log1p(self.in_count[host]),
                 math.log1p(len(self.out_neighbors[host])),
                 math.log1p(len(self.in_neighbors[host]))], dtype=torch.float32)

        edges = []
        for left in hosts:
            for right in hosts:
                state = self.pairs.get((left, right))
                if state:
                    edges.append((len(state["times"]), left, right,
                                  self._pair_attr(state, timestamp)))
        edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
        edges = edges[:self.max_edges]
        edge_index = torch.tensor(
            [[host_id[s], host_id[d]] for _, s, d, _ in edges], dtype=torch.long
        ).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor([attrs for _, _, _, attrs in edges],
                                 dtype=torch.float32).reshape(-1, 4)
        source_index, destination_index = host_id[source], host_id[destination]
        return Data(
            num_nodes=len(hosts), sequence=sequence, lengths=lengths, x=features,
            edge_index=edge_index, edge_attr=edge_attr,
            edge_weight=torch.tensor([count for count, *_ in edges], dtype=torch.float32),
            target_index=torch.tensor([[source_index], [destination_index]], dtype=torch.long),
            target_attr=torch.tensor(
                [self._pair_attr(self.pairs.get((source, destination)), timestamp)],
                dtype=torch.float32),
            y=torch.tensor([token], dtype=torch.long),
            sample_position=torch.tensor([sample_position], dtype=torch.long))


def add_sample_position(data, position):
    data.sample_position = torch.tensor([position], dtype=torch.long)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth", type=Path, default=DATA / "auth.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/phase2_graphsage")
    parser.add_argument("--phase1-predictions", type=Path,
                        default=PHASE1 / "predictions_sample.csv")
    parser.add_argument("--split-time", type=int, default=147600)
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=50000,
                        help="Uniform reservoir sample of pre-cutoff events; 10%% held out for threshold calibration.")
    parser.add_argument("--test-negatives", type=int, default=10000,
                        help="Matched subset selected from Phase 1's sampled test nonmatches.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--progress-every", type=int, default=10000000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    output = args.output.resolve()
    if (output / "phase2_report.json").exists():
        raise FileExistsError(f"Run already exists: {output}")
    if not args.auth.is_file() or not args.phase1_predictions.is_file():
        raise FileNotFoundError("Phase 2 needs auth.txt and the saved Phase 1 sample predictions.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")

    test_rows, phase1_attacks, phase1_negatives = load_phase1_test(
        args.phase1_predictions, args.test_negatives, args.seed)
    scan_ranges = [(0, args.split_time - 1)] if args.split_time > 0 else []
    scan_ranges.extend((max(0, int(row["timestamp"]) - WINDOW_SECONDS),
                        int(row["timestamp"]))
                       for row in test_rows.values())
    scan_ranges.sort()
    merged_ranges = []
    for left, right in scan_ranges:
        if merged_ranges and left <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0],
                                 max(merged_ranges[-1][1], right))
        else:
            merged_ranges.append((left, right))
    scan_ranges = merged_ranges
    selected_test = set(test_rows)
    test_labels = {line: int(row["redteam_match_for_evaluation"])
                   for line, row in test_rows.items()}
    baseline_alert = {line: int(row["review_request"]) for line, row in test_rows.items()}
    rng = random.Random(args.seed)
    train_samples = []
    train_seen = 0
    test_samples = {}
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    rolling = RollingGraph(max_length=args.context)
    physical_rows = malformed_rows = remote_logons = train_rows = 0
    test_context_logons = 0
    physical_lines_streamed = 0
    train_redteam = 0
    previous_timestamp = -1
    group_timestamp = None
    active_range = None
    range_index = 0
    pending = []
    start = time.time()

    def flush_pending():
        for timestamp, source, destination, user, result, token in pending:
            rolling.add(timestamp, source, destination, user, result, token)
        pending.clear()

    print(f"Phase 2: scanning every auth row from {args.auth}; "
          f"cutoff={args.split_time:,}; selected Phase 1 test negatives="
          f"{len(phase1_negatives):,}; context ranges={len(scan_ranges):,}; "
          f"device={device}", flush=True)
    with args.auth.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            physical_lines_streamed = line_no
            first_field = raw.partition(b",")[0].strip()
            try:
                timestamp = int(first_field)
            except ValueError:
                malformed_rows += 1
                continue
            if timestamp < previous_timestamp:
                raise ValueError(f"Auth timestamps decrease at line {line_no:,}")
            previous_timestamp = timestamp
            physical_rows += 1
            if line_no % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"Scanned {line_no:,} rows; retained auth LogOns={remote_logons:,}; "
                      f"train={train_rows:,}; test snapshots={len(test_samples):,}; "
                      f"elapsed={elapsed / 60:.1f} min", flush=True)

            while (range_index < len(scan_ranges)
                   and timestamp > scan_ranges[range_index][1]):
                range_index += 1
            if (range_index >= len(scan_ranges)
                    or timestamp < scan_ranges[range_index][0]):
                if active_range is not None:
                    flush_pending()
                    active_range = None
                    group_timestamp = None
                continue
            if active_range != range_index:
                if active_range is not None:
                    flush_pending()
                rolling.clear()
                group_timestamp = None
                active_range = range_index

            fields = raw.rstrip(b"\r\n").split(b",")
            if len(fields) != AUTH_FIELDS:
                malformed_rows += 1
                continue
            try:
                row = [value.decode("ascii") for value in fields]
            except UnicodeDecodeError:
                malformed_rows += 1
                continue
            if group_timestamp is None or timestamp != group_timestamp:
                if group_timestamp is not None:
                    flush_pending()
                group_timestamp = timestamp
                rolling.expire(timestamp)

            source, destination = row[3], row[4]
            if (row[7] != "LogOn" or source == destination
                    or source in ("", "?") or destination in ("", "?")):
                continue
            remote_logons += 1
            if timestamp >= args.split_time:
                test_context_logons += 1
            category = event_category(row)
            if timestamp < args.split_time and category not in vocabulary:
                vocabulary[category] = len(vocabulary)
            token = vocabulary.get(category, 1)
            label = test_labels.get(line_no, 0)

            if timestamp < args.split_time:
                train_rows += 1
                if label:
                    train_redteam += 1
                train_seen += 1
                if len(train_samples) < args.train_samples:
                    pos = len(train_samples)
                    graph = rolling.snapshot(timestamp, source, destination, token, pos)
                    train_samples.append((graph, timestamp, token))
                else:
                    slot = rng.randrange(train_seen)
                    if slot < args.train_samples:
                        graph = rolling.snapshot(timestamp, source, destination, token, slot)
                        train_samples[slot] = (graph, timestamp, token)
            elif line_no in selected_test:
                pos = len(test_samples)
                graph = rolling.snapshot(timestamp, source, destination, token, pos)
                phase1_row = test_rows[line_no]
                test_samples[line_no] = (graph, timestamp, source, destination,
                                         row[1], token, label,
                                         int(phase1_row["interpreter_score"]),
                                         baseline_alert[line_no])

            pending.append((timestamp, source, destination, row[1], row[8], token))
    flush_pending()
    if train_redteam:
        raise ValueError("Red Team matches appeared in the training interval.")
    if len(train_samples) != args.train_samples:
        raise RuntimeError(f"Only retained {len(train_samples):,} train samples.")
    missing_test = selected_test - set(test_samples)
    if missing_test:
        raise RuntimeError(f"Could not build {len(missing_test):,} selected test graphs; "
                           "check line IDs and the Phase 1 preprocessing filter.")

    split_rng = random.Random(args.seed + 101)
    split_rng.shuffle(train_samples)
    validation_count = max(1, int(round(len(train_samples) * 0.10)))
    validation_samples = train_samples[:validation_count]
    fitting_samples = train_samples[validation_count:]
    train_data = [add_sample_position(row[0], i) for i, row in enumerate(fitting_samples)]
    validation_data = [add_sample_position(row[0], i) for i, row in enumerate(validation_samples)]
    ordered_test_lines = sorted(test_samples)
    test_data = [add_sample_position(test_samples[line][0], i)
                 for i, line in enumerate(ordered_test_lines)]
    categories = len(vocabulary)
    model = GraphContextBuilder(features=categories, hidden_size=32,
                                embedding_size=32, gnn_layers=2,
                                seed=args.seed, device=device)
    print(f"Training on {len(train_data):,} graph snapshots; "
          f"validation={len(validation_data):,}; test={len(test_data):,}", flush=True)
    model.fit(train_data, epochs=args.epochs, batch_size=args.batch_size,
              learning_rate=0.003, verbose=True)
    validation_scores = model.score(validation_data, batch_size=args.batch_size)
    threshold = model.calibrate(validation_scores, args.threshold_quantile)
    test_scores = model.score(test_data, batch_size=args.batch_size)
    gnn_alerts = model.predict(test_scores)
    labels = np.asarray([test_labels[line] for line in ordered_test_lines], dtype=np.int8)
    baseline_alerts = np.asarray([baseline_alert[line] for line in ordered_test_lines], dtype=np.int8)
    gnn_metrics = binary_metrics(labels, gnn_alerts, test_scores)
    baseline_metrics = binary_metrics(labels, baseline_alerts)

    model.save(output / "deepcase_gnn.pt", vocabulary)
    predictions_path = output / "predictions_matched.csv"
    with predictions_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_line", "timestamp", "source_user", "source_computer",
                         "destination_computer", "event_category_id", "redteam_match_for_evaluation",
                         "gnn_surprise", "gnn_review_request", "phase1_interpreter_score",
                         "phase1_review_request"])
        for line, score, alert in zip(ordered_test_lines, test_scores, gnn_alerts):
            _, timestamp, source, destination, user, token, label, phase1_score, phase1_alert = test_samples[line]
            writer.writerow([line, timestamp, user, source, destination, token, label,
                             f"{score:.8f}", int(alert), phase1_score, phase1_alert])

    elapsed = time.time() - start
    report = {
        "phase": 2,
        "objective": "Add a separate GRU + directed GraphSAGE event-context model while leaving Phase 1 DeepCASE unchanged.",
        "architecture": {
            "node": "computer host",
            "edge": "directed remote LogOn observed in the previous 300 seconds",
            "node_history": f"last {args.context} outgoing event categories inside the previous 300 seconds, encoded by GRU",
            "graph_model": "two mean-aggregation GraphSAGE layers over the directed host graph",
            "event_decoder": "MLP over source embedding, destination embedding, and historical pair features; predicts the current category conditioned on the observed target endpoints",
            "interpreter": "validation 99th-percentile surprise threshold; not DeepCASE attention-vector DBSCAN",
            "redteam_use": "held-out evaluation only; never used for graph construction, training labels, or threshold selection"
        },
        "data": {"auth_path": str(args.auth.resolve()), "complete_auth_file_streamed": True,
                 "physical_lines_streamed": physical_lines_streamed,
                 "physical_rows_with_valid_timestamps": physical_rows,
                 "malformed_rows": malformed_rows,
                 "context_ranges_used": len(scan_ranges),
                 "remote_logons_decoded_for_graph_context": remote_logons,
                 "test_context_remote_logons": test_context_logons,
                 "train_rows_before_cutoff": train_rows,
                 "train_redteam_matches": train_redteam, "split_time": args.split_time,
                 "event_category_count_including_reserved": categories,
                 "feature_categories": ["authentication_type", "logon_type", "orientation",
                                        "result", "user_mismatch"]},
        "sampling": {"train_reservoir_snapshots": len(train_samples),
                     "fit_snapshots": len(train_data), "validation_snapshots": len(validation_data),
                     "all_phase1_redteam_test_events": len(phase1_attacks),
                     "matched_test_nonmatches": len(phase1_negatives),
                     "test_total": len(test_data), "seed": args.seed,
                     "graph_neighbor_cap_per_endpoint_direction": 4,
                     "graph_edge_cap_per_snapshot": 64,
                     "test_negative_sampling": "uniform random subset of Phase 1's saved 100k nonmatch sample; identical event IDs scored by both models"},
        "config": {"context_length": args.context, "window_seconds": WINDOW_SECONDS,
                   "hidden_size": 32, "embedding_size": 32, "graphsage_layers": 2,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": 0.003, "threshold_quantile": args.threshold_quantile,
                   "threshold_surprise": threshold, "device": device},
        "training_event_category_nll": model.loss_history,
        "validation_surprise": {"count": len(validation_scores),
                                "mean": float(np.mean(validation_scores)),
                                "threshold": threshold},
        "matched_test_metrics": {"phase1_deepcase": baseline_metrics,
                                 "phase2_gru_graphsage": gnn_metrics},
        "runtime_minutes": elapsed / 60.0,
        "limitations": [
            "This is an event-level surprise/review experiment, not SOC severity classification or analyst-labeled grouping.",
            "Phase 2 uses a separate GRU + GraphSAGE next-event scorer and a quantile threshold. It does not yet use the original DeepCASE attention decoder or cluster-based Interpreter.",
            "Graph neighborhoods are capped at four incoming and four outgoing neighbors per target endpoint, and 64 edges, for bounded computation.",
            "The complete auth file is streamed, but training is based on a uniform 50k pre-cutoff reservoir; the matched test uses all known attacks and a 10k subset of Phase 1 sampled nonmatches.",
            "Precision, F1, PR-AUC, and average precision use the sampled negatives. The threshold is validation-calibrated, so its false-alert rate can differ from Phase 1's interpreter rule.",
            "A high surprise score is not a probability that an event is malicious; Red Team labels are incomplete ground truth."
        ]
    }
    (output / "phase2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "scan_minutes": round(elapsed / 60, 2),
                      "vocabulary": categories, "validation_threshold": threshold,
                      "phase1_metrics_on_matched_subset": baseline_metrics,
                      "phase2_metrics": gnn_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
