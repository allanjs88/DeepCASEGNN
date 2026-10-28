"""Streaming LANL authentication preprocessing for the standalone Phase 1."""
import csv
import random
from collections import Counter, defaultdict, deque

import numpy as np

AUTH_COLUMNS = ("timestamp", "source_user", "destination_user", "source_computer",
                "destination_computer", "authentication_type", "logon_type",
                "orientation", "result")


def load_redteam(path):
    with open(path, "r", newline="") as stream:
        return {(int(r[0]), r[1], r[2], r[3])
                for r in csv.reader(stream) if r}


def event_category(row):
    """Categorical event attributes only; exclude host/account identities."""
    return "|".join((row[5], row[6], row[7], row[8],
                     str(int(row[1] != row[2]))))


def read_full_auth(auth_path, redteam_path, split_time=147600, context_length=10,
                   train_limit=100000, reference_limit=20000,
                   test_negative_limit=100000, seed=7, progress_every=10_000_000):
    """Scan every auth row, build causal contexts, and retain bounded samples.

    Red Team tuples are used for label auditing and held-out evaluation only.
    They are never added to features or GRU targets.
    """
    redteam = load_redteam(redteam_path)
    rng = random.Random(seed)
    histories = defaultdict(lambda: deque(maxlen=context_length))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    train, reference, test_attacks, test_negatives = [], [], [], []
    counts = Counter()
    seen = Counter()
    previous_time = -1

    with open(auth_path, "rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            fields = raw.rstrip(b"\r\n").split(b",")
            if len(fields) != 9:
                counts["malformed_rows"] += 1
                continue
            try:
                row = [x.decode("ascii") for x in fields]
                timestamp = int(row[0])
            except (UnicodeDecodeError, ValueError):
                counts["malformed_rows"] += 1
                continue
            if timestamp < previous_time:
                raise ValueError(f"LANL auth timestamps decrease at line {line_no}")
            previous_time = timestamp
            counts["physical_rows"] += 1
            src, dst = row[3], row[4]
            key = (timestamp, row[1], src, dst)
            is_redteam = int(key in redteam)
            counts["redteam_matches_all_auth"] += is_redteam
            if row[7] != "LogOn" or src == dst or src in ("", "?") or dst in ("", "?"):
                if line_no % progress_every == 0:
                    print(f"Scanned {line_no:,} auth rows; retained {counts['remote_logon_rows']:,} remote LogOns",
                          flush=True)
                continue

            counts["remote_logon_rows"] += 1
            counts["remote_logon_redteam_matches"] += is_redteam
            category = event_category(row)
            if timestamp < split_time and category not in vocabulary:
                vocabulary[category] = len(vocabulary)
            token = vocabulary.get(category, 1)
            history = histories[src]
            while history and timestamp - history[0][0] > 86400:
                history.popleft()
            context = [0] * (context_length - len(history)) + [item[1] for item in history]
            item = (context, [token])
            history.append((timestamp, token))

            if timestamp < split_time:
                counts["train_rows"] += 1
                counts["train_redteam_matches"] += is_redteam
                seen["train"] += 1
                if len(train) < train_limit:
                    train.append(item)
                else:
                    slot = rng.randrange(seen["train"])
                    if slot < train_limit:
                        train[slot] = item
                seen["reference"] += 1
                if len(reference) < reference_limit:
                    reference.append(item)
                else:
                    slot = rng.randrange(seen["reference"])
                    if slot < reference_limit:
                        reference[slot] = item
            else:
                counts["test_rows"] += 1
                counts["test_redteam_matches"] += is_redteam
                record = (item, (line_no, timestamp, src, dst))
                if is_redteam:
                    test_attacks.append(record)
                else:
                    seen["test_negatives"] += 1
                    if len(test_negatives) < test_negative_limit:
                        test_negatives.append(record)
                    else:
                        slot = rng.randrange(seen["test_negatives"])
                        if slot < test_negative_limit:
                            test_negatives[slot] = record

            if line_no % progress_every == 0:
                print(f"Scanned {line_no:,} rows; train={counts['train_rows']:,}; "
                      f"test={counts['test_rows']:,}; Red Team={counts['test_redteam_matches']:,}",
                      flush=True)

    counts["redteam_tuples_in_file"] = len(redteam)
    counts["redteam_tuples_not_matched_any_auth"] = max(0, len(redteam) - counts["redteam_matches_all_auth"])
    counts["training_windows_seen"] = seen["train"]
    counts["reference_windows_seen"] = seen["reference"]
    counts["test_nonmatches_seen"] = seen["test_negatives"]
    counts["training_windows_retained"] = len(train)
    counts["reference_windows_retained"] = len(reference)
    counts["test_attacks_retained"] = len(test_attacks)
    counts["test_nonmatches_retained"] = len(test_negatives)
    return vocabulary, counts, train, reference, test_attacks, test_negatives
