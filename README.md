# Phase 1 — DeepCASE on complete LANL authentication data

This is a standalone experiment under `Tests/deepcase-test`. It contains a minimal local copy of the original DeepCASE implementation required for the GRU Context Builder, attention/event decoders, and Interpreter. It does not copy the original repository's examples, notebooks, GNN package, or dataset.

## Phase 1 workflow

1. Stream the complete `auth.txt` file in timestamp order.
2. Build each remote `LogOn` event's context from the previous ten event categories on its source host, within one day.
3. Train the original DeepCASE Context Builder to predict the observed event category from its context. The GRU is not trained on Red Team labels.
4. Fit the Interpreter on pre-attack contexts treated as normal. It uses attention querying and DBSCAN to group similar event contexts.
5. Score every held-out Red Team authentication event plus a reproducible sample of nonmatches. A negative Interpreter code is a request for analyst review because the context did not confidently match a known normal cluster; it is not a maliciousness probability.

Event categories use authentication method, logon type, orientation, result, and whether the source and destination users differ. Raw user and host identities and Red Team labels are not model features. The exact Red Team tuple is used for label audit and held-out evaluation only.

## Complete data and bounded memory

The preprocessor scans every physical row of the complete LANL `auth.txt`. It keeps bounded reservoir samples for training windows, normal Interpreter references, and test nonmatches. The stream still updates host histories and audits Red Team matches across the whole file. This lets the experiment use the complete chronological timeline without loading the 69 GB file or hundreds of millions of windows into memory. Every held-out Red Team match is retained for evaluation.

## Run

The existing local virtual environment is used:

```bash
cd /home/allanmurillo/Documents/src/deepcase/DeepCASE-GNN/Tests/deepcase-test
.venv/bin/python run_phase1.py --output runs/phase1_full_auth
```

Optional settings include `--epochs`, `--train-limit`, `--reference-limit`, `--test-negatives`, `--context`, and `--split-time`. Default cutoff 147,600 is the LANL first-41-hours pre-attack period.

Outputs are `metrics.json`, `predictions_sample.csv`, `vocabulary.json`, and `deepcase.pt` under the selected run directory. Precision, recall, F1, and false-alert rate are reported on the attack-complete, sampled-negative evaluation set. DeepCASE Interpreter codes are categorical review outcomes, so the report does not treat them as a continuous ranking score. Sample precision/F1 are not full-population estimates because nonmatches are sampled.

## Minimal local DeepCASE copy

`deepcase/` contains only source modules needed by this experiment: the original `DeepCASE` wrapper, GRU encoder, attention and event decoders, loss/embedding utilities, and the clustering Interpreter. The LANL stream reader is implemented in `deepcase/preprocessing/lanl.py`. No GNN code is included in Phase 1.
