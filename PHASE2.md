# Phase 2: GRU + GraphSAGE event-context pilot

Phase 2 adds an independent `deepcase_gnn` implementation next to the copied
original `deepcase` package. The Phase 1 source and saved checkpoint are left
unchanged. The runner streams the complete LANL `auth.txt`, uses the same
timestamp cutoff as Phase 1, and writes its outputs under `runs/phase2_graphsage`.

## What the model does

For each target authentication event, the runner builds a causal host graph from
the previous 300 seconds. A computer is a node and an observed remote LogOn is
a directed source-to-destination edge. Each host's recent outgoing event
categories are embedded and encoded by a GRU. Two mean-aggregation GraphSAGE
layers pass information over the directed graph. An event decoder combines the
source and destination host embeddings with historical source-destination pair
statistics and predicts the current authentication category, conditioned on
the target event's observed source and destination hosts. It does not predict
whether a host pair should communicate or whether the event is an attack.

Training minimizes categorical cross-entropy for the observed event type. Red Team records and
labels do not enter the model, graph, or calibration. At evaluation time, the
model's surprise is the negative log probability of the observed category. The
interpreter is a validation-set 99th-percentile threshold: events above it are
sent for review.

This is a first graph-enhanced event-surprise baseline. It does **not yet**
preserve DeepCASE's attention decoder or its DBSCAN-style attention-vector
Interpreter, and it does not assign SOC severity groups. Those are explicit
follow-up work, not results claimed by this phase. In Phase 1, the copied
original DeepCASE model remains the baseline.

## Matched comparison

The complete auth file is streamed. The runner decodes all fields in the full
pre-cutoff training interval and in the union of the selected test events'
preceding 300-second windows; outside those ranges it reads timestamps only.
This preserves complete prior graph context for every selected test event while
avoiding graph updates in unrelated periods. The GNN fit uses a 50,000-event
uniform reservoir from the pre-cutoff period, with 10% held out for threshold
calibration. The test contains all 702 Phase 1 Red Team matches and 10,000
nonmatches sampled from Phase 1's saved nonmatch sample. Thus both models are
scored on the same test event IDs. The GNN's thresholding rule differs from
Phase 1's Interpreter rule, so this comparison is informative but not a matched
alert-budget comparison. Precision, F1, PR-AUC, and average precision are
sample-based, not estimates over all non-attack events in LANL.

Graph construction is bounded to four incoming and four outgoing neighbors per
target endpoint and at most 64 edges per snapshot. All rows in the selected
time ranges update the rolling graph, so each sampled target uses its complete
prior 300-second context rather than context reconstructed only from sampled
rows.

## Run

From this directory:

```bash
.venv/bin/python run_phase2.py --epochs 10 --output runs/phase2_graphsage
```

For a different environment, install the optional graph dependency with
`.venv/bin/python -m pip install -r requirements-gnn.txt` after configuring a
PyTorch build appropriate for the machine.

The run writes `deepcase_gnn.pt`, `phase2_report.json`, and
`predictions_matched.csv`. The report includes Phase 1 baseline metrics
recomputed on the identical test sample, GNN metrics, training loss, threshold,
sampling details, and limitations.

## Run results

The run completed on CPU in approximately 35.6 minutes. It streamed 1,051,430,459
auth rows, built 50,000 pre-cutoff graph snapshots (45,000 fit and 5,000
calibration), and evaluated 702 known Red Team matches plus 10,000 matched
nonmatches. The GNN's validation 99th-percentile surprise threshold was 2.5563.

| Model and review rule | TP | FN | FP | TN | Precision | Recall | F1 | False-alert rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 1 DeepCASE, negative Interpreter code | 475 | 227 | 339 | 9,661 | 58.35% | 67.66% | 0.6266 | 3.39% |
| Phase 2 GRU + GraphSAGE, surprise > 2.5563 | 30 | 672 | 82 | 9,918 | 26.79% | 4.27% | 0.0737 | 0.82% |

The GNN requested fewer reviews and produced fewer false alerts at this
threshold, but missed 672 of the 702 known attacks. Its F1 is much lower in
this run, so this is not an improvement over Phase 1. The operating points are
not identical: Phase 1 uses its learned-context Interpreter rule; Phase 2 uses
a 99th-percentile validation threshold.

For example, the Red Team event at timestamp `151648` (source `C17693`,
destination `C728`) had category `NTLM|Network|LogOn|Success|0`. The GNN surprise
was `2.9818`, above threshold, so it requested review; Phase 1 also requested
review. Another Red Team event at timestamp `150885` (source `C17693`,
destination `C1003`) had the same event category but GNN surprise `0.4212`, so
it was not reviewed by Phase 2; Phase 1 did request review. The same event type
can be routine or unusual depending on its recent host and relationship
context.

The matched evaluation has 702 attacks among 10,702 rows (6.56%), because
nonmatches were sampled. Consequently its Phase 1 precision and F1 are not
directly comparable to the earlier Phase 1 report that used 100,000 sampled
nonmatches (12.10% precision and F1 0.2052). The matched-subset results are
appropriate for comparing both models on identical events. PR-AUC and average
precision are reported only for the GNN's continuous surprise score; Phase 1
Interpreter outputs are categorical codes, not a ranking score.

This is still event-level review triage, not grouping into SOC `INFO`, `LOW`,
`MEDIUM`, `HIGH`, or `ATTACK` categories. Red Team matches are evaluation labels,
not analyst-assigned severity labels.

Use a new output folder for a repeat run. To change the sample sizes or number
of epochs, for example:

```bash
.venv/bin/python run_phase2.py \
  --train-samples 50000 --test-negatives 10000 --epochs 10 \
  --output runs/phase2_graphsage_repeat
```
