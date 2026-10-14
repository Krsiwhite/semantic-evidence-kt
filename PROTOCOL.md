# Semantic Evidence KT: fixed first-pass protocol

This project tests two research hypotheses on top of a causal repair of the
public BAIM sparseKT implementation. Neither hypothesis is an established novel
method or an established improvement before controlled experiments finish.

## Provenance and baseline

The vendored network is from [BAIM](https://github.com/sjin4861/BAIM), commit
`a9501b5e44dafc0e81d2f17e717766a4106d0122`, file
`src/pykt-toolkit/pykt/models/sparsekt_baim.py`. Its Apache-2.0 license is retained
in `vendor/LICENSE.BAIM`; edits are listed in `UPSTREAM_AUDIT.json`.

The public sparse-attention implementation reintroduces forbidden attention
weights in its first five query rows. A CPU intervention verified that a future
answer changes earlier predictions in that implementation. This project zeros
forbidden attention weights again and renormalizes before attention dropout.
This finding concerns the inspected public commit. It does not establish which
code produced the paper's experiments. We do not train the known noncausal code.

The [paper](https://aclanthology.org/2026.findings-acl.1125.pdf) reports 83.21 ±
0.10 percent AUC for the Qwen3-VL-32B cache in its main table and 83.22 ± 0.08 for
the Qwen3-VL-8B variant in another table. These are author-reported five-fold
results, with an unresolved implementation discrepancy. A fold-zero result,
or a result using the causal repair, is not by itself a claim to have surpassed
those scores under the exact published procedure.

The baseline keeps the pretrained question vectors trainable, as in the
official implementation. All four variants share the same repaired backbone,
question-level data representation and random-seed initialization of shared
parameters. Core defaults follow the official CLI: hidden dimension 256, two
blocks, four heads, FFN 256, output MLP 256/256, dropout 0.1, sparse ratio 0.8,
k-index 5. The CLI explicitly overrides the bare network's output-MLP default
512 with 256. The load-balancing auxiliary term is 0.01 times squared deviation
of average gate probability from uniform, including padded positions, matching
the upstream implementation. Padding is on the right.
Padding uses item zero when it has training responses; otherwise it uses the
most frequent warm item. This prevents a held-out item zero from receiving
synthetic ID updates through the padding gate loss. The chosen ID is recorded.

## Two hypotheses and four variants

1. **Support-weighted semantic residuals with grouped pseudo-cold training.**
   Keep the four-stage semantic prior frozen, learn one shared adapter, and add
   an item residual multiplied by `n / (n + tau)`, where `n` is the training-only
   item count. In a training minibatch, all occurrences of an item share one
   Bernoulli mask that hides its ID residual. Defaults: adapter width 256,
   residual-mask probability 0.15, initial learnable tau 100; adapter output and
   item residual start at zero. A zero-response item therefore uses the prior
   and shared adapter. Frequency shrinkage alone is not a strong novelty claim;
   the hypothesis is the combined representation and pseudo-cold training
   mechanism. Distinguish it experimentally from semantic KT and empirical
   Bayes calibration work before claiming novelty.
2. **Causal historical-error evidence correction.** Retrieve at most 16 prior
   interactions using projected semantic similarity, shared-KC bonus and lag
   decay. Add a support-gated weighted sum of `response - detached_base_prob`
   to the base logit. Defaults: retrieval dimension 64, temperature 0.2,
   initial KC bonus 1, initial lag decay 0.1, support factor
   `effective_neighbors / (effective_neighbors + 4)`, initial amplitude 0.05
   bounded between zero and two. Each reported signed contribution is exactly
   additive in this correction. This does not explain the full BAIM base logit,
   imply causal educational effects, or expose a measured cognitive state.

The four fixed variants are `baseline`, `residual`, `evidence`, and `full`.
Baseline/evidence learn their raw per-item pretrained vectors; residual/full
replace that table with the frozen-prior/shared-adapter/residual module.
Evidence retrieval always uses a frozen semantic prior and training-time
learned shared projection. This representation change is part of mechanism 1,
so its ablation includes both freezing and residual adaptation.

## Data and primary evaluation

Data are from [XES3G5M](https://github.com/ai4ed/XES3G5M), provided under MIT.
Downloads need no paid API. The archive is approximately 356 MiB and the BAIM
cache approximately 90 MiB. `python data.py download --root DATA_ROOT` performs
the explicit download; importing modules or preparing a run never downloads.
Both files receive local SHA-256 records. Large caches and student sequences
are external artifacts and should be ignored by Git.

Use the official **question-level** CSVs, preserving all KCs of one question in
one timestep; do not expand a multi-KC answer into consecutive interactions.
For fold `i`, that fold is validation and the other four are training. The
official main test is `test_sequences_quelevel.csv`, not the separate windowed
test CSV. Preserve length-200 segments. The primary metric pools all selected
question targets and computes ROC-AUC. As in the official loader, standard
targets require `selectmasks != -1`, a valid response, and position greater than
zero. Item frequencies come only from training inputs. `prepare` never opens
the test CSV; its returned `test_factory` opens and hashes it only when called.

The downloaded archive contains the original test and the windowed test, but
omits the main segmented test. `prepare_test_sequences.py` invokes the unchanged
`generate_sequences` function from the pinned BAIM preprocessing source, with
`min_seq_len=3`, `maxlen=200`, and `pad_val=-1`. It preserves the original fixed
student split, event order, IDs, responses and multi-KC strings. Its source hashes,
input/output hashes and aggregate counts are in `reports/test_sequences_manifest.json`.
The resulting main test has 7,711 segments and 1,094,990 shifted scored targets.
This asset was materialized after fold0 selection froze and before any test
inference, recovering a missing data-preparation step without changing frozen
training/model source or rerunning completed configurations.

The official maximum is 200 epochs with patience five and required validation
AUC improvement greater than 0.0002, Adam learning rate 0.0001 and effective
batch size 128. The experiment runner records its actual compute cap; stopping
at that cap must be reported as budget-limited rather than full convergence.
Select by validation AUC, lock the four first-pass configurations, and evaluate
the fixed test only after that lock. Do not repeat completed runs or tune on
test results. Seed/fold expansions and cold evaluation are conditional on the
remaining shared 24-GPU-hour budget, with stage three allotted about 18 hours.
The default three hours for cold evaluation are a minimum reservation, not a
hard three-hour cap. Time saved when standard runs finish early carries forward
to the cold runs, whose shared deadline is the total stage-three allowance minus
the final inference reserve. The overall allowance is never extended.

## Secondary cold-item evaluation

This is a separate protocol and cannot be compared numerically with the
standard BAIM table. Never initialize a cold run from a standard checkpoint
that saw the held-out items' responses.

Partition every item ID in the full catalog using `numpy.default_rng(42)` to
shuffle `arange(N)`: the first floor(70% of N) are warm, the next floor(15% of N)
are cold validation, and the remainder are cold test. Persist sorted,
mutually exclusive item lists, actual counts and their SHA-256. This approved
70/15/15 assignment is independent of all responses, folds and test contents.
Only assigned warm items with actual training-fold responses may be included
as historical context; the exact supported-history item list is also recorded.

Physically remove all cold-item interactions from training sequences and
compact the remaining warm history within each original segment. Evaluate
each cold target independently, with at most 199 strictly past warm-item
responses followed by the current query. Another cold query is never included
in its history, even with a fabricated response. The only scored position is
the query; a query with no warm history is allowed. Current-query response is
stored as the loss label, and strict causal model checks establish that it
cannot influence its own prediction. Validation targets use cold-validation
items and validation students; test targets use cold-test items and official
test students. This can require substantially more inference than standard
evaluation because every cold query has its own window.

The cached vectors use PCA fit across the known item catalog. Therefore this
protocol means **zero training responses for held-out items with a preknown
item catalog**, not strict inductive generalization to previously unseen text.
A strict inductive study would instead use the official original RoBERTa
features without cross-item PCA, or fit any transform only on warm items; that
is not silently substituted into the present experiment.

## API and validation

Only `torch` and `numpy` are needed for model/data/CPU checks and the runner's
locally implemented metrics. Reuse the existing ROCm Python environment;
do not replace the GPU package with a generic dependency synchronization.

`data.prepare(data_root, fold=0, protocol='standard', cold_seed=42)` returns
`train`, `val`, `test_factory`, `embeddings`, `item_counts`, and JSON-serializable
`metadata`. Samples have fixed length 200 and contain `qids`, `responses`,
`valid`, `scoremask`, `concepts` (seven slots with -1 padding), and `timestamps`.
No custom collator is required. IDs and labels are integer tensors; masks are
boolean. Metadata contains stable source hashes, counts, fold and cold-split
hash. Test metadata is populated only when the lazy factory is invoked.

`models.SemanticEvidenceKT(...).forward(batch)` returns `logits`,
`probabilities` (also `probs`), `base_probs` and `auxiliary_loss`.
`return_contributions=True` adds evidence values, historical indices and masks.
Use masked BCEWithLogits plus the already-weighted auxiliary loss.

The single executed CPU invariant check is stored in `CPU_CHECKS.json`.
It verifies all four variants against every current/future response flip,
finite backward gradients, exact evidence decomposition within float32 error,
zero-support residual behavior, fixed-size batches, lazy test access and
warm-only cold-query history. These checks are correctness evidence, not AUC
or training-throughput evidence.
