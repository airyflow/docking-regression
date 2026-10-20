# Research log

Chronological record of what was done and why. Findings are distilled in
[FINDINGS.md](FINDINGS.md); this file keeps the reasoning and the dead ends.

---

## 2026-10-16 — Repo created

**Why not reuse `otter-knowledge`.** The forked IBM repo accompanies
Martínez Galindo et al., *Nat. Commun.* 17:8081 (2026), on molecular
multimodality. Audit of the fork found it carries the older AAAI-2024
Otter-Knowledge code and none of the paper's experiments: only 2 of 7 modality
encoders (Morgan fingerprint, ESM), no knowledge-graph construction, no GNN
pretraining, no FFNN baseline, no ADMET pipeline, no analysis code. The paper's
own Code Availability section concedes this — it depends on IBM proprietary
infrastructure (LSF, Deep Search) and some pretrained models were deleted.

More decisively, the architecture does not fit the task. Every docking pool
here is a **single receptor**, so the protein tower is a constant column and
the knowledge-graph GNN has no drug–protein edges to exploit. What survives is
conceptual, not code: an MLP regressing a score from a molecule embedding, and
late fusion by averaging per-backbone predictions.

Decision: new standalone repo, molecule-only regression.

### Data audit

Verified on disk rather than assumed, because a silent misalignment trains
happily and means nothing. Three traps found:

1. **`enhits` scores are not in the pool directory at all.** Embeddings live in
   `embed_shuffled/`; the only scores are in
   `al-molecular/data/EnamineHTS_scores.csv.gz`, which is aligned to the
   *unshuffled* `embed/enhits_large_smiles.txt` and is **sorted by score**.
   Joining naively yields scrambled labels. `shuffle_permutation.npy` maps
   embedding row *i* to original row `perm[i]`; confirmed
   `shuf[i] == orig[perm[i]]` and, independently via a SMILES→score dict, zero
   mismatches over 50k random rows.
2. **`enhits` Uni-Mol2 is 768-dim, not 1536** as on `ampc`/`d4` — a different
   extraction. Not comparable across pools.
3. **`enhits` names GROVER `grover_embeddings.npy`**, the other pools
   `grover3400_embeddings.npy`.

Also: `enhits` row count is **2,104,318**, not the 2,104,250 in the pool notes.
`ampc` and `d4` are directly row-aligned (sampled and confirmed).
`enamine_real` left out of the registry entirely — its embeddings are
unstitched per-chunk files (MHG-GED 339/1000, Uni-Mol 146/1000).

All three traps are encoded in `pools.py` so they need not be remembered.
`prepare.py` checks every SMILES row against the reference file and refuses to
write on a mismatch; all 2,104,318 `enhits` rows passed.

### Pipeline

`prepare` → `splits` → `train` → `fuse`, plus `report` and `sizecontrol`.
Design points worth keeping:

- Nothing is loaded whole; `data.gather()` reads contiguous slabs where sorted
  indices are dense enough to beat per-row reads (Lustre punishes scatter).
- Failed docks dropped at **split** time, so train/val/test share one universe
  and the filter is recorded beside the indices. `d4` has ~5% non-negative
  scores (max +328); `enhits` has 16.
- Test split read once, after the best epoch is chosen on validation.
  Predictions saved per run — this is what makes fusion and size control
  retroactive, with no retraining.

### Smoke test, and the finding that reframed the project

First runs (`enhits`, 200k train, 50k test, 5 epochs) gave MHG-GED 0.938
Pearson against MolFormer 0.629. A gap that large is a leakage smell, so it was
checked: single raw MHG-GED dimensions correlate up to 0.65 with the score.

That led to the real explanation. **Heavy atom count alone correlates −0.66
with the DOCK score.** DOCK scores scale with molecular size. MolFormer and
SMI-TED were therefore performing no better than counting atoms.

Added `descriptors.py` as a pseudo-backbone: ~12 RDKit descriptors through the
identical split, model and metrics. At full budget it reaches Pearson 0.891 —
a demanding floor that costs nothing to compute. Every backbone comparison now
reports it.

### Refactor: prefix caching

Originally each training budget re-gathered from the pool, so a 4-budget sweep
would have walked GROVER's 28.6 GB file four times. Since split indices are
stored in **random order**, budget *N* is exactly the first *N* rows of the
full cache. Now each backbone is materialised once and budgets are prefix
reads. Added early stopping (patience 6) so runs end when validation stops
improving rather than at an arbitrary epoch — the fairness problem in the
smoke test.

Evaluation also moved to the **full** 105,215-row val/test splits (the smoke
test used 50k), so smoke-test numbers are not comparable to later ones.

### Sweep launched

24 runs: 6 backbones × 4 budgets (50k / 200k / 1M / all 1,893,872), 40 epochs
max, patience 6, resumable. Small-dim backbones first so GROVER's gather does
not block the cheap results. Descriptors complete; MolFormer building its cache.

### Size control

Built `sizecontrol.py`, which re-analyses saved predictions — no retraining.
Three independent ways of removing size: partial correlation, residual
correlation after a degree-3 fit, and metrics within equal-count heavy-atom
bins.

First result **contradicted the working hypothesis**, usefully. Size alone
scores raw *r* = 0.659 but collapses to within-bin *r* = 0.148 and EF@1% = 2.6
— as it must, being near-constant inside a size bin. Descriptors at full budget
go 0.891 raw → 0.797 partial → 0.800 within-bin, EF@1% 40.7 → 30.1.

So the descriptor baseline is **not** a size artifact. It retains strong
genuine signal (logP, TPSA, rings) after size is removed. The earlier "it's all
size" framing was wrong. This makes the baseline more demanding, not less, and
it means the bar an embedding must clear is a real one.

---

## 2026-10-16 (later) --- Two I/O bugs, and the first converged backbone

### Bug 1: `np.asarray()` on a memmap does not copy

`_maybe_load()` printed "loading train into RAM" and called `np.asarray(arr)`.
For a `np.memmap` --- an `ndarray` subclass --- `asarray` returns a **view
sharing the mapped buffer**, not a copy. So nothing was ever loaded, and every
epoch page-faulted against Lustre for the full array.

Symptom: the MolFormer full-budget run sat for **2.5 hours without completing
a single epoch**. After switching to `np.array()`: load ≈ 12 s, epochs
15--22 s. Confirmed the semantics directly before fixing rather than guessing.

This also explains the smoke test's 71 s first epoch dropping to 3 s --- that
was the page cache warming, not the model speeding up.

### Bug 2: the cache builder scattered its writes to disk

`gather()` scatters writes to restore `idx` order. Doing that straight into a
disk-backed memmap turns one streaming write into millions of scattered page
writes. MolFormer's 5.8 GB cache took **4.7 hours** (01:38--06:22).

Fixed by building in RAM and writing once with `np.save`, with a fallback to
the memmapped path above a RAM budget. The node has 1.2 TB free, so even
GROVER's 25.7 GB build stays in RAM.

Compounded, these two would have put GROVER's cache near a day. Existing caches
remain valid --- the on-disk format did not change, only how it is produced.

**Lesson worth keeping:** both bugs made the pipeline *slow*, not *wrong*, which
is why neither surfaced in the smoke test. Timing anomalies were the only
signal. A run that stalls for hours deserves a diagnosis, not patience.

### First converged backbone: MolFormer

Properly converged with early stopping on validation, full 105,215-row test set
(not the 5-epoch smoke test):

| Train rows | MolFormer *r* | Descriptors *r* | Gap |
|---:|---:|---:|---:|
| 50,000 | 0.5948 | 0.8748 | -0.280 |
| 200,000 | 0.6265 | 0.8802 | -0.254 |
| 1,000,000 | 0.6502 | 0.8872 | -0.237 |

MolFormer sits far below the baseline and closes the gap only slowly: 20x more
data buys it +0.055 Pearson against the baseline's +0.024. The smoke-test
numbers for this backbone (0.629 at 200k) were close to correct after all ---
the 5-epoch run was undertrained, but not by much.

One backbone, one seed, one fixed head. This is precisely the result the
ridge / boosted-trees check exists to defend.

---

## Open threads

- Sweep running; MolFormer/SMI-TED convergence will decide whether the headline
  is "embeddings lose to descriptors" or an ordinary backbone comparison.
- Mean fusion collapsed in the smoke test (see FINDINGS F4) — needs re-testing
  on converged runs.
- Not yet done: multiple seeds, a second model family (ridge / boosted trees)
  to defend against the "you mis-tuned the MLP" objection, intermediate fusion,
  literature check on whether the descriptor control is already published.

---

## 2026-10-16 (evening) --- A silent bad number, and the pivot

### F5 was wrong, and why

The first version of F5 said "embeddings lose to descriptors," generalised from
MolFormer alone. With four backbones converged the result is **bimodal**:
Uni-Mol2 (0.956) and MHG-GED (0.952) beat the descriptor baseline (0.891)
substantially, while SMI-TED (0.689) and MolFormer (0.661) fall far below it.
The split is by representation family --- 3D/graph versus SMILES language model
--- not by dimensionality, since all three 768-dim models span the whole range.

Lesson: one data point drawn from a bimodal distribution is worse than none,
because it reads as a trend.

### A 3-epoch run leaked into the results tables

The 50k descriptor row had been reported as Pearson 0.8628. That figure came
from a **3-epoch debug invocation** run while verifying the prefix-caching
refactor. The sweep is resumable by "does metrics.json exist", so it skipped
the run rather than training it properly. Converged, the number is **0.8748**
(EF@1% 37.5 not 35.7, EF@0.1% 238 not 171).

Corrected in FINDINGS.md, the size-control table and the paper. The learning
curve conclusion is unchanged --- in fact slightly strengthened, since the
baseline now gains only +0.016 Pearson across 38x more data rather than +0.028,
so it saturates even earlier.

**Fix:** resumability now compares the recorded `epochs` and `patience` against
what is being requested, instead of treating any `metrics.json` as done.
Skipping on file existence alone silently promotes debug runs into results.

### Pivot: scaling as the contribution

The literature review (LITERATURE.md) showed the "descriptors beat embeddings"
framing was already taken. The available pools --- 2M, 98M, 116M, 1.5B --- span
five orders of magnitude with six backbones, which no published work covers;
Deng and Praski work at 10^2--10^4 compounds, MolPAL at 10^8 with one
representation family.

New framing: **representation family, not dimensionality, determines docking
surrogate quality --- and here is how that holds from 10^4 to 10^8 molecules.**
Descriptors become the calibrated reference line, not the headline.

Constraint to keep honest: this is **not one curve**. Each pool is a different
target and the 1.5B pool uses Glide rather than DOCK3.7, so it is a robustness
check rather than the curve's top end. `d4` alone gives 50k--116M on one target
with six backbones, already far beyond published work.

### Infrastructure for 10^8 rows

Added `--cache-budget`: caching a full d4 split would be 1.5 TB for GROVER.
Budgets are now prefixes of a capped cache rather than of the whole split.

---

## 2026-10-16 (afternoon) --- DOCKSTRING, two corrections, and a bug that came back

### Discovering `al-eval-framework`

Searching for DOCKSTRING turned it up not in `FusionAL` but in
`al-eval-framework`: the dataset, four prepared targets (PARP1, ESR2, F2,
EGFR), a MolPAL-based AL loop, C1--C4 diagnostics and a paper draft, *"What
Makes a Molecular Representation Good for Active Learning?"*. The AL-loop work
this project had been calling a gap already existed.

Its headline result contradicts its own stated hypothesis. DESIGN.md predicts
static performance will **not** predict AL efficiency; the measured Spearman
between static-RMSE rank and AL-AUDC rank is 0.886 pooled and 1.000 on EGFR.
Robust to dropping the derived `fusional_*` variants and to dropping morgan,
so not an artifact of either. Underpowered though --- 4 independent
representations gives p ~ 0.33.

### Correction 1: the ordering disagreement is real, not a bug

The two repos rank representations oppositely: MolFormer near-best and Uni-Mol
poor on DOCKSTRING, the reverse on EnamineHTS. Running this repo's protocol on
their data reproduced *their* ordering, so neither pipeline is wrong --- the
datasets genuinely differ.

### Correction 2: "DOCKSTRING is flat" was wrong, twice over

First conclusion: representation choice is irrelevant on DOCKSTRING (spread
0.019 vs EnamineHTS's 0.349), and `al-eval-framework` had therefore tested its
hypothesis on a dataset unable to falsify it.

That was measured at 45,000 training rows. Reading `src/metrics/static.py`
showed `train_frac=0.005` --- **250 molecules**. At that budget the DOCKSTRING
spread is **0.297**, comparable to EnamineHTS. The criticism was withdrawn in
both repos' docs.

The real contrast is convergence: DOCKSTRING reaches 0.89--0.91 for everything
by 45k, while EnamineHTS keeps MolFormer at 0.66 and Uni-Mol2 at 0.956 even at
1.9M rows. The ordering also reverses with budget --- descriptors 1st at 250
rows, last at 45k.

**Lesson.** Compare at the regime the other experiment used. Both errors came
from assuming a default instead of reading four lines of their code. Reading
the sibling repo's code should have preceded criticising its design.

### The `np.asarray` memmap bug, twice more

`np.asarray()` on a memmap returns a **view**, not a copy. Already fixed once
in `train.py`; the same mistake was then written into `gather_stream` and left
in `gather`'s slab path. There the fancy-index on the following line
page-faults scattered rows --- exactly the access pattern those functions exist
to avoid.

Measured on d4: ~23 MB/s before, ETA 2.9 hours for one pass. After switching to
`np.array()`: 408,916 of 900,000 rows in 1.7 minutes. Both sites now carry a
comment explaining why.

### Process errors worth not repeating

- **Misread task notifications twice.** Wrapping `nohup ... &` *inside* a
  `run_in_background` command means the harness tracks the launcher, which
  exits immediately, so the completion notification fires while the real job is
  still running. Acting on one of those launched a training run against a cache
  that did not exist yet, putting two processes on the same 238 GB file. Do not
  nest `nohup &` inside `run_in_background`; verify with `ps` before acting.
- **A debug run leaked into results** (logged earlier the same day). The sweep
  now compares recorded `epochs`/`patience` rather than treating any
  `metrics.json` as done.

### New results logged

- **F7 rewritten** --- DOCKSTRING saturates, EnamineHTS does not; the
  spread-vs-budget table.
- **F8 added** --- the family split survives ridge, GBDT and MLP, so it is a
  property of the representations rather than the head. Uni-Mol2's signal is
  mostly linearly accessible (ridge 0.9368); GROVER scores *higher* under ridge
  than the MLP, confirming a calibration problem in its MLP run.
- **`docs/SOURCE_PAPER.md`** --- full analysis of the Nature Communications
  paper, including its Q1/Q2 confound and asymmetric training protocols.
- **`docs/CROSS_REPO_SYNTHESIS.md`** and
  **`al-eval-framework/CROSS_REPO_FINDING.md`** --- the two-repo picture, with
  the retraction kept visible.

### In flight

Uni-Mol v1 (512d) vs Uni-Mol2 (1536d) on d4 --- same oracle, same pool, same
split --- isolating backbone version from oracle and chemistry. If v1 tracks
v2, F5's family story holds and the dataset difference is oracle or
distribution. If v1 is far weaker, part of what F5 calls a family split is
model quality, and `al-eval-framework`'s poor Uni-Mol result is explained by
version alone.

---

## 2026-10-17 --- Seeds, versions, and the fusion suite

**GROVER full-budget seeds amended F11.** The day before, F11 had replaced an
earlier "calibration problem" description with "optimisation instability".
Five full-budget seeds showed the first reading was right: Spearman sd 0.0001
and EF@1% sd 0.38 against Pearson sd 0.0494 and RMSE doubling. The ranking is
deterministic; only the value mapping moves. **Two corrections in two days on
the same point** --- the lesson is that "unstable" needs a metric attached.

**d4 settled the Uni-Mol version question (F12).** v2 (1536-d) beats v1
(512-d) by ~0.07 Pearson at every budget, under one oracle on one pool. That
points *opposite* to the EnamineHTS screen, where the narrower 768-d beat this
pipeline's own 1536-d extraction. Not a contradiction --- the two "Uni-Mol2"
files differ in extraction --- but it means neither width nor version ordering
transfers between pools.

**Fusion suite built and run (F13).** Feature-level (native concat, PCA-128)
and prediction-level (mean, z-mean, rank, NNLS, ridge, best-single) with
everything fitted on validation. Every arm beat the best single model. Feature
concat of the right pair beat every prediction-level method, which runs
opposite to the source paper's late-beats-intermediate conclusion.

## 2026-10-18 --- A fusion sweep, and two bugs of my own making

**F14: the fusion gain is the one thing that is not budget-conditional.**
+0.004 to +0.011 across a 4,000-fold range, no trend, while unimodal
performance moves 0.88 to 0.95 over the same range. A prediction made before
running --- that PCA-128 would win at low budgets as regularisation --- was
**wrong**: native concat wins or ties everywhere, a third confirmation of F10.

**Bug: `featurefuse.py` truncated shared caches.** It cached under the same tag
as `train.py` but with only `max_train` rows, so running it at 200k overwrote
the full-split caches for MolFormer and SMI-TED. Fixed to cache the same rows
`train.py` would, then slice.

**Bug: no atomic cache writes.** Sixteen parallel workers each found a stale
GROVER cache and began rebuilding the same 25.8 GB file. The result was a
corrupt 9.8 GB file that could not be opened. Fixed with write-to-temp plus
`os.replace`. Added `scripts/check_caches.py` to audit readability and row
counts, because a corrupt cache otherwise surfaces much later as what looks
like a modelling failure.

**A contributing cause that was not mine:** two copies of the same sweep were
briefly submitted and ran concurrently on different nodes, so 32 workers raced
on one file. Both causes are now handled.

**Slurm scaffolding added** to this repo (config.env, slurm/_preamble.sh),
mirroring al-molecular's pattern. First submission failed because I used
`$(dirname "$0")` to source the preamble --- sbatch copies the script to the
spool directory while starting it in SLURM_SUBMIT_DIR, so a repo-relative path
is required. al-molecular's scripts already did it correctly and I had read
that line.

## 2026-10-19 --- The subset control, and a claim of mine that was wrong

**F15: training-subset resampling, 90 cells, five draws each.** The control
Story A had been missing.

- Top and bottom tiers hold: Uni-Mol2 rank 1 and MHG-GED rank 2 in 5 of 5
  draws at every budget; both SMILES models last in 5 of 5.
- **GROVER and descriptors swap ranks at every budget**, so the middle of the
  ranking is not resolvable. Story A narrows from "ranking reverses with
  budget" to "the extremes are stable and well separated; the middle is not".
- **I was wrong**, repeatedly and in writing, that training-subset draw is
  "the dominant source of variability" at small budgets. For five of six
  backbones it is ~0.003, the same order as initialisation variance. It is
  true for exactly one backbone, GROVER, which is unstable under both axes.

**Process notes.** Thirteen cells failed in the first Slurm run because of the
cache bugs above; they were finished locally on CPU after the fixes. An
orphaned `while pgrep` waiter from the d4 chain was found still running after
2 days 22 hours and killed. A Slurm resubmission sat in `PD (Priority)` for
over 30 minutes for 13 cells of small-MLP work --- a whole 64-CPU node with
200 GB was far more than the job needed.

**Recurring failure mode worth naming:** several times I reported a job as
"running" or "N of M complete" by counting run *directories* rather than
`metrics.json` files. A killed run leaves the directory behind. Count outputs,
not folders.
