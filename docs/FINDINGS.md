# Findings

Current state of what the experiments show. Process and reasoning in
[LOG.md](LOG.md). Last updated 2026-10-16, sweep in progress.

All numbers below: pool `enhits` (EnamineHTS, DOCK3.7), split
`seed0_v0.05_t0.05`, test n = 105,215, single seed.

---

## F1 — Three row-alignment traps, all silent

Confirmed on disk, all now handled in `pools.py` / `prepare.py`:

- `enhits` embeddings are shuffled relative to the only available scores, which
  are themselves **sorted by score**. A naive join trains on scrambled labels
  without erroring.
- `enhits` Uni-Mol2 is **768-dim, not 1536** — a different extraction from
  `ampc`/`d4`. Never compare `unimol2` across those pools.
- `enhits` row count is **2,104,318**, not 2,104,250.

Verification: all 2,104,318 SMILES checked row by row during `prepare`, plus an
independent SMILES→score dict check on 50k random rows (zero mismatches).

**Why it matters:** any result produced before this was verified would have
been noise presented as signal.

---

## F2 — The descriptor baseline is strong, and it saturates early

Twelve RDKit descriptors (heavy atoms, MW, logP, TPSA, rings, rotatable bonds,
HBD/HBA, fraction sp³, halogens, aromatic rings, heteroatoms):

| Train rows | Pearson | Spearman | RMSE | EF@1% | EF@0.1% |
|---:|---:|---:|---:|---:|---:|
| 50,000 | 0.8748 | 0.8613 | 0.4611 | 37.5 | 238 |
| 200,000 | 0.8802 | 0.8669 | 0.4517 | 38.4 | 248 |
| 1,000,000 | 0.8872 | 0.8744 | 0.4391 | 38.7 | 257 |
| 1,893,872 | 0.8907 | 0.8783 | 0.4326 | 40.7 | 286 |

38× more data buys **+0.028 Pearson**. The curve is flat past ~200k, which is
what a 12-feature model should do: it runs out of capacity long before it runs
out of data.

**Why it matters:** this is the floor every learned embedding must clear. A
backbone needing millions of rows to reach 0.89 has not earned its storage,
because 12 free features get there on 200k.

---

## F3 — The baseline is *not* a size artifact

DOCK scores scale with molecular size, so the obvious worry was that everything
above is an elaborate way of counting atoms. Three independent controls say
otherwise (`sizecontrol.py`, computed on saved predictions):

| Model | raw *r* | partial *r* | residual *r* | within-bin *r* | within-bin EF@1% |
|---|---:|---:|---:|---:|---:|
| size only (−heavy atoms) | 0.6592 | 0.000 | 0.000 | 0.1478 | 2.6 |
| descriptors, 50k | 0.8748 | 0.7647 | 0.7598 | 0.7692 | 25.8 |
| descriptors, 200k | 0.8802 | 0.7756 | 0.7710 | 0.7797 | 28.1 |
| descriptors, 1M | 0.8872 | 0.7896 | 0.7853 | 0.7933 | 29.6 |
| descriptors, all | 0.8907 | 0.7966 | 0.7924 | 0.8000 | 30.1 |

Heavy atom count alone correlates −0.66 with the score but is nearly useless
*within* a size stratum (within-bin *r* = 0.15, EF 2.6 against a random
baseline of 1.0) — exactly as expected for a feature that is near-constant
inside its own bin.

Descriptors lose only ~0.09 Pearson to the size control and keep within-bin
EF@1% of 30. The non-size descriptors (logP, TPSA, ring counts) are carrying
real signal.

**This contradicted the working hypothesis.** The initial read was "descriptors
only look good because they encode size." Wrong. Roughly 26% of raw enrichment
is size-driven (EF 40.7 → 30.1); the rest is genuine chemistry.

**Why it matters:** the bar is legitimate, so "embeddings fail to beat
descriptors" — if it survives convergence — is a claim about representation
quality, not an artifact of a degenerate baseline. Report raw *and*
size-controlled numbers for every backbone.

---

## F4 (SUPERSEDED by F6) — Naive mean fusion is dragged down by its worst member

**Superseded by F6**, which measures this on converged models and derives the
governing rule. Retained because the direction was right and the early signal
is what prompted the proper analysis. The numbers below are from a 5-epoch
undertrained sweep and must not be cited.

| Combination | Pearson |
|---|---:|
| MHG-GED alone | 0.938 |
| MHG-GED + SMI-TED | 0.890 |
| MHG-GED + MolFormer | 0.887 |
| MHG-GED + MolFormer + SMI-TED | 0.839 |

Averaging a strong model with two weak ones **loses 0.10 Pearson**.

Late fusion in the multimodality literature — including the Nature Comms paper
this work started from — assumes roughly comparable members. That assumption
fails when backbone quality is this uneven. Options: rank fusion
(`--mode rank`), validation-weighted averaging, or fusing only models that
clear the descriptor baseline.

**Why it matters:** "more modalities help on average" does not transfer here
without a gating rule. That is itself a result worth reporting.

---

## F5 (CORRECTED) --- Representation *family* splits the field, not dimensionality

**An earlier version of this finding claimed "embeddings lose to descriptors."
That was based on MolFormer alone and was wrong.** With four backbones
converged, the picture is bimodal.

Full budget (1,893,872 train rows), test n = 105,215:

| Backbone | Type | Dim | Storage | Pearson | RMSE | EF@1% | vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Uni-Mol2 | 3D conformer | 768 | 6.5 GB | **0.9564** | **0.2779** | **58.6** | +0.066 |
| MHG-GED | hypergraph grammar | 1024 | 8.6 GB | 0.9516* | 0.2925* | 56.7* | +0.064 |
| **descriptors** | --- | 12 | 0 | 0.8907 | 0.4326 | 40.7 | --- |
| SMI-TED | SMILES LM | 768 | 6.5 GB | 0.6892 | 0.6895 | 24.3 | -0.202 |
| MolFormer | SMILES LM | 768 | 6.5 GB | 0.6611 | 0.7152 | 19.7 | -0.230 |

\* at 1M rows; full-budget run still in flight.

**The split is not by dimensionality.** All three 768-dim models appear --- one
at the top (Uni-Mol2) and two at the bottom (SMI-TED, MolFormer). What
separates them is modality:

- **Winners encode structure**: 3D conformers (Uni-Mol2) and molecular graphs
  (MHG-GED). Both substantially beat the descriptor baseline.
- **Losers are both SMILES language models**: SMI-TED and MolFormer. Both fall
  far below 12 free descriptors.

This is consistent with docking being a 3D physics calculation evaluated over a
posed conformer. Scope: SMILES does encode connectivity, so this is not
evidence that string representations are inherently unsuitable --- only that
these two frozen SMILES-language embeddings expose substantially less
DOCK3.7-relevant signal on this pool.

**Why it matters.** It is also not what the general representation-learning
literature predicts. Praski et al. and Deng et al. (see
[LITERATURE.md](LITERATURE.md)) find pretrained embeddings roughly *tied* with
ECFP on experimental property tasks. Neither reports a clean family split, and
neither tested docking scores. A bimodal outcome this wide --- Uni-Mol2 nearly
halving the baseline's RMSE while MolFormer trails it by 0.23 Pearson --- is
docking-specific and appears to be new.

### Scaling behaviour

| Backbone | 200k | 1M | 1.89M | trend |
|---|---:|---:|---:|---|
| Uni-Mol2 | 0.9426 | 0.9544 | 0.9564 | saturating, high |
| descriptors | 0.8802 | 0.8872 | 0.8907 | saturated |
| SMI-TED | 0.6609 | 0.6827 | 0.6892 | still climbing |
| MolFormer | 0.6265 | 0.6502 | 0.6611 | still climbing |

The winners have saturated; the losers have not. A crude log-linear
extrapolation puts MolFormer's closure rate near 0.025 Pearson per decade,
implying it would need of order \(10^9\) more molecules to draw level --- i.e.
never. That is a two-point extrapolation over 1.3 decades and should be
treated as a motivating guess, not a result. Settling it empirically is what
the 10^8-scale pools are for.

**Caveats.** One seed, one fixed prediction head. The ridge / boosted-trees
check still matters --- now in both directions, since it must confirm that
Uni-Mol2's advantage is also not an artifact of the head.

**Lesson.** The earlier version of F5 generalised from a single backbone. One
data point in a bimodal distribution is worse than no data point, because it
reads as a trend.

---

## F6 --- Multimodality helps, but only for one pair, and the rule is exact

Late fusion over the six converged full-budget models (test n = 105,215):

| Combination | Pearson | RMSE | EF@1% |
|---|---:|---:|---:|
| **MHG-GED + Uni-Mol2** | **0.9605** | **0.2650** | **61.6** |
| Uni-Mol2 alone | 0.9564 | 0.2779 | 58.6 |
| + GROVER (3-way) | 0.9567 | 0.2770 | 60.7 |
| + descriptors (4-way) | 0.9524 | 0.2915 | 58.4 |
| descriptors + Uni-Mol2 | 0.9415 | 0.3228 | 53.2 |

The best subset is **2 of 6**. Every further addition degrades it. Descriptors
score 0.891 alone yet *lower* Uni-Mol2 when averaged with it.

### The governing rule

Error magnitude and error correlation against Uni-Mol2:

| Model | error sigma | error corr. w/ Uni-Mol2 | sigma ratio | threshold | helps? |
|---|---:|---:|---:|---:|:--:|
| MHG-GED | 0.284 | 0.777 | 1.02 | 1.12 | yes |
| descriptors | 0.433 | 0.634 | 1.56 | 1.21 | no |
| GROVER | 0.361 | 0.656 | 1.30 | 1.20 | no |
| MolFormer | 0.715 | 0.379 | 2.57 | 1.39 | no |

For an equal-weight average of two models to beat the better one,

    sigma_2 / sigma_1  <  -rho + sqrt(rho^2 + 3)

This predicts every observed outcome, and the implied RMSE for the winning
pair (0.265) matches the measured 0.2650.

**The interesting part:** MolFormer's errors are the *least* correlated with
Uni-Mol2's (rho = 0.379), so it carries genuinely complementary information.
It is simply too inaccurate to pay for the variance it adds. Equal-weight
averaging cannot exploit complementarity it must pay for in error variance.

**Why it matters.** "More modalities help on average" --- the central claim of
the multimodality benchmark this project started from --- does not hold here,
and the reason is quantifiable rather than anecdotal. It is a statement about
member-quality homogeneity, not about modality count. Validation-weighted or
precision-weighted fusion is the obvious remedy and is not yet tested.

### Complementarity structure

Prediction correlations show two tight clusters: the three structure-based
models correlate 0.955--0.979 with each other, and the two SMILES models 0.931
with each other, while cross-family correlation is only 0.67--0.72. The
families really are distinct information sources --- the weaker one is just
not accurate enough to contribute under equal weighting.

---

## F7 (CORRECTED TWICE) --- DOCKSTRING saturates; EnamineHTS does not

**Correction history.** v1 claimed "representation choice is irrelevant on
DOCKSTRING." That was measured at 45,000 training rows. `al-eval-framework`'s
`static_eval` uses `train_frac=0.005` --- **250 molecules**. Compared at the
right budget the claim inverts.

### Representation spread vs training budget, DOCKSTRING PARP1 / Vina

| Train rows | descriptors | MolFormer | GROVER | Uni-Mol v1 | Morgan | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | **0.8447** | 0.8037 | 0.6133 | 0.7471 | 0.5475 | **0.297** |
| 1,000 | **0.8711** | 0.8413 | 0.8087 | 0.7997 | 0.7002 | 0.171 |
| 5,000 | **0.8771** | 0.8701 | 0.8663 | 0.8479 | 0.8134 | 0.064 |
| 45,000 | 0.8905 | 0.8957 | **0.9095** | 0.8905 | 0.8995 | 0.019 |

At 250 rows the spread is 0.297 --- comparable to EnamineHTS's 0.349. DOCKSTRING
separates representations perfectly well; it just **converges** by 45k.

The ordering also reverses completely with budget. Descriptors go 1st to last;
Morgan goes last to 2nd. Any claim of the form "representation X is better than
Y for docking" is budget-conditional unless the budget is stated.

### The actual contrast

| Dataset | spread @ ~50k | spread @ max | converges? |
|---|---:|---:|---|
| DOCKSTRING PARP1 / Vina | 0.019 (45k) | 0.019 | **yes, by 45k** |
| EnamineHTS / DOCK3.7 | 0.349 (50k) | 0.295 (1.9M) | **no** |

On DOCKSTRING everything reaches 0.89--0.91. On EnamineHTS, MolFormer and
SMI-TED plateau at 0.66--0.69 while Uni-Mol2 reaches 0.956, and the gap
survives 1.9M training rows.

So the finding is not "representation matters here and not there" but **"the
Vina task has a low ceiling every representation reaches; the DOCK3.7 task has
a higher ceiling only structure-aware representations reach."**

**Eliminated as explanations:** protocol (this pipeline reproduces
al-eval-framework's ordering on their data), training-set size (swept above),
and the size confound (heavy-atom correlation -0.659 vs -0.648, nearly
identical). **Still open:** the oracle, the molecule distribution
(ChEMBL-derived drug-like vs screening-diverse), and backbone version --- the
last being tested on d4, which carries both Uni-Mol v1 and Uni-Mol2 under one
oracle.

### Per-target size dependence (side result)

| Target | heavy-atom vs score | variance explained |
|---|---:|---:|
| EnamineHTS / DOCK3.7 | -0.659 | 43.5% |
| PARP1 / Vina | -0.648 | 42.0% |
| F2 / Vina | -0.623 | 38.8% |
| **ESR2 / Vina** | **-0.199** | **4.0%** |

ESR2 is nearly size-free --- the cleanest target for isolating chemical signal.

**Consequence for the sibling repo.** v1 of this finding was used to argue that
`al-eval-framework` had tested its decoupling hypothesis on a dataset unable to
falsify it. **That argument was wrong.** At their budget the spread is 0.297,
ample room for two rankings to diverge. Their coupling result stands on its own
terms; see [CROSS_REPO_SYNTHESIS.md](CROSS_REPO_SYNTHESIS.md).

**Lesson.** Compare at the regime the other experiment actually used. Both
errors here came from assuming a default rather than reading the code --- the
fix was four lines of `static.py`.

---

## F8 --- The family split is independent of the prediction head

The main results hold the head fixed (MLP 1024-512), which invites the
objection that the architecture suits some representations better than others.
Refitting on identical cached embeddings and splits:

| Backbone | MLP (1.9M) | Ridge (1.9M) | GBDT (200k) |
|---|---:|---:|---:|
| Uni-Mol2 | 0.9564 | 0.9368 | 0.9356 |
| MHG-GED | 0.9543 | 0.9198 | 0.9421 |
| GROVER | 0.9267 | 0.9339 | 0.9350 |
| descriptors | 0.8907 | 0.8455 | 0.8769 |
| SMI-TED | 0.6892 | 0.6336 | 0.6517 |
| MolFormer | 0.6611 | 0.5987 | 0.6096 |

Three heads with very different inductive biases --- linear, axis-aligned
trees, and a neural net --- give the same ordering and preserve the
structure-vs-sequence gap. The "you mis-tuned the MLP" objection is answered.

Two incidental results:

- **Uni-Mol2's signal is mostly linearly accessible**: ridge alone reaches
  0.9368, so the MLP contributes only +0.02.
- **GROVER scores higher under ridge (0.9339) than under the MLP (0.9267)**,
  confirming that its MLP run had a calibration problem rather than a
  representation one --- consistent with its anomalous Pearson/Spearman gap
  (0.9267 vs 0.9464) noted in F5.

---

## F9 --- Independently corroborated by the active-learning project

`/N/slate/mengjing/repos/al-molecular` runs an AL benchmark on the **same
ENHITS embedding files**, with a different metric (fraction of true top-1000
recovered), a different method (5-backbone ensemble, leave-one-out) and a
separate code path. It reached the same conclusion about the same backbone,
without reference to this work.

Their route: Uni-Mol2's ENHITS embedding has effective rank 2.2 of 768, 82% of
variance in one PC, mean pairwise cosine 0.984 --- apparently degenerate. They
hypothesised it was dead weight and **tested it**: dropping the slot costs
**-0.063 recall**, larger than every deliberate technique in their project
combined. Their post-mortem: "effective rank was the wrong instrument ... a
representation whose dominant direction is score-aligned scores low while
being highly informative."

Our route: the same file is the **best single static predictor** at Pearson
0.9564 (F5).

**Our size control answers the obvious objection to their post-mortem.** If the
dominant direction were simply molecular size (which correlates -0.659 with the
score here), the backbone would be an elaborate atom counter. It is not: after
partialling out heavy-atom count, Uni-Mol2 retains partial *r* = 0.9215, the
highest of any backbone, and loses the least to the control (-0.035, against
-0.17 for the two SMILES models). It carries the most **non-size** signal.

**Why it matters.** Two unrelated instruments agreeing on the same artifact is
the strongest validation available without new data. It also supplies a better
diagnostic than spectrum entropy for "is this embedding informative":
score-aligned signal surviving a size control, computed from held-out
predictions at no extra training cost.

**A prediction was registered** in their document (§8.5c) before their four
pending leave-one-out arms report: because Uni-Mol2 and MHG-GED predictions
correlate 0.979, dropping MHG-GED should cost within ~0.02 of dropping
Uni-Mol2, and dropping either SMILES model should cost less. If instead only
Uni-Mol2 is expensive to drop, static redundancy does not transfer to ensemble
dynamics --- which would itself be worth knowing.

**Shared dependency to carry.** Every `enhits` number here and every number in
their results document rests on the same inherited embedding files, which their
provenance work shows were extracted by a third party with different settings
(768-d Uni-Mol2 rather than this pipeline's 1536-d; GROVER at 1600-d in the
older set). A re-extraction moves both projects' results together.

---

## F10 --- Variance concentration is anti-correlated with information

The ENHITS Uni-Mol2 embedding has 82% of its variance in one principal
component and effective rank 2.2 of 768, yet it is the best single predictor
here (F5) and the most valuable backbone in the sibling AL project (F9). Two
diagnostics settle why.

### The dominant direction is molecular size, not score

First 64 PCs of 40,000 test rows:

| embedding | PC1 var | eff. rank | PC1 vs heavy atoms | PC1 vs score | PC1 vs row index |
|---|---:|---:|---:|---:|---:|
| **Uni-Mol2 768** | 81.9% | 2.1 | **0.802** | 0.228 | 0.0007 |
| Uni-Mol2 1536 | 23.1% | 17.9 | 0.102 | 0.014 | 0.0043 |
| MHG-GED | 7.9% | 41.0 | 0.063 | 0.143 | 0.0013 |
| MolFormer | 14.0% | 26.9 | 0.148 | 0.092 | 0.0036 |

**No leakage**: PC1 correlates ~0.001--0.004 with row index for every backbone,
which matters because the source pool was score-sorted before shuffling.

This **corrects the mechanism proposed in `al-molecular` §8.5b**, which held
that effective rank fails when "a representation whose dominant direction is
score-aligned scores low while being highly informative". The dominant
direction is **size**-aligned (0.802), not score-aligned (0.228). Since size
itself correlates -0.659 with the score, that inference reached the right
conclusion by the wrong route.

### Truncation: 90% of the variance carries a third of the signal

PCA truncation on the 768-d embedding alone, varying only *k* --- one
embedding, so no provenance confound. 200,000 training rows:

| top-*k* PCs | cumulative variance | Pearson | EF@1% |
|---:|---:|---:|---:|
| 1 | 82% | **0.284** | 1.9 |
| 2 | 90% | **0.328** | 4.5 |
| 4 | — | 0.629 | 14.1 |
| 8 | — | 0.813 | 27.3 |
| 16 | — | 0.912 | 42.9 |
| 32 | — | 0.929 | 45.7 |
| 128 | — | 0.945 | 51.9 |
| all 768 | 100% | 0.948 | 53.4 |

The two components holding 90% of the variance deliver roughly a third of the
achievable correlation. Most performance needs ~16--32 components and the
maximum needs the full space: **the signal lives in the low-variance tail**,
and per-feature standardisation is what makes it accessible by rescaling those
directions to unit variance.

**Why it matters.**

1. Spectral diagnostics --- effective rank, participation ratio,
   explained-variance thresholds --- do not identify informative embeddings,
   because dominant variance directions can be nuisance structure. Here they
   are *anti*-correlated with usefulness.
2. **PCA-reducing embeddings on a variance criterion is unsafe.** Retaining
   90% of variance costs two thirds of the performance. Retaining 128
   components (>99%) costs 0.003 Pearson and is fine --- but the variance
   threshold gives no warning about which regime you are in. This is a common
   storage-saving step at pool scale.

**Scope.** One embedding, one pool, one seed. The claim is about variance-based
diagnostics, not about dimensionality: an earlier draft framed this as "width
is not quality" from the 768-d versus 1536-d comparison, which is confounded
by provenance, pooling and checkpoint and cannot support it. The truncation
experiment varies only *k* and is clean.

---

## F11 (AMENDED) --- Seed replication: rankings hold; GROVER is *calibration*-unstable

Five initialisation seeds per cell, EnamineHTS, at the three small budgets
(72 new runs, zero failures). **Note the scope:** `--seed` varies model
initialisation and batch shuffling, *not* which rows enter the training
subset. At 250 rows the subset draw is the larger source of variability and
is not yet measured.

### Rankings are stable

| budget | Uni-Mol2 | MHG-GED | descriptors | GROVER | SMI-TED | MolFormer |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | **0.8794** ±0.0020 | 0.8494 ±0.0027 | 0.8282 ±0.0025 | 0.7409 ±**0.0885** | 0.4808 ±0.0123 | 0.3118 ±**0.0610** |
| 1,000 | **0.9137** ±0.0004 | 0.8927 ±0.0006 | 0.8525 ±0.0005 | 0.8479 ±**0.0502** | 0.5511 ±0.0043 | 0.4614 ±0.0083 |
| 5,000 | **0.9243** ±0.0008 | 0.9167 ±0.0005 | 0.8614 ±0.0002 | 0.6932 ±**0.1567** | 0.6049 ±0.0009 | 0.5513 ±0.0020 |

Uni-Mol2 is rank 1 in **5 of 5 seeds at every budget**; MHG-GED rank 2 in 5 of
5. The only rank changes anywhere involve GROVER.

### The spread persists

| budget | spread (mean ± sd) | range |
|---:|---:|---|
| 250 | 0.568 ± 0.060 | [0.512, 0.666] |
| 1,000 | 0.452 ± 0.009 | [0.444, 0.462] |
| 5,000 | 0.391 ± 0.042 | [0.370, 0.465] |

The EnamineHTS spread does not collapse, and seed variability is far smaller
than the spread itself. **F7's non-convergence claim survives replication** on
this axis.

### GROVER's MLP training is unstable — a fourth independent sign

| budget | GROVER sd | range across seeds |
|---:|---:|---|
| 250 | 0.0885 | 0.622 – 0.823 |
| 1,000 | 0.0502 | 0.759 – 0.882 |
| 5,000 | **0.1567** | **0.459 – 0.824** |

At 5,000 rows GROVER ranges from 0.459 to 0.824 depending only on
initialisation — two to three orders of magnitude more variance than any other
backbone (Uni-Mol2's sd is 0.0008 at the same budget).

This now converges with three earlier anomalies, all for the same backbone:

1. Pearson 0.9267 against Spearman 0.9464 at full budget (F5).
2. Scores *higher* under ridge (0.9339) than under the MLP (0.9267), where
   every other backbone loses under ridge (F8).
3. All three failures of the prospective fusion rule involve it (IV.5).
4. Seed instability, here.

The common explanation is that GROVER's MLP optimisation is unreliable on this
embedding, not that the representation is weak — ridge, which has no
optimisation to fail, places it third.

### AMENDMENT: full-budget seeds show this is calibration, not ranking

The paragraph above replaced an earlier "calibration problem" description with
"optimisation instability". **Five full-budget seeds show the original reading
was right and that amendment over-corrected.**

| seed | Pearson | Spearman | RMSE | EF@1% |
|---:|---:|---:|---:|---:|
| 0 | 0.9267 | 0.9464 | 0.3608 | 57.3 |
| 1 | **0.8246** | 0.9462 | **0.5966** | 56.6 |
| 2 | 0.9499 | 0.9464 | 0.2974 | 56.8 |
| 3 | 0.9047 | 0.9463 | 0.4155 | 57.2 |
| 4 | 0.9344 | 0.9463 | 0.3401 | 56.5 |
| **sd** | **0.0494** | **0.0001** | **0.1167** | **0.38** |

Spearman varies by 0.0002 across seeds and EF@1% by 0.8, while Pearson swings
0.125 and RMSE doubles. **The induced ranking is essentially deterministic;
only the mapping from embedding to predicted value is unstable.** That is
calibration by any usual definition.

Uni-Mol2 at full budget, by contrast, is stable in every metric (0.9564 at both
additional seeds, to four decimals).

**Consequence, revised.** GROVER's *screening* numbers --- Spearman, recall and
enrichment --- were never in doubt and need no caveat. Its *regression* numbers
do: the single-seed Pearson of 0.9267 in F5 and RMSE of 0.3608 in the fusion
tables are one draw from a distribution spanning 0.125 Pearson, and must carry
the seed range. For virtual screening, where only ordering matters, GROVER has
been stable throughout.

This also explains why GROVER scores higher under ridge than the MLP (F8) and
why the prospective fusion rule fails on its pairs: both depend on error
*magnitudes*, which are exactly what varies.

**MolFormer at 250 rows** is also unstable (sd 0.061, range 0.212–0.366),
though it is rank 6 regardless.

---

## F12 --- Backbone version matters: Uni-Mol2 beats Uni-Mol v1 on d4

`d4` carries both Uni-Mol versions under one oracle and one molecule set, which
isolates version from oracle and chemistry. Cached at 500k rows, evaluated on
200k held-out:

| train rows | Uni-Mol v1 (512-d) | Uni-Mol2 (1536-d) | gap |
|---:|---:|---:|---:|
| 50,000 | 0.5465 | **0.6298** | +0.083 |
| 200,000 | 0.5948 | **0.6700** | +0.075 |
| 500,000 | 0.6256 | **0.6923** | +0.067 |

Uni-Mol2 wins at every budget by roughly 0.07 Pearson, and the gap narrows only
slowly.

**Why it matters.** `al-eval-framework` ranks "Uni-Mol" poorly on DOCKSTRING,
and this project ranks "Uni-Mol2" first on EnamineHTS --- an apparent
contradiction that prompted the cross-repo investigation
([CROSS_REPO_SYNTHESIS.md](CROSS_REPO_SYNTHESIS.md)). Those are **different
models**: v1 at 512-d versus v2. On a pool carrying both, v2 is clearly better.

This removes one candidate explanation for the cross-repo disagreement and
converts the version mismatch from a hypothetical confound into a measured one.
It does not fully resolve the disagreement, since that repo's
MolFormer-over-Uni-Mol ordering is on DOCKSTRING/Vina, a different task from
either pool here.

**A third task regime.** `d4` tops out at Pearson 0.69 at 500k rows, far below
EnamineHTS's 0.95. So `d4` at this budget is substantially harder than either
benchmark studied so far, and is neither the "converges" case (DOCKSTRING) nor
the "wide persistent gap" case (EnamineHTS). Its spread between the two Uni-Mol
versions is 0.07, an order of magnitude smaller than EnamineHTS's 0.30 across
families.

**Caveats.** Single seed. 500k of a 102M-row training split, so these are
low-budget points on a very large pool, not converged results. The two versions
also differ in more than width --- extraction pipeline, conformer generation and
checkpoint --- so this is "v2 as extracted here beats v1 as extracted here",
not a width claim (see F10's scope note).

---

## F13 --- Feature-level fusion: every arm beats the best single model

The first unambiguous multimodality result here. All arms at a matched 200,000
training rows, EnamineHTS, same split and head.

| Strategy | Combination | dims / params | Pearson | RMSE |
|---|---|---:|---:|---:|
| **feature, native concat** | **Uni-Mol2 + MHG-GED** | 1,792 / 2.9M | **0.9533** | **0.2875** |
| feature, PCA-128 per block | Uni-Mol2 + MHG-GED | 256 / 0.8M | 0.9507 | 0.2952 |
| prediction, ridge stacking | 4--5 members | — | 0.9507 | 0.2952 |
| prediction, NNLS weights | GROVER+MHG-GED+Uni-Mol2 | — | 0.9503 | 0.2966 |
| feature, native concat | all six | 6,740 / 7.4M | 0.9499 | 0.2984 |
| prediction, mean | MHG-GED + Uni-Mol2 | — | 0.9495 | 0.2989 |
| feature, PCA-128 per block | all six | 652 / 1.2M | 0.9492 | 0.3007 |
| *best single* | *Uni-Mol2* | *768* | *0.9426* | *0.3180* |

### Three results

**1. Feature-level concatenation of the right pair beats every prediction-level
method** (0.9533 against ridge's 0.9507). Learning jointly from two embeddings
beats combining their predictions.

This runs **opposite to the source paper's central conclusion** that late
fusion consistently outperforms intermediate fusion. Two caveats before
leaning on it: their "intermediate fusion" trains modality-specific
projections jointly, which is a different architecture from concatenation plus
one head (hence the name "feature-level" here, see IV.4 of the progress
report); and this is one budget, one dataset, one seed.

**2. "Two beats six" replicates at the feature level.** The pair beats the full
panel under both variants --- 0.9533 vs 0.9499 (concat), 0.9507 vs 0.9492
(PCA). Late fusion found the same thing (F6) by a different mechanism, so it is
not an artifact of equal weighting: **adding weak modalities hurts even when
the head is free to downweight them.**

**3. Capacity matching does not explain the native-width result.** Native
concatenation lets GROVER contribute 3,400 of 6,740 features, 283x the
descriptor baseline's 12, before any learning --- the obvious objection to
reading anything into it. Equalising every block at 128 components
*slightly hurts* rather than helps (0.9507 vs 0.9533 for the pair). So
feature-count dominance was not driving it. The likeliest explanation is
[F10](#f10): PCA discards low-variance directions, and those are where the
signal lives.

### Caveats

Single seed. One budget (200k) and one dataset. The full-budget comparison is
not run --- native concatenation of all six modalities at 1.9M rows is roughly
51 GB. Descriptors are carried at native width (12) in the PCA arms because 12
components cannot be projected up to 128, so capacity is approximately, not
exactly, matched.

Timing note: `all6-pca` took 1,711 s against `pair-pca`'s 429 s despite being
only 2.5x the fused width --- the PCA fits on 3,400-d GROVER dominate.

---

## F14 --- The fusion gain is the one thing that is *not* budget-conditional

The gap flagged in F13: every fusion number sat at a single budget while this
project's headline concerns budget-dependence. Swept now, for the winning pair
(Uni-Mol2 + MHG-GED) under both feature-level variants.

| budget | best single | feature-concat | feature-PCA-128 | concat gain |
|---:|---:|---:|---:|---:|
| 250 | 0.8783 | 0.8842 | 0.8815 | +0.0059 |
| 1,000 | 0.9142 | 0.9214 | 0.9175 | +0.0071 |
| 5,000 | 0.9250 | 0.9350 | 0.9350 | +0.0100 |
| 50,000 | 0.9440 | 0.9478 | 0.9454 | +0.0037 |
| 200,000 | 0.9426 | **0.9533** | 0.9507 | +0.0107 |
| 1,000,000 | 0.9544 | 0.9592 | 0.9578 | +0.0048 |

### Fusion helps at every budget, by roughly the same amount

Across a 4,000-fold range of training data the gain stays between +0.004 and
+0.011 Pearson, with no trend. That is worth stating precisely because almost
nothing else here behaves that way: unimodal performance moves 0.88 to 0.95
over the same range, and the spread *across* backbones moves 0.57 to 0.30
(F7, F11).

**So fusion is the one intervention measured in this project that is not
budget-conditional.** In a paper whose headline is that representation choice
depends on labelling budget, that is a useful counterpoint rather than a
footnote: *which* representation to use depends on budget, but *whether to
fuse two good ones* does not.

### PCA loses at every budget, including the smallest

A prediction made before the sweep: PCA-128 should win in the low-data regime,
where cutting 1,792 inputs to 256 acts as regularisation. **It does not.**
Native concatenation wins or ties everywhere, including at 250 rows where the
model takes roughly 62 gradient steps in total.

This is a third independent confirmation of [F10](#f10): discarding
low-variance directions costs real signal, and it costs it even when the model
is starved of data and should welcome the regularisation. The first two were
the truncation curve itself and the all-six comparison at 200k.

### Caveats

Single seed throughout. The sweep covers the winning **pair** only --- the
all-six arms remain at 200k, so "two beats six" (F13) is still verified at one
budget. Native concatenation of six modalities is 6,740 dimensions, roughly
51 GB at full budget, and needs per-batch block assembly rather than
materialising the whole matrix before it can be swept.

---

## F15 --- Training-subset resampling: the top and bottom tiers hold, the middle does not

The control Story A was missing. `--subset-seed` redraws *which* training rows
are used while holding val/test and the embedding cache fixed, separating
training-subset variance from the initialisation variance measured in F11.
Five draws per cell, 90 cells, EnamineHTS.

| budget | Uni-Mol2 | MHG-GED | descriptors | **GROVER** | SMI-TED | MolFormer |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.0038 | 0.0032 | 0.0028 | **0.1198** | 0.0298 | 0.0474 |
| 1,000 | 0.0018 | 0.0018 | 0.0020 | **0.0239** | 0.0112 | 0.0082 |
| 5,000 | 0.0007 | 0.0005 | 0.0002 | **0.0164** | 0.0015 | 0.0035 |

(standard deviation of test Pearson across the five draws)

### What holds

- **Uni-Mol2 rank 1 and MHG-GED rank 2 in 5 of 5 draws at every budget**,
  separated by 10--20x their own standard deviation.
- **SMI-TED and MolFormer last in 5 of 5** at every budget. The family gap is
  the most robust result in this project: it survives three prediction heads
  (F8), five initialisation seeds (F11) and five subset draws.
- The spread persists at every budget, so F7's non-convergence claim stands.

### What does not

**GROVER and descriptors swap ranks across draws at every budget.** That is
the one ordering flagged in advance as fragile --- they sit ~0.005 apart while
GROVER's subset sd is 0.12 at 250 rows.

**Consequence for Story A.** "Representation ranking reverses with budget"
must become **"the top and bottom tiers are stable and well separated; the
middle is not resolvable at these budgets."** The narrower claim is the honest
one, and it is still a useful statement: a practitioner can trust which family
to pick and cannot trust a mid-table ordering read off a single run.

### A correction to my own reasoning

Repeatedly in this project I argued that *which* molecules are drawn is "the
dominant source of variability" at small budgets, and that initialisation seeds
could not substitute for it. **That is wrong for five of six backbones.**
Subset sd at 250 rows is 0.003--0.004 for the strong backbones --- the same
order as the initialisation sd of 0.002--0.003 in F11. Neither dominates; both
are small.

It is right for exactly one backbone, GROVER, which is unstable under *both*
resampling axes (subset sd 0.12, initialisation sd 0.089 at the small budgets)
while every other backbone is stable under both.

**Every GROVER figure in this document therefore needs its range quoted**, not
only the full-budget ones flagged in F11. Its *screening* metrics remain the
exception: Spearman and EF were deterministic across initialisation seeds
(F11), and whether that also holds across subset draws has not been checked.

### Caveats

Single pool. Five draws is enough to show GROVER is unstable and the tiers are
not, but not enough for confidence intervals on any individual value. Val/test
are held fixed by design, so this measures training-subset variance only ---
evaluation-set variance is a separate question and is not addressed.

---

## Superseded / provisional

Smoke-test backbone numbers (MHG-GED 0.938, SMI-TED 0.648, MolFormer 0.629)
came from **5 epochs on a 50k test subset**, pre-refactor. MolFormer has since
been superseded by F5; MHG-GED and SMI-TED have not yet been rerun. The descriptor model
had converged; the embedding models were all still improving when stopped. They
are **not** comparable to the tables above (full 105k test, early stopping) and
must not be cited. The running sweep replaces them.

If MolFormer and SMI-TED converge near 0.92, the headline becomes an ordinary
backbone comparison rather than a negative result. Both outcomes are publishable
but they are different papers.

---

## Not yet addressed

- **Multiple seeds** — everything above is a single seed; no significance tests.
- **A second model family** — the strongest objection to a negative result is
  "you mis-tuned the MLP." Ridge and gradient-boosted trees on the same
  embeddings would make the claim architecture-independent. Six cheap runs.
- **Intermediate fusion** — only late fusion implemented.
- **Other pools/oracles** — `ampc` and `d4` share DOCK3.7; the Glide pool is
  blocked on stitching. Nothing here yet separates "true of docking" from
  "true of DOCK3.7."
- **Literature check** — whether the descriptor control is already published in
  the deep-docking / surrogate literature. Novelty is assumed, not verified.
