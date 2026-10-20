# When does molecular representation choice matter for docking-score surrogates?

**Progress report — 2026-10-16**

Repositories: `docking-regression` (built for this work),
`al-eval-framework` (pre-existing, active-learning side)

This report is organised as the investigation actually proceeded: each section
states the question that motivated it, what was done, what was found, and what
that forced next. Dead ends and corrections are included, because three of them
changed a conclusion.

---

## Part I — Origin: what question, and why a new codebase

### I.1 Starting point

The work began from Martínez Galindo et al., *Nat. Commun.* 17:8081 (2026),
*Impact of molecular multimodality on neural network models for prediction
tasks related to drug discovery*. That paper trains 1,467 models across 7
molecular modalities, 2 architectures, 2 fusion strategies and 29 TDC
benchmarks, and concludes that more modalities help on average, that late
fusion beats intermediate fusion, and that knowledge-enhanced pretraining adds
further gains.

The initial question was whether its accompanying repository could reproduce
those experiments, and then whether the approach transfers to docking-score
prediction.

### I.2 Why the existing codebase could not be reused

Audit of the forked IBM repository found it contains the older AAAI-2024
Otter-Knowledge code, not the paper's experiments: 2 of 7 modality encoders, no
knowledge-graph construction, no GNN pretraining, no feed-forward baseline, no
ADMET pipeline, no analysis code. The paper's own Code Availability section
concedes dependence on IBM proprietary infrastructure and states that some
pretrained models were deleted.

**More decisively, the architecture does not fit the target task.** Every
docking pool here pairs one receptor with a molecular library. The protein
input is therefore a constant column, and the knowledge-graph GNN has no
drug–protein edges to exploit. What survives is conceptual rather than code: an
MLP regressing a score from a molecule embedding, and late fusion by averaging
per-modality predictions.

**Decision:** build a standalone pipeline for molecule-only docking-score
regression.

### I.3 Why this task is scientifically tractable

DOCK3.7's score is a physics sum:

$$E_{\text{total}} = E_{\text{vdW}} + E_{\text{elec}} - \Delta G_{\text{desolv}}^{\text{ligand}}$$

van der Waals from AMBER potentials on a precomputed grid, electrostatics from
a Poisson–Boltzmann grid multiplied by atomic partial charges, ligand
desolvation from AMSOL-derived grids; receptor desolvation is treated as
constant and dropped. The reported score is a minimum over sampled poses,
conformers and protomers.

Two consequences shaped the whole study. First, the van der Waals term is a sum
over per-atom contacts, so the score scales with molecular size — a confound
that must be controlled, not ignored. Second, the score is a function of 3D
shape complementarity, which predicts that representations encoding geometry
should transfer to it and string representations should not. Both predictions
were later tested.

---

## Part II — Building a benchmark that can be trusted

### II.1 The problem: silent misalignment

Surrogate benchmarking assumes row *i* of the embedding array corresponds to
row *i* of the score vector. Violating that assumption raises no error. It
produces a model that trains to a plausible loss curve and reports meaningless
numbers.

**What we found on disk.** Three traps, verified rather than assumed:

1. **EnamineHTS embeddings are shuffled relative to the only available score
   file, which is itself sorted by score.** A naive join yields labels that are
   both permuted and systematically ordered. A permutation array recovers the
   mapping.
2. **EnamineHTS Uni-Mol2 is 768-dimensional, not 1,536** as on the other pools
   — a different extraction, so that backbone is not comparable across pools.
3. **The pool has 2,104,318 rows**, not the 2,104,250 recorded in project
   notes, and stores GROVER under a different filename convention.

**Response.** `prepare.py` reconstructs the permutation, applies it, then
checks the SMILES of *every* row against a reference list and refuses to write
on any mismatch. All 2,104,318 rows pass. An independent SMILES-to-score
dictionary check on 50,000 random rows gives zero mismatches. The
pool-registry file encodes the other two quirks so they need not be
remembered.

This is reported as a result in its own right: any benchmark built on these
public artifacts without an equivalent check may be reporting noise.

### II.2 Design decisions, and the reasoning behind each

| Decision | Reason |
|---|---|
| One split shared by every backbone | A comparison between representations must not be confounded by different rows |
| Training indices stored in random order | Makes a budget of *N* the first *N* indices, so learning-curve points are nested and the cache for budget *N* is a prefix of the full cache |
| Failed docks removed at **split** time | Train/val/test then share one universe and the filter is recorded beside the indices rather than applied ad hoc. Affects 2.2% of `d4` (max observed +328), 16 rows of `enhits` |
| Prediction head held fixed | Isolates the representation. Trades per-backbone optimality for comparability; the objection this invites is answered in Part IV.3 |
| Test split read once, predictions saved | Makes fusion and size-control analyses retrospective, with no retraining |
| Both regression and screening metrics | They answer different questions and routinely disagree: Pearson is dominated by the millions of non-binders, enrichment at 0.1% describes the only region a campaign acts on |

### II.3 Pools

| Pool | Target | Oracle | Molecules | Backbones |
|---|---|---|---:|---:|
| `enhits` | EnamineHTS | DOCK3.7 | 2,104,318 | 5 + descriptors |
| `ampc` | AmpC β-lactamase | DOCK3.7 | 98,489,350 | 6 + descriptors |
| `d4` | D4 dopamine receptor | DOCK3.7 | 116,225,927 | 6 + descriptors |
| DOCKSTRING | PARP1, ESR2, F2, EGFR | AutoDock Vina | 50,000 each | 4 + descriptors |

---

### II.4 Which dataset produced which result

Most substantive results come from EnamineHTS. Stated explicitly because it
bounds what each claim covers.

| Result | Dataset | Oracle | Budget |
|---|---|---|---|
| Family split (IV.1), size control (III.3), head independence (IV.2) | EnamineHTS | DOCK3.7 | 1.89M |
| Fusion suite and prospective rule (IV.3–IV.5) | EnamineHTS | DOCK3.7 | 1.89M |
| Budget sweep, 250 → 1.89M | EnamineHTS | DOCK3.7 | 7 points |
| Budget sweep, 250 → 45k | DOCKSTRING PARP1 | Vina | 4 points |
| Size dependence by target (V.5) | DOCKSTRING PARP1/F2/ESR2 | Vina | — |
| Uni-Mol v1 vs v2 | `d4` | DOCK3.7 | in progress |
| — | `ampc` | DOCK3.7 | prepared, not run |

### II.5 Provenance of the EnamineHTS embeddings — a scope limit

Work in the companion repository `al-molecular` established that **the
EnamineHTS embeddings were not produced by either project's pipeline.** They
were inherited from a colleague's directory and extracted with different
settings. Two consequences are documented there and confirmed here:

- Its Uni-Mol2 is **768-dimensional** where this project's own extractor
  produces 1536-d, and it has effective rank 2.2 of 768 (82% of variance in a
  single principal component).
- The older EnamineHTS embedding set carries GROVER at 1600-d where the
  current one carries 3400-d.

This matters beyond bookkeeping. The 768-d Uni-Mol2 is the **best single
model** in every result below and appears in every winning fusion combination,
and GROVER is the member on which the prospective fusion rule fails (IV.5).
Every EnamineHTS claim in this report is therefore a statement about *these
particular embedding files*; a re-extraction would move the fusion results as
well as the unimodal ones.

A matched 1536-d Uni-Mol2 for this pool **already exists** (2,141,500 × 1536).
The join was verified in detail rather than asserted: every one of the
2,104,318 scored molecules matches exactly one embedding row; there are **no
duplicate SMILES on either side**, so no mapping is ambiguous; and the 37,182
additional rows in the larger file are simply unused. The swap can therefore be
screened with a single training run on a cached embedding, through the same
split and head as every other backbone, before any larger commitment.

---

## Part III — First results, and the hypothesis they generated

### III.1 An anomaly worth chasing

The first smoke test (200k train, 5 epochs) gave MHG-GED Pearson 0.938 against
MolFormer 0.629. A gap that large is a leakage smell, so it was checked:
individual raw MHG-GED dimensions correlate up to 0.65 with the score.

That led to the real explanation rather than leakage. **Heavy-atom count alone
correlates −0.659 with the DOCK3.7 score.** MolFormer and SMI-TED were
performing no better than counting atoms.

### III.2 Consequence: a baseline the field does not use

If a representation can be matched by counting atoms, every comparison needs a
trivial control. We added twelve RDKit descriptors — heavy atoms, molecular
weight, logP, TPSA, rings, rotatable bonds, H-bond donors and acceptors,
fraction sp³, halogens, aromatic rings, heteroatoms — as a pseudo-backbone
running through the identical split, head and metrics.

It is demanding and it saturates early:

| Train rows | Pearson | Spearman | RMSE | EF@1% |
|---:|---:|---:|---:|---:|
| 50,000 | 0.8748 | 0.8613 | 0.4611 | 37.5 |
| 200,000 | 0.8802 | 0.8669 | 0.4517 | 38.4 |
| 1,000,000 | 0.8872 | 0.8744 | 0.4391 | 38.7 |
| 1,893,872 | 0.8907 | 0.8783 | 0.4326 | 40.7 |

38× more data buys +0.016 Pearson — what a 12-feature model should do, running
out of capacity long before it runs out of data.

### III.3 Testing the obvious objection — and being wrong

**Hypothesis:** the descriptor baseline only looks good because it encodes
size.

Three controls were built, all computed from saved predictions: partial
correlation with heavy-atom count removed; residual correlation after a
degree-3 size fit; and metrics within ten equal-count heavy-atom strata. A
degenerate size-only predictor serves as the reference.

| Model | raw *r* | partial *r* | within-stratum *r* | within-stratum EF@1% |
|---|---:|---:|---:|---:|
| size only (−heavy atoms) | 0.6592 | 0.000 | 0.1478 | 2.6 |
| descriptors, full budget | 0.8907 | 0.7966 | 0.8000 | 30.1 |

**The hypothesis was wrong.** The control behaves as intended — size alone
collapses within a stratum, to enrichment 2.6 against a random baseline of 1.0,
exactly as a near-constant-within-bin feature must. But descriptors lose only
0.09 Pearson and retain enrichment of 30.1. Roughly 26% of their raw enrichment
is size-driven; the remainder reflects **non-size molecular information carried
by the other eleven descriptors**. Attributing it to specific features (logP,
TPSA, ring counts) would require per-feature ablation or conditional importance
analysis, which has not been run.

This mattered for everything downstream: a baseline that was merely a size
proxy would be a weak comparator, whereas one retaining within-stratum
enrichment of 30 makes failure to beat it genuinely informative.

---

## Part IV — Establishing the main result on one pool

### IV.1 A correction: one backbone is not a trend

An interim finding stated "embeddings lose to descriptors," generalising from
MolFormer alone. With all backbones converged the picture is **bimodal**, and
the interim claim was withdrawn.

Full budget, EnamineHTS, with size controls:

| Backbone | Type | Dim | raw *r* | partial *r* | within-stratum *r* | within EF@1% |
|---|---|---:|---:|---:|---:|---:|
| Uni-Mol2 | 3D conformer | 768 | **0.9564** | **0.9215** | **0.9233** | 53.3 |
| MHG-GED | hypergraph grammar | 1024 | 0.9543 | 0.9177 | 0.9196 | **53.9** |
| GROVER | graph transformer | 3400 | 0.9267 | 0.8681 | 0.8867 | 50.3 |
| descriptors | — | 12 | 0.8907 | 0.7966 | 0.8000 | 30.1 |
| SMI-TED | SMILES LM | 768 | 0.6892 | 0.5205 | 0.5351 | 15.6 |
| MolFormer | SMILES LM | 768 | 0.6611 | 0.4930 | 0.5063 | 10.4 |

**The split follows representation family, not dimensionality.** All three
768-dimensional models appear — one at the top (Uni-Mol2) and two at the bottom
(SMI-TED, MolFormer). The high performers encode 3D geometry or molecular
graphs; the low performers are both SMILES language models. This is consistent
with the prediction from I.3, since DOCK3.7 evaluates a posed 3D conformer.

Note the scope: SMILES strings do encode connectivity, so this is not evidence
that string representations are inherently unsuitable. What is shown is that
**these two frozen SMILES-language embeddings expose substantially less
DOCK3.7-relevant signal on this pool** than the geometry- and graph-based
embeddings do.

**The split survives size control and widens under it.** The SMILES models lose
most (−0.17) and the structure models least (−0.035): the weak models were
leaning on size more heavily.

*Lesson recorded:* one data point drawn from a bimodal distribution is worse
than none, because it reads as a trend.

### IV.2 Does the prediction head explain it?

Holding the head fixed invites the objection that a 1024-512 MLP may suit
768-dimensional geometry embeddings and handicap 12 descriptors. Two unrelated
families were refit on identical cached embeddings and splits.

| Backbone | MLP (1.9M) | Ridge (1.9M) | LightGBM (200k) |
|---|---:|---:|---:|
| Uni-Mol2 | 0.9564 | 0.9368 | 0.9356 |
| MHG-GED | 0.9543 | 0.9198 | 0.9421 |
| GROVER | 0.9267 | 0.9339 | 0.9350 |
| descriptors | 0.8907 | 0.8455 | 0.8769 |
| SMI-TED | 0.6892 | 0.6336 | 0.6517 |
| MolFormer | 0.6611 | 0.5987 | 0.6096 |

All three heads reproduce the same broad separation between the
high-performing structure-aware representations, the descriptor baseline, and
the two SMILES-language embeddings. **The objection is answered at the level of
tiers: the family gap is a property of the representations, not of the head.**

The exact ordering *within* the top tier is not preserved — ridge places GROVER
above MHG-GED, LightGBM places MHG-GED above Uni-Mol2 — so no claim is made
about rank order among the three structure-aware models, whose full-budget
Pearson values span only 0.03.

Two incidental observations. Uni-Mol2's signal is largely *linearly accessible*:
ridge alone reaches 0.9368, so the MLP adds only +0.02. And GROVER scores higher
under ridge than under the MLP, while showing Pearson 0.9267 against Spearman
0.9464 — a pattern consistent with a monotone but non-linear distortion of its
predicted values, or with outliers. It could equally reflect optimisation,
regularisation or model-selection differences; the cause has not been
diagnosed.

### IV.3 When does combining representations help?

The source paper's central claim is that more modalities help on average under
late fusion. Testing it here on converged models:

| Combination | Pearson | RMSE | EF@1% |
|---|---:|---:|---:|
| **MHG-GED + Uni-Mol2** | **0.9605** | **0.2650** | **61.6** |
| Uni-Mol2 alone | 0.9564 | 0.2779 | 58.6 |
| + GROVER (3-way) | 0.9567 | 0.2770 | 60.7 |
| + descriptors (4-way) | 0.9524 | 0.2915 | 58.4 |
| descriptors + Uni-Mol2 | 0.9415 | 0.3228 | 53.2 |

The best subset is **2 of 6**; every further addition degrades it. For an
equal-weight average of two models to beat the better one,

$$\sigma_2/\sigma_1 < -\rho + \sqrt{\rho^2+3}$$

| Model | error σ | ρ with Uni-Mol2 | σ ratio | threshold | helps? |
|---|---:|---:|---:|---:|:--:|
| MHG-GED | 0.284 | 0.777 | 1.02 | 1.12 | yes |
| GROVER | 0.361 | 0.656 | 1.30 | 1.20 | no |
| descriptors | 0.433 | 0.634 | 1.56 | 1.21 | no |
| MolFormer | 0.715 | 0.379 | 2.57 | 1.39 | no |

The criterion predicts every outcome, and the implied RMSE for the winning pair
(0.265) matches the measured 0.2650.

*Derivation scope.* The criterion as written assumes zero-mean errors, so that
the cross term is \(\rho\sigma_1\sigma_2\). This was verified rather than
assumed: mean errors contribute at most 0.18% of MSE for any model here
(largest, MolFormer, \(\mu = 0.030\) against RMSE 0.715), and the centered and
uncentered cross terms for the winning pair differ by 0.01%. Recomputing
exactly with \(E[e_1 e_2]\) gives the same 0.2650. For biased surrogates the
uncentered form should be used.

**The informative part:** MolFormer's errors are the *least* correlated with
Uni-Mol2's (ρ = 0.379), so it carries genuinely complementary information. It is
simply too inaccurate to pay for the variance it adds. Equal-weight averaging
cannot exploit complementarity it must pay for in error variance — which
reframes "more modalities help" as a claim about member-quality homogeneity
rather than modality count.

### IV.4 A full fusion suite, fitted honestly on validation

The result above concerns equal-weight averaging only, which is one of several
fusion strategies and not the one the multimodality literature emphasises. Two
further families were implemented. **All weights, scalings and subset choices
are estimated from validation predictions; the test split is read once, to
score.**

**Prediction-level methods** (`dockreg/latefuse.py`): unweighted mean,
validation-standardised z-mean, rank mean, non-negative least-squares weights,
ridge stacking, and best-single-chosen-on-validation. A retrospectively best
subset is reported separately as an explicit upper bound.

**Feature-level methods** (`dockreg/featurefuse.py`): per-modality
standardisation then concatenation at native width, and a dimension-controlled
variant projecting each modality to 128 components by PCA fitted on training
rows. The second exists because native-width concatenation lets a 3400-d
backbone contribute 283× more features than the 12-d descriptor baseline before
any learning occurs. Capacity is only *approximately* matched even then, since
12 descriptors cannot be projected up to 128. These are called "feature-level",
not "intermediate", because the literature's intermediate fusion trains
modality-specific projections jointly, which this does not.

EnamineHTS, full budget:

| Method | Fitted on | test RMSE | Pearson | EF@1% |
|---|---|---:|---:|---:|
| **ridge stacking** | validation | **0.2638** | **0.9608** | 61.5 |
| NNLS weights | validation | 0.2648 | 0.9606 | 61.6 |
| mean, best pair | — | 0.2650 | 0.9605 | 61.6 |
| *oracle mean-subset* | *test (upper bound)* | *0.2650* | *0.9605* | *61.6* |
| best single (Uni-Mol2) | validation | 0.2779 | 0.9564 | 58.6 |

**Ridge fitted on validation beats every equal-weight subset, including the
retrospectively test-optimal one.** The narrow claim is that validation-fitted
linear weighting outperforms equal weighting over any subset here; it does
*not* follow that subset selection is irrelevant, since optimised weighting
over a selected subset might do better still. That has not been tested.

*Not yet reported, and needed before this is quotable:* the ridge coefficients
and their signs, the regularisation strength and how it was chosen (currently a
fixed λ = 10⁻³·n, not selected), and performance with MolFormer or SMI-TED
removed. Because the same validation split fits the coefficients and would also
select λ, that selection must move to cross-validation *within* the validation
partition. Small non-zero coefficients on the weak members are not by
themselves evidence that those members contribute. Ridge's best subsets also *include* MolFormer and SMI-TED, the two weak
members that mean fusion had to exclude: weighting can exploit
complementary-but-inaccurate members, averaging cannot.

*Reporting caveat.* Z-mean and rank fusion change the output scale, so their
RMSE values (7.87 and ~60,000 respectively) are meaningless. Only
correlation and enrichment are comparable across these methods.

### IV.5 The fusion criterion tested prospectively — it does not hold up

Section IV.3 computed σ and ρ on test predictions and then explained test
outcomes, which is circular. The prospective form estimates both from
**validation** predictions and predicts which pairs help on test.

An earlier draft of this section reported "80% accuracy (12/15)" and called the
criterion usable prospectively. **That framing was wrong, because the base rate
is extreme.**

| | value |
|---|---:|
| pairs evaluated | 15 |
| pairs that actually help | **2** |
| rule accuracy | 0.800 |
| **accuracy of always predicting "no"** | **0.867** |
| balanced accuracy | 0.673 |
| Matthews correlation | 0.294 |

Confusion matrix: TP 1, FP 2, FN 1, TN 11.

**The rule is less accurate than a trivial baseline that never recommends
fusion.** With only 2 of 15 pairs genuinely helping, raw accuracy is dominated
by correctly saying "no", and the criterion gets one of the two useful pairs
while producing two false positives.

The two false positives sit almost exactly on the decision boundary —
GROVER+MHG-GED at ratio 1.04 against threshold 1.06 (margin +0.022), and
GROVER+Uni-Mol2 at 1.07 against 1.10 (margin +0.039). So the criterion is not
wildly miscalibrated; it cannot resolve cases this close with validation
statistics from a single split.

**What can honestly be claimed.** The algebraic identity in IV.3 is exact and
its assumptions were verified. Its *prospective* value as a selection rule is
**not demonstrated** on this evidence: 15 pairs with 2 positives cannot
separate a weak rule from a useless one, and the point estimate is below the
trivial baseline. An earlier observation that "all three failures involve
GROVER" is consistent with that member's other anomalies but is equally
consistent with sampling variability at n = 15.

**What would settle it:** many more pairs (the 58-target DOCKSTRING panel
supplies hundreds), paired bootstrap intervals on the predicted-versus-observed
RMSE change rather than a binary label, and seed replication so the boundary
cases are not decided by one split.

---|---|
| Not involving GROVER | **10 / 10** |
| Involving GROVER | **2 / 5** |
| Overall | 12 / 15 (80%) |

**All three failures involve GROVER**, the one member with an independently
diagnosed pathology (Pearson 0.9267 against Spearman 0.9464; higher under
ridge than under the MLP, IV.2). Its validation error statistics do not
transfer to test.

This is a more useful result than the retrospective 100%: the criterion is
usable prospectively, its failure mode is identified, and that failure is tied
to a defect diagnosed by a separate line of evidence rather than to chance.

### IV.4b Feature-level fusion, run

All four arms at a matched 200,000 training rows, with prediction-level methods
recomputed at the same budget for a like-for-like comparison:

| Strategy | Combination | dims / params | Pearson | RMSE |
|---|---|---:|---:|---:|
| **feature, native concat** | **Uni-Mol2 + MHG-GED** | 1,792 / 2.9M | **0.9533** | **0.2875** |
| feature, PCA-128 per block | Uni-Mol2 + MHG-GED | 256 / 0.8M | 0.9507 | 0.2952 |
| prediction, ridge stacking | 4–5 members | — | 0.9507 | 0.2952 |
| prediction, NNLS | GROVER+MHG-GED+Uni-Mol2 | — | 0.9503 | 0.2966 |
| feature, native concat | all six | 6,740 / 7.4M | 0.9499 | 0.2984 |
| prediction, mean | MHG-GED + Uni-Mol2 | — | 0.9495 | 0.2989 |
| feature, PCA-128 per block | all six | 652 / 1.2M | 0.9492 | 0.3007 |
| *best single* | *Uni-Mol2* | *768* | *0.9426* | *0.3180* |

**Every fusion arm beats the best single model** — the first unambiguous
multimodality result in this project.

**Feature-level concatenation of the right pair beats every prediction-level
method.** This runs opposite to the source paper's conclusion that late fusion
consistently outperforms intermediate fusion. Two caveats before leaning on it:
their intermediate fusion trains modality-specific projections jointly, a
different architecture from concatenation plus one head; and this is one
budget, one dataset, one seed.

**"Two beats six" replicates at the feature level** (0.9533 vs 0.9499 concat;
0.9507 vs 0.9492 PCA). Late fusion found the same thing by a different
mechanism, so it is not an artifact of equal weighting: adding weak modalities
hurts even when the head is free to downweight them.

**Capacity matching does not explain the native-width result.** Native
concatenation lets GROVER contribute 3,400 of 6,740 features before any
learning, which is the obvious objection. Equalising every block at 128
components *slightly hurts* rather than helps, so feature-count dominance was
not driving it. The likeliest explanation is IV.6: PCA discards low-variance
directions, which is where the signal lives.

### IV.4c Fusion across budgets

The gap above, closed for the winning pair (Uni-Mol2 + MHG-GED):

| budget | best single | feature-concat | feature-PCA-128 | concat gain |
|---:|---:|---:|---:|---:|
| 250 | 0.8783 | 0.8842 | 0.8815 | +0.0059 |
| 1,000 | 0.9142 | 0.9214 | 0.9175 | +0.0071 |
| 5,000 | 0.9250 | 0.9350 | 0.9350 | +0.0100 |
| 50,000 | 0.9440 | 0.9478 | 0.9454 | +0.0037 |
| 200,000 | 0.9426 | **0.9533** | 0.9507 | +0.0107 |
| 1,000,000 | 0.9544 | 0.9592 | 0.9578 | +0.0048 |

**Fusion helps at every budget, by roughly the same amount.** Across a
4,000-fold range the gain stays between +0.004 and +0.011 Pearson with no
trend — while unimodal performance moves 0.88 → 0.95 and the spread across
backbones moves 0.57 → 0.30 over the same range.

**Fusion is therefore the one intervention measured here that is not
budget-conditional.** In a report whose central claim is that representation
choice depends on labelling budget, this is a direct counterpoint worth
stating: *which* representation to use depends on budget; *whether to fuse two
good ones* does not.

**PCA loses at every budget, including the smallest.** The prediction before
running this was that PCA-128 would win in the low-data regime, where cutting
1,792 inputs to 256 acts as regularisation. It does not — native concatenation
wins or ties everywhere, including at 250 rows. That is a third independent
confirmation of IV.6: discarding low-variance directions costs real signal,
even when the model is starved of data.

**Remaining gap:** the sweep covers the pair only. The all-six arms are still
at 200k, so "two beats six" is verified at one budget. Native concatenation of
six modalities is 6,740 dimensions (~51 GB at full budget) and needs per-batch
block assembly before it can be swept.

---

### IV.6 Resolving the embedding-provenance risk, and what it revealed

II.5 flagged that every EnamineHTS result rests on a third-party embedding set
whose Uni-Mol2 is 768-dimensional with effective rank 2.2 — and which is the
best single model in every result here. That risk has now been retired, and
investigating it produced a result of its own.

**The matched 1536-d extraction loses.** Cached through the same split and
trained with the same head:

| | dim | Pearson | partial *r* | RMSE | EF@1% |
|---|---:|---:|---:|---:|---:|
| **Uni-Mol2 768 (inherited)** | 768 | **0.9564** | **0.9215** | **0.2779** | **58.6** |
| Uni-Mol2 1536 (this pipeline) | 1536 | 0.9496 | 0.9091 | 0.2983 | 56.2 |

So the anomalous file is genuinely the better representation, not an accident
that happened to score well. The EnamineHTS results are not an artifact of a
broken embedding. (Single seed; the Pearson gap is 0.0068, but RMSE and EF
move with it.)

This also settles a question posed in the sibling AL repository, which had
proposed swapping in the 1536-d version: on this evidence the swap would cost
performance, and the screen cost one training run rather than an AL campaign.

**No leakage.** Because the source pool was score-sorted before shuffling, PC1
of each embedding was checked against row index: 0.0007 for Uni-Mol2 768, at
most 0.0043 for any backbone.

**The dominant direction is molecular size.** PC1 of the 768-d embedding
correlates **0.802 with heavy-atom count** and only **0.228 with the docking
score**. It is a large nuisance direction, not a signal direction.

**Variance concentration is anti-correlated with information.** PCA truncation
on that embedding alone, varying only *k* — one embedding, so no provenance
confound — at 200,000 training rows:

| top-*k* PCs | cumulative variance | Pearson | EF@1% |
|---:|---:|---:|---:|
| 1 | 82% | 0.284 | 1.9 |
| 2 | 90% | 0.328 | 4.5 |
| 8 | — | 0.813 | 27.3 |
| 16 | — | 0.912 | 42.9 |
| 32 | — | 0.929 | 45.7 |
| 128 | — | 0.945 | 51.9 |
| all 768 | 100% | 0.948 | 53.4 |

The two components carrying 90% of the variance deliver about a third of the
achievable correlation. The signal lives in the **low-variance tail**, and
per-feature standardisation is what makes it usable by rescaling those
directions.

Two consequences:

1. Spectral diagnostics — effective rank, participation ratio,
   explained-variance thresholds — do not identify informative embeddings here;
   they are *anti*-correlated with usefulness.
2. **PCA-reducing embeddings on a variance criterion is unsafe.** Retaining 90%
   of variance costs two thirds of the performance; retaining 128 components
   (>99%) costs 0.003 Pearson. The variance threshold gives no warning which
   regime applies, and this reduction is a common storage-saving step at pool
   scale.

**Scope.** This is a claim about variance-based diagnostics, not about
dimensionality. An earlier draft framed it as "width is not quality" from the
768-versus-1536 comparison; that comparison confounds width with provenance,
pooling, checkpoint and conformer generation, and cannot support the broader
claim. The truncation experiment varies only *k* and is clean.

---

### IV.7 Seed replication, and an unstable backbone

Five initialisation seeds per cell at the three small EnamineHTS budgets; 72
runs, zero failures.

**Scope first:** `--seed` varies model initialisation and batch shuffling. It
does **not** resample which rows enter the training subset. At 250 rows the
subset draw is the larger source of variability, and measuring it requires
new splits and rebuilt caches. These are optimisation seeds.

**Rankings are stable.** Uni-Mol2 is rank 1 in 5 of 5 seeds at every budget and
MHG-GED rank 2 in 5 of 5. The only rank changes anywhere involve GROVER.

**The spread persists:** 0.568 ± 0.060 at 250 rows, 0.452 ± 0.009 at 1,000,
0.391 ± 0.042 at 5,000. Seed variability is far smaller than the spread, so
**V.3's non-convergence claim survives replication on this axis.**

**GROVER's training is unstable**, which is new and consequential:

| budget | GROVER sd | range across seeds | Uni-Mol2 sd for comparison |
|---:|---:|---|---:|
| 250 | 0.0885 | 0.622 – 0.823 | 0.0020 |
| 1,000 | 0.0502 | 0.759 – 0.882 | 0.0004 |
| 5,000 | **0.1567** | **0.459 – 0.824** | 0.0008 |

At 5,000 rows GROVER's result depends on initialisation to the tune of 0.37
Pearson. This converges with three earlier anomalies for the same backbone —
its Pearson/Spearman gap (IV.1), scoring higher under ridge than under the MLP
where every other backbone loses (IV.2), and all three failures of the
prospective fusion rule (IV.5). The consistent reading is **optimisation
instability rather than a weak representation**: ridge, which has no
optimisation to fail, ranks it third.

**Consequence:** every single-seed GROVER figure in this report, including
0.9267 at full budget, is one draw from a wide distribution and is not
quotable until seeded. The earlier description of its Pearson/Spearman gap as
a "calibration problem" is superseded by this.

---

## Part V — Testing whether the result generalises

### V.1 Literature review: the novelty target moved

A search established that two of the three things this work appeared to
contribute were already published.

- **Descriptors beating learned representations** is established in general
  molecular property prediction: Deng et al. (*Nat. Commun.* 2023) find RDKit2D
  descriptors frequently beat learned representations; Praski et al.
  (arXiv:2508.06199) find nearly all of 25 pretrained models fail to improve on
  ECFP across 25 datasets.
- **Trivial baselines rivalling docking itself** is published: a 2026 JCAMD
  evaluation finds most docking programs do not outperform a molecular-weight
  regression, framing MW regression explicitly as a null model.
- **Size bias in docking scores is textbook** — ligand efficiency has
  normalised by heavy-atom count for decades.

Not found: a descriptor null baseline in the ultra-large docking surrogate
literature; a size-decomposition protocol applied to surrogate models rather
than labels; and any budget-resolved comparison showing rank reversal on
docking tasks. These searches were keyword-driven with no full texts read, so
absence of evidence is weak.

Deng et al. additionally warn that concatenating descriptors onto GROVER
inflates its apparent advantage — directly relevant to any fusion arm.

### V.2 An external dataset, and a contradiction

DOCKSTRING (50,000 molecules × 58 targets, AutoDock Vina) was identified as the
natural external benchmark: a second oracle, published baselines, and target
breadth. It was then found already prepared in `al-eval-framework`, together
with an AL loop, C1–C4 diagnostics and a paper draft.

**The two projects disagreed about which representations are good.** MolFormer
ranked near-best and Uni-Mol poor on DOCKSTRING; the reverse on EnamineHTS.

**Test:** run this project's protocol on their data, holding everything but the
dataset fixed. Result: it **reproduced their ordering** (MolFormer > Uni-Mol
v1). Neither pipeline is wrong; the datasets genuinely differ. Protocol is
eliminated.

### V.3 Two corrections, and the real contrast

**First conclusion (wrong):** representation choice is irrelevant on DOCKSTRING
— spread 0.019 against EnamineHTS's 0.349 — and `al-eval-framework` therefore
tested its hypothesis on a dataset unable to falsify it.

**What was missed:** their `static_eval` uses `train_frac=0.005` — **250
molecules**, not 45,000. Sweeping the budget:

| Train rows | descriptors | MolFormer | GROVER | Uni-Mol v1 | Morgan | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | **0.8447** | 0.8037 | 0.6133 | 0.7471 | 0.5475 | **0.297** |
| 1,000 | **0.8711** | 0.8413 | 0.8087 | 0.7997 | 0.7002 | 0.171 |
| 5,000 | **0.8771** | 0.8701 | 0.8663 | 0.8479 | 0.8134 | 0.064 |
| 45,000 | 0.8905 | 0.8957 | **0.9095** | 0.8905 | 0.8995 | 0.019 |

At their budget the spread is 0.297. The criticism was withdrawn in both
repositories' documentation.

Two things follow. **The ordering reverses with budget** — descriptors run 1st
at 250 rows and last at 45,000; Morgan runs last and then 2nd. And the
comparison had to be repeated symmetrically on EnamineHTS:

| Train rows | descriptors | Uni-Mol2 | MHG-GED | GROVER | SMI-TED | MolFormer | spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.8293 | **0.8783** | 0.8532 | 0.8234 | 0.4709 | 0.3660 | **0.512** |
| 1,000 | 0.8523 | **0.9142** | 0.8926 | 0.8822 | 0.5490 | 0.4518 | 0.462 |
| 5,000 | 0.8615 | **0.9250** | 0.9173 | 0.8082 | 0.6035 | 0.5510 | 0.374 |
| 50,000 | 0.8748 | **0.9440** | 0.9316 | 0.8761 | 0.6376 | 0.5948 | 0.349 |
| 200,000 | 0.8802 | **0.9426** | 0.9402 | 0.9376 | 0.6609 | 0.6265 | 0.316 |
| 1,000,000 | 0.8872 | **0.9544** | 0.9516 | 0.9430 | 0.6827 | 0.6502 | 0.304 |
| 1,893,872 | 0.8907 | **0.9564** | 0.9543 | 0.9267 | 0.6892 | 0.6611 | 0.295 |

**This is the central result — with one important qualification.**

| Benchmark | spread @ 250 | spread @ ~45–50k | spread @ max |
|---|---:|---:|---:|
| DOCKSTRING PARP1 / Vina | 0.297 | 0.019 | 0.019 (45k) |
| EnamineHTS / DOCK3.7 | 0.512 | 0.349 | 0.295 (1.9M) |

Both panels begin representation-sensitive; only one collapses. This already
rules out the simpler explanation that DOCKSTRING merely lacked spread at the
AL-relevant budget.

**Qualification: the two panels are not the same representations.** DOCKSTRING
lacks MHG-GED, SMI-TED and Uni-Mol2; EnamineHTS lacks Morgan and Uni-Mol v1;
and GROVER differs in dimensionality (1600 vs 3400). Panel-wide spread
therefore partly reflects *which* representations each panel contains, and
"the two benchmarks differ in convergence" is premature as a panel-level claim.

Among the **exactly shared** representations — descriptors and MolFormer — the
contrast is nonetheless stark:

| Benchmark | descriptors − MolFormer gap @ ~50k | @ max budget |
|---|---:|---:|
| DOCKSTRING PARP1 / Vina | 0.005 (45k) | 0.005 |
| EnamineHTS / DOCK3.7 | 0.280 (50k) | 0.230 (1.9M) |

Two matched representations become nearly indistinguishable on PARP1 and remain
widely separated on EnamineHTS. That is a genuine task interaction on matched
evidence, but two representations do not establish panel-wide convergence.
Extending DOCKSTRING with the missing backbones is the decisive next
experiment.

The reframed question this suggests is sharper than the original:

> **Why does additional supervision erase representation differences on some
> docking-score landscapes but not others?**

**The evaluated representations converge to a narrow common performance band on
PARP1, whereas on EnamineHTS only the structure-aware representations reach the
higher observed performance.** We avoid calling 0.90 a "ceiling": it is where
*these* representations converge, and a better model may exceed it. Nothing here
establishes an intrinsic limit on either task.

*Lesson recorded:* compare at the regime the other experiment actually used.
Both errors came from assuming a default instead of reading four lines of the
other project's code.

### V.4 A result from the sibling repository

`al-eval-framework`'s design document predicts that static performance will
**not** predict AL sample efficiency. Measured Spearman between static-RMSE
rank and AL-AUDC rank:

| Subset | pooled | EGFR | ESR2 | F2 |
|---|---:|---:|---:|---:|
| all 6 representations | 0.886 | 1.000 | 0.943 | 0.943 |
| 4 independent (drop derived fusion variants) | 0.800 | 1.000 | 0.800 | 0.800 |
| 3 learned only | 1.000 | 1.000 | 0.500 | 1.000 |

The ranks **agree**, robustly to removing the derived fusion variants (which
could have anchored both rankings) and to removing Morgan (a clear worst on
both axes). **Power is low:** with four independent representations ρ = 0.8
gives p ≈ 0.33. The defensible statement is that the predicted decoupling did
not appear — not that static reliably predicts AL.

### V.5 Side result: size dependence varies sharply by target

| Target | heavy-atom vs score | variance explained |
|---|---:|---:|
| EnamineHTS / DOCK3.7 | −0.659 | 43.5% |
| PARP1 / Vina | −0.648 | 42.0% |
| F2 / Vina | −0.623 | 38.8% |
| **ESR2 / Vina** | **−0.199** | **4.0%** |

PARP1 and EnamineHTS have nearly identical *marginal* size dependence yet
opposite convergence behaviour, which argues against simple label-level size
dependence explaining V.3. It does not eliminate differences in size
*distribution*, non-linear size effects, or representation-by-size
interactions. ESR2 is nearly size-free, making it the cleanest target for isolating
chemical signal — and suggesting size-explained variance as a landscape
descriptor for predicting how much representation choice matters.

---

## Part VI — Engineering defects that affected results

Recorded because two distorted measurements and one corrupted a reported
number.

| Defect | Effect | Resolution |
|---|---|---|
| `np.asarray()` on a memmap returns a **view**, not a copy | "Loading into RAM" did not; every epoch page-faulted against the filesystem. One run sat 2.5 hours without completing an epoch | `np.array()`; load 12s, epochs 15–22s |
| Same bug, written again into the streaming gather and the slab path | Scattered page reads inside each window — the pattern those functions exist to avoid. ~23 MB/s | Fixed at both sites with explanatory comments; ~25× throughput |
| Cache builder wrote scattered pages to disk | One streaming write became millions of scattered page writes; a 5.8 GB cache took 4.7 hours | Build in RAM, write once sequentially |
| Resume logic skipped on "`metrics.json` exists" | A 3-epoch debug run silently entered the results tables as the 50k descriptor baseline (0.8628; true value 0.8748) | Resume now compares recorded epochs and patience |
| `featurefuse.py` cached under the shared tag with only `max_train` rows | Truncated the full-split caches for MolFormer and SMI-TED to 200k, breaking 13 downstream runs | Caches the same rows `train.py` would, then slices |
| Cache writes were not atomic | 16 parallel workers rebuilding one 25.8 GB cache left a corrupt 9.8 GB file that could not be opened | Write to a pid-unique temp file, then `os.replace`; added `scripts/check_caches.py` |
| Slurm preamble sourced via `$(dirname "$0")` | First submission died instantly: sbatch copies the script to the spool dir while starting it in `SLURM_SUBMIT_DIR` | Repo-relative path, matching al-molecular's convention |

The first three made the pipeline *slow*, not *wrong*, which is why none
surfaced in testing — a timing anomaly was the only signal. The later three
were caught by downstream failures rather than by any check, which is why a
cache auditor now exists.

**A reporting failure mode, recorded because it recurred.** Several times
during this work a job was described as "running" or "N of M complete" on the
basis of run *directories* rather than `metrics.json` files. A killed run
leaves its directory behind, so the count overstates progress. Count outputs,
not folders.

---

## Part VII — What is and is not established

### Robust (not dependent on a single training subset)

1. **Row-alignment verification** — all 2,104,318 rows checked, plus an
   independent 50,000-row cross-check (II.1).
2. **The descriptor baseline is substantive after size control** — three
   independent controls, computed on held-out predictions, with a degenerate
   size-only reference that behaves as expected (III.3).
3. **Full-budget family separation survives three prediction heads** — the gap
   between structure-aware embeddings, descriptors, and SMILES-language
   embeddings reproduces under linear, tree and neural probes (IV.2). Rank
   order *within* the top tier is not stable and no claim is made about it.
4. **The fusion criterion** is an algebraic identity whose assumptions were
   verified numerically on this data (IV.3).

### Observed, single-seed, awaiting replication

Every learning curve uses **one training subset per budget**. These are
observations, not established results:

5. Budget-dependent **rank reversals** (V.3) — most fragile at 250 rows, where
   the choice of training subset matters most.
6. The precise **spread values** and convergence trajectories in V.3.
7. The **cross-benchmark contrast**, which additionally rests on unmatched
   representation panels (see V.3 qualification).

**Replication must resample the training subset, not merely bootstrap the test
set.** Test-set bootstrap would quantify evaluation noise while leaving the
dominant source of variability at small budgets — which 250 molecules were
drawn — entirely unmeasured.

### Not established

- **Seeds.** *Substantially addressed.* Five initialisation seeds (IV.7) and
  five training-subset draws (F15 in FINDINGS.md) now cover EnamineHTS at
  250/1,000/5,000 rows. The top and bottom tiers are stable under both; GROVER
  and descriptors swap ranks across subset draws at every budget, so the
  middle of the ranking is not resolvable. The earlier claim here that
  training-subset draw is "the dominant variance source" was **wrong** for
  five of six backbones --- both variance sources are ~0.003 and comparable.
  Still single-seed: all full-budget results, all DOCKSTRING results, and the
  entire fusion analysis.
- **Oracle versus chemistry.** The two benchmarks differ in both scoring
  function and molecular library; the design cannot attribute V.3's convergence
  difference to either.
- **Backbone version.** The comparison repository uses Uni-Mol v1 (512d) and
  GROVER 1600d; this work uses Uni-Mol2 and GROVER 3400d. A direct test on
  `d4`, which carries both Uni-Mol versions under one oracle, is **in
  progress**.
- **Cross-dataset matching.** Only MolFormer, Morgan and descriptors exist on
  both benchmarks; the two representations carrying the EnamineHTS result are
  not yet computed on DOCKSTRING.
- **Scale.** `ampc` and `d4` are prepared and verified but not yet swept.
- **Fusion across budgets** — *addressed for the winning pair (IV.4c)*; the
  all-six arms remain at 200k, and nothing is replicated across seeds.
- **Embedding provenance** — *resolved, IV.6*. The matched 1536-d extraction
  was screened and loses on every metric, so the EnamineHTS results are not an
  artifact of the anomalous file. Single seed.
- **Active learning in the non-converging regime**, where the decoupling
  question is most meaningful.

---

## Part VIII — Proposed paper and next steps

**Question.** When does molecular representation choice matter for
docking-score surrogates?

**Claim (as supportable today).** Across single-seed learning curves,
molecular-representation rankings and performance gaps depend strongly on
labelling budget and on the docking benchmark. The evaluated representations
converge to a narrow common band on PARP1/Vina, whereas a large
structure-versus-SMILES gap persists on EnamineHTS/DOCK3.7 through 1.9 million
labels and across linear, tree-based and neural probes. Matched-representation
and multi-seed experiments are underway to determine whether the contrast
generalises across targets and scoring environments.

**Contributions.** A budget-resolved benchmark from 250 to 1.9M labels;
cross-benchmark evaluation over two docking engines; head-robust evaluation
with linear, tree and neural probes; a size-controlled descriptor baseline and
its protocol; and an error-variance criterion predicting when fusion helps.

**Minimum further experiments.** Seeds throughout; Uni-Mol2 and MHG-GED on the
DOCKSTRING molecule set so cross-dataset claims use a matched representation
set; `ampc` and `d4` budget points as external DOCK3.7 validation; the
backbone-version control.

**Extension worth considering.** Treat *representation sensitivity* — the
spread across representations — as the dependent variable, and predict it from
landscape properties such as size-explained variance, which already ranges from
4% to 43% across targets. DOCKSTRING's 58 targets share one molecule set, so
embeddings extracted once serve all of them and each additional target costs
only surrogate training. That converts a ranking table into an account of
*when* rankings differ.

**Immediate queue.**

1. Finish `d4` Uni-Mol v1 versus Uni-Mol2 (running).
2. Seeds on all headline comparisons.
3. Extract MHG-GED, SMI-TED, Uni-Mol2 on DOCKSTRING — the bottleneck for every
   cross-dataset claim.
4. Budget points on `ampc` and `d4`.
5. Ridge probes over all 58 DOCKSTRING targets; compute landscape descriptors
   and per-target sensitivity.
6. Active learning in the non-converging regime.

---

## Appendix — supporting documents

| File | Contents |
|---|---|
| `docs/FINDINGS.md` | F1–F8 with correction history preserved |
| `docs/LOG.md` | Chronological record including dead ends |
| `docs/LITERATURE.md` | Literature review and novelty assessment |
| `docs/SOURCE_PAPER.md` | Analysis of the source paper, including two design flaws not to repeat |
| `docs/CROSS_REPO_SYNTHESIS.md` | Relationship to `al-eval-framework`, with retractions visible |
| `al-eval-framework/CROSS_REPO_FINDING.md` | Companion file in that repository |
| `work/enhits/results.csv` | All runs, machine-readable |
