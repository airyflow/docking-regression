# Which repository owns which claim

Written 2026-10-20. The canonical copy lives here; `al-molecular` and
`al-eval-framework` carry short pointers to it.

This exists because three repositories in this workspace were independently
answering overlapping questions about molecular representations, and on
2026-10-16 two of them turned out to disagree. `al-eval-framework` ranked
MolFormer near-best and Uni-Mol poor; `docking-regression` ranked Uni-Mol2
best and MolFormer worst. Resolving it took most of a day and produced two
retractions, and the eventual answer was that both were right about their own
data. That is a cheap outcome compared to discovering the same thing during
review of a manuscript, which is what this document is meant to prevent.

The rule underneath everything below is that a claim belongs to the repository
whose experiment is designed to test it, and the others cite rather than
re-derive it.

---

## docking-regression — static prediction

This repository owns everything about predicting a docking score from a frozen
molecular embedding with no acquisition loop: how representations rank, how
that ranking changes with labelling budget, whether the ranking survives a
change of prediction head, how much of any result is molecular size, and when
combining representations helps.

Concretely that means the EnamineHTS budget sweep from 250 to 1.89M labels, the
DOCKSTRING PARP1 comparison, the `d4` Uni-Mol version test, the descriptor
baseline and its size-control protocol, the ridge and LightGBM probes, and both
families of fusion. Fifteen findings are recorded in `docs/FINDINGS.md`, with
the narrative in `docs/PROGRESS_REPORT.md`.

What it deliberately does not own is anything about acquisition. It has no AL
loop and should not grow one; when a question is about what happens across
rounds of selection, it belongs in `al-molecular`.

## al-molecular — the active-learning loop

This is the most methodologically mature of the three, and it owns the
acquisition side: which molecules to pick given a model, what the surrogate
trains on, and when it retrains. Its benchmark is recall of the true top-1000
on ENHITS at a fixed label budget, with paired per-seed comparisons and
pre-registered decision rules.

Its settled results are that the acquisition axis is closed — 145 pre-registered
runs established that no choice of β, no schedule and no learned controller
beats greedy — and that the top-weighted loss, more rounds, and the prefilter
are the interventions that survive. Its most useful strategic finding is that
correctness fixes and representation changes moved recall roughly six times
more than the best deliberate technique, which is what makes the static work in
`docking-regression` worth doing at all.

It also owns the embedding extraction pipeline and the provenance record for
the ENHITS files, which matters for the shared dependency below.

## al-eval-framework — the relationship between the two

This repository owns the question neither of the others can answer alone:
whether static prediction performance predicts active-learning sample
efficiency. It runs a MolPAL-based loop on four DOCKSTRING targets with
AutoDock Vina, and adds the C1–C4 diagnostics — local smoothness, predictive
calibration, exploration diversity, target-aware organisation — intended to
explain any decoupling.

Its measured result contradicts its own stated hypothesis: static-RMSE rank and
AL-AUDC rank agree, with Spearman between 0.80 and 1.00 depending on the
subset. The agreement is robust to dropping the derived fusion variants and to
dropping Morgan, but it is underpowered, since four independent representations
give p ≈ 0.33. The honest statement is that the predicted decoupling did not
appear, not that static reliably predicts AL.

## The others

`FusionAL` holds earlier AL fusion work that `al-molecular` has largely
superseded; its `molpal/analysis/criteria.py` is still the source of the C1–C4
implementations. `benchmarking_molecular_models` is Praski et al.'s own code,
kept as reference for the benchmark this work positions against.
`al-embed`, `al-screen`, `molpal-fusion-hts` and `optimol` are older or
infrastructural and make no live claims.

---

## Where they overlap, and what to do about it

Three overlaps are real rather than apparent, and each has a resolution.

**The ENHITS pool is used by both `docking-regression` and `al-molecular`, on
the same embedding files.** Those files were not produced by either pipeline —
they were inherited from a colleague and extracted with different settings,
which is why the Uni-Mol2 there is 768-dimensional with effective rank 2.2
rather than this project's own 1536-d. Both repositories' ENHITS numbers move
together if those files are ever re-extracted. That dependency is recorded in
`docking-regression` §II.5 and `al-molecular` §8.5b, and neither should be
updated without the other.

Relatedly, the two reached the same conclusion about that embedding by
different routes: a leave-one-out AL experiment found dropping it costs 0.063
recall, while static regression found it the best single predictor at Pearson
0.9564. Agreement from unrelated instruments is the strongest validation
available here, and it is cited in both directions rather than claimed twice.

**All three repositories can produce a representation ranking.** Only
`docking-regression` should publish one. The other two consume it: if
`al-molecular` needs to choose a backbone, the static screen answers that at
the cost of one training run rather than an AL campaign — which is exactly how
the `bb5fix` question was settled, and the screen predicted the swap would
lose.

**`docking-regression` and `al-molecular` both evaluate fusion.** The split is
by stage: `docking-regression` owns fusion as a static modelling question —
feature-level versus prediction-level, when weighting beats averaging, the
error-variance criterion — while `al-molecular` owns fusion inside the loop,
where the members are ensemble heads and the metric is recall.

---

## What this means for writing

The planned paper lives in `docking-regression`. It should cite
`al-eval-framework` for the static-versus-AL relationship and `al-molecular`
for the AL-side magnitudes, rather than reproducing either. If an AL validation
experiment is wanted for that paper, it should be run in `al-molecular` with
its existing harness and referenced, not reimplemented.

The reverse also holds: `al-eval-framework`'s manuscript draft, *What Makes a
Molecular Representation Good for Active Learning?*, should cite
`docking-regression` for the static rankings it uses as the independent
variable, rather than measuring them again with a different protocol. The one
time both measured the same thing independently, the protocols differed enough
— 250 training rows versus 45,000 — that the comparison looked like a
contradiction for a day.

## Not creating a fourth repository

Reproducing Martínez Galindo et al. in a new repository was considered and
rejected. The released code is missing five of seven encoders, the
knowledge-graph builder, the pretraining loop, the FFNN baseline, the ADMET
pipeline and all the analysis, so it would be a reimplementation from the
Methods section rather than a reproduction — months of work confirming findings
that are mostly unsurprising, one of which is confounded. Nothing in the
current paper plan depends on their numbers. The two pieces worth taking from
them, their jointly-trained intermediate fusion and their statistical
apparatus, belong in `docking-regression` where the comparison already lives.
A tutorial covering that paper and its code is at
`otter-knowledge/docs/TUTORIAL.md`.
