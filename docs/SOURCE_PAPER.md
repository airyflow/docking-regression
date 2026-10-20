# Analysis: Martínez Galindo et al., *Nat. Commun.* 17:8081 (2026)

*Impact of molecular multimodality on neural network models for prediction
tasks related to drug discovery.* IBM Research Dublin.
DOI [10.1038/s41467-026-74487-x](https://doi.org/10.1038/s41467-026-74487-x)

The paper this project started from. Read in full 2026-10-15, re-read
2026-10-16. Recorded here because its design decisions, and two flaws in them,
directly shape what our study should and should not do.

Related: [LITERATURE.md](LITERATURE.md) for the wider field.

---

## The open questions they pose

Framed against a stated gap: prior multimodal work "focused on **early** fusion
strategies, in which all modalities are combined during pretraining." They
address intermediate and late fusion instead — and never test early fusion
themselves.

| # | Question (their words) | Where |
|---|---|---|
| Q1 | "Is the incorporation of multiple modalities always beneficial?" | Intro |
| Q2 | "A crucial design choice is the fusion strategy, how strongly does it influence performance?" | Intro |
| Q3 | "What is the impact of multimodality when knowledge-enhanced representation learning is applied?" | Intro |
| Q4 | "Is there a best modality? Is there a modality that guarantees better performance when combined with others?" | Results |

---

## How they addressed them

A three-dimensional factorial grid.

**Architectures (2)**
- **FFNN** — one FC layer per modality into a shared 512-d space, drug
  embeddings averaged across modalities, protein from ESM alone, prediction =
  inner product. MSE loss, 25,000 epochs, lr 1e-4, best model by validation
  Pearson.
- **Otter-Knowledge (OK)** — R-GCN on a multimodal KG built from BindingDB +
  UniProt (Drug / Protein / Measurement entities). Encoder projects each node
  modality to 512-d, one hidden R-GCN layer, skip connections; decoder is a
  binary link-prediction head plus an RMSE regression objective. GNN
  autoscaling (GAS) with METIS, 1000 partitions, 73 epochs, lr 1e-4, **no early
  stopping and no best-model selection**. Pretrained encoders are then
  fine-tuned on the benchmark, with four FC layers and a regression head.

**Fusion (2)** — intermediate (shared latent space, jointly processed) and late
(per-modality predictions averaged).

**Modalities (7 + 1)** — ChemBERTa, MolFormer, SMI-TED, Morgan fingerprint,
MolGen, ImageMol-with-classification-layer, Uni-Mol; plus ESM-2 650M for
protein sequence.

**Benchmarks (29)** — TDC DTI: DG (temporal split), DAVIS and KIBA (random,
drug and target splits) = 7 splits. Plus 22 TDC ADMET. ADMET uses FFNN only,
since with no protein sequences the KG would collapse to a single drug node.

**Scale** — 1467 models, 6039 experiments, 5 predefined seeds each, results
averaged. 39 intermediate-fusion combinations (restricted to at most one
sequence-based model, as CB/MOLF/SMI are all SMILES-pretrained and MG is
SELFIES), 127 late-fusion combinations (no restriction, since combining already
trained models is cheap).

**Leakage control** — overlapping triples between test sets and the pretraining
KG were removed.

**Analysis machinery**
- Relative importance: `(G6−G1)/|G1|`, `(LF−IF)/|IF|`, `(OK−FFNN)/|OK|`
- Wilcoxon signed-rank tests
- Random Forest surrogate over (modality indicators, dataset type,
  architecture, fusion) → SHAP TreeExplainer, leave-one-dataset-out CV
- Holm-adjusted Conover tests for pairwise feature importance
- Pairwise Pearson correlation between unimodal predictions (complementarity)
- Cosine similarity between embeddings via PCA (128 components) → GCCA (32)

---

## Their answers

**Q1 — yes on average, no in particular.** Relative importance of 1→6
modalities averages **0.151**. Interquartile ranges shrink and quartiles rise
with modality count. But "the optimal setup for a given dataset rarely includes
all available modalities." Wilcoxon significant in most arms **except FFNN-IF,
where there is no significant difference at all.**

**Q2 — fusion is the strongest lever.** LF beats IF in the vast majority of
cases, significantly, for both architectures. The effect is larger for FFNN,
which they read as "LF is especially beneficial when no knowledge-enhanced
representation learning is used."

**Q3 — knowledge helps, weakly and unevenly.** OK > FFNN in most cases but
"not as clear as the fusion approach," and reversed on Kiba target. OK's
advantage is larger under IF than LF. Their own Discussion concedes FFNN-LF
"often achieves superior performance."

**Q4 — no best modality, but one standout.** Score distributions are "broadly
similar," so "most modalities provide informative representations." MolFormer
and Morgan fingerprint are strong and stable; SMI-TED underperforms; ImageMol
is architecture-dependent (−0.06 SHAP under FFNN-IF, +0.02 under OK-LF).
Statistically, **only Uni-Mol and late fusion had SHAP values significantly
different from zero**, and Conover placed both above every other feature.
Uni-Mol is also least cosine-similar to the others — complementary geometric
information — while MOLF/SMI/CB/MG cluster tightly, reflecting shared SMILES
inductive bias and redundancy.

---

## Contributions

1. **The grid itself.** Converts an assumption "widely discussed in the
   research community but had not been systematically evaluated before" into a
   measured, statistically tested result. This is the core contribution: the
   paper introduces no new architecture, encoder or algorithm.
2. **State of the art** on DTI-DG (Pearson 0.58 → 0.64, +10.34%) and on Half
   Life Obach, Clearance Hepatocyte AZ, DILI, BBB Martins, CYP2C9 substrate and
   CYP3A4 substrate.
3. **A decision aid** — Figure 2 quantifies where to spend compute across the
   three levers.
4. **Cost/accuracy analysis** — Table 1: CB-IMC-UM late fusion 0.6773 at
   71,138,382 parameters; the same trio under intermediate fusion 0.6443 at
   20,338,197; single modalities ~0.60–0.62 at ~17M. IF adds parameters
   marginally; LF needs one model per modality but afterwards every subset is
   explorable and new modalities integrate without retraining.
5. **Released artifacts** — six supplementary CSVs of all experiments, a
   results website, and single-modality pretrained models on HuggingFace.

---

## Two methodological problems

### 1. The Q1 answer is confounded with Q2

Figure 1's own reported group sizes:

| Modalities | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Intermediate fusion | 7 | 15 | 13 | 4 | **0** | **0** |
| Late fusion | 7 | 21 | 35 | 35 | 21 | 7 |

Intermediate fusion has **no 5- or 6-modality experiments**, because of the
one-sequence-model restriction. So `G6` in `(G6−G1)/|G1|` is late-fusion only,
and the headline 0.151 "benefit of more modalities" mixes modality count with
fusion strategy. Consistently, the arm that cannot reach high counts (FFNN-IF)
is exactly the one showing no significant modality-count effect.

The restriction is defensible on cost and redundancy grounds. The conclusion is
stated more broadly than the design supports.

### 2. The architecture comparison uses asymmetric protocols

By their own description: no early stopping or best-model selection for OK,
but best-model selection for FFNN, justified as fairness because OK is the more
complex architecture. This **handicaps OK**, which makes OK's wins more
credible but renders the magnitude in Figure 2 unreliable. The two arms are not
trained under the same protocol.

### 3. No trivial baseline anywhere in 1467 models

Nothing in the paper establishes that any configuration beats simple
physicochemical descriptors. Given Deng et al. (*Nat. Commun.* 2023) finding
RDKit2D descriptors frequently beat learned representations — and specifically
warning that concatenating descriptors onto GROVER inflates its apparent
advantage — this is a live concern for the Figure 7 modality analysis, which
may partly be measuring which embeddings encode size and polarity.

---

## What this means for our study

This paper cleared *Nature Communications* on coverage, established
leaderboards, released resources and timeliness — not on a new idea. There is
no new architecture, encoder or algorithm in it. We cannot win the same way:
three pools against twenty-nine benchmarks is not a contest. What we have that
they lack is a controlled descriptor baseline, a size-decomposition protocol,
genuine reproducibility — their Code Availability concedes dependence on IBM
proprietary infrastructure and deleted checkpoints — and pools three to five
orders of magnitude larger.

The inheritance worth taking is structural. Their three-axis design, modality
crossed with architecture crossed with fusion, is a good skeleton, and we have
adapted it by swapping their architecture axis for a prediction-head axis,
since with one fixed receptor their knowledge-graph arm has no drug–protein
edges to exploit. Their statistical apparatus is worth copying outright: five
seeds, Wilcoxon signed-rank, Holm-adjusted Conover for pairwise importance, and
complementarity measured through prediction correlation rather than asserted.
We have done the correlation analysis and partly done the seeds; the paired
significance tests remain missing and are cheap, since every representation
shares a split. Their cost and parameter table is also worth reproducing.

Their failures instruct as much as their methods. Never compare across modality
counts when the available counts differ by fusion arm, which is how their
headline 0.151 ended up mixing modality count with fusion strategy. Never
compare architectures trained under different protocols. And always carry a
trivial baseline, because 1,467 models that never meet twelve RDKit descriptors
cannot establish that any of the machinery earned its place. Our fixed head and
shared splits already guard against the second trap; the first is a live risk in
our own fusion work, where subset sizes vary by arm.

What transfers least well is their actual conclusions. Their Q4 finding that
"most modalities provide informative representations", with broadly similar
distributions, does not hold here: we see a bimodal split with Uni-Mol2 and
MHG-GED near Pearson 0.95, SMI-TED and MolFormer at 0.66–0.69, and twelve free
descriptors at 0.89 between them (FINDINGS F5). Their Q1 and Q2 conclusions
fare no better — the best subset on our data is two of six at every budget
tested, and feature-level concatenation of the right pair beats every
prediction-level method we implemented, which is the reverse of their
late-beats-intermediate result. Their Uni-Mol finding is the one point of
agreement, and a notable one: it is their only statistically significant
modality, and our best backbone is also the 3D conformer model.

That docking behaves differently is not surprising once you notice what a
DOCK3.7 score is. It is a physics sum — van der Waals, Poisson–Boltzmann
electrostatics, AMSOL desolvation — evaluated over a posed 3D conformer, rather
than an experimental measurement of the kind their ADMET and DTI endpoints
record. A representation carrying geometry should transfer to it; a string
model has further to go.

## What follows for docking-score prediction

Three lines of work follow, in increasing order of ambition.

The nearest is finishing what is underway: seeds on the full-budget and
DOCKSTRING results, extending DOCKSTRING with MHG-GED, SMI-TED and Uni-Mol2 so
the cross-dataset comparison rests on more than the two matched representations
it currently has, and running `ampc` as a third DOCK3.7 pool to test whether
the family split is specific to EnamineHTS.

The most distinctive opportunity is one this paper could not have taken, and it
engages a different literature. MolPAL and its successors report recovering
roughly 90% of the top-50k on AmpC and D4, and as far as our literature search
reached, none of them reports what twelve free descriptors recover on the same
split. Given descriptors reach EF@1% of 40.7 on EnamineHTS, that number is
worth knowing. Supplying a null baseline for the ultra-large docking surrogate
benchmark is specific, checkable, and argues with named prior work rather than
adding another leaderboard. The caveat is that it needs pool-level retrieval —
recall of the true top-N after acquiring M — which is not the random-split
enrichment we currently compute, so it needs new evaluation code.

The most ambitious would treat representation sensitivity — the spread across
representations on a given target — as the quantity to be explained rather than
reported, and predict it from landscape properties such as size-explained
variance, which already ranges from 4% on ESR2 to 43% on EnamineHTS.
DOCKSTRING's 58 targets share one molecule set, so embeddings extracted once
serve all of them and each additional target costs only surrogate training.
That converts a ranking table into an account of when rankings matter at all,
which is a better paper than a comparison — but it depends entirely on the
DOCKSTRING extraction in the first line of work, and should not be started
before it.
