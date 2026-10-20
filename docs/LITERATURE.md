# Literature review

Carried out 2026-10-16 to test the novelty assumption flagged in
[FINDINGS.md](FINDINGS.md) and the `\todo` in the paper's related-work section.

**Bottom line: the novelty is narrower than we assumed. Two of the three things
we thought were contributions are already established. One genuine gap
remains, and it is more specific — and more defensible — than the framing we
had.**

---

See [SOURCE_PAPER.md](SOURCE_PAPER.md) for a full analysis of the Nature
Communications multimodality paper that prompted this project.

## What is already established

### 1. Docking scores are size-biased — thoroughly known

This is textbook medicinal chemistry, not a finding. Docking scoring functions
sum per-atom interaction terms, so larger ligands accumulate more negative
totals almost mechanically. The standard correction, **ligand efficiency**
(score ÷ heavy-atom count), dates back decades and is built into commercial
software ([Schrödinger KB](https://www.schrodinger.com/kb/1622)). Recent work
re-examines its statistical pitfalls — LE embeds its own size bias and its
uncertainty scales inversely with heavy-atom count
([Rethinking Ligand Efficiency, ACS Med. Chem. Lett. 2025](https://pubs.acs.org/doi/10.1021/acsmedchemlett.5c00652)).

Our measured −0.66 correlation on EnamineHTS/DOCK3.7 is a confirmation, not a
discovery. **We must not present it as novel.**

### 2. Simple descriptors beating learned representations — established

This is the one that most damages our earlier framing.

- **[Deng et al., *Nat. Commun.* 2023](https://www.nature.com/articles/s41467-023-41948-6)**
  — fixed RDKit2D descriptors frequently beat learned representations from
  RNNs, GCNs, GINs and pretrained models. Also warns that *concatenating*
  RDKit2D descriptors onto GROVER inflates its apparent advantage, so reported
  gains are not from representation learning alone. Directly relevant to our
  fusion experiments.
- **[Praski, Adamczyk & Czech, arXiv:2508.06199](https://arxiv.org/abs/2508.06199)**
  — 25 pretrained models × 25 datasets under hierarchical Bayesian testing.
  Nearly all neural models show negligible or no improvement over a plain ECFP
  fingerprint; the only exception is itself fingerprint-based.
  [Code](https://github.com/MLCIL/benchmarking_molecular_models).
- **[Frozen SMILES embeddings vs fingerprints (ChemRxiv preprint)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004188/v1)**
  — classical baselines win 36 task-rows to 7 on ADMET and CNS bioactivity.

So "pretrained molecular embeddings fail to beat simple baselines" is a known
result in molecular property prediction. Our F5 (MolFormer losing to 12
descriptors) is **consistent with** this literature rather than contradicting
it — which is reassuring for correctness and deflating for novelty.

### 3. Trivial baselines beating docking itself — established

- **[Evaluating molecular docking for binding affinity, *J. Comput. Aided Mol. Des.* 2026](https://link.springer.com/article/10.1007/s10822-026-00929-9)**
  ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13526069/)) — on the
  Schrödinger JACS set, most docking programs **did not outperform a molecular
  weight regression baseline** (MW null model: mean R² 0.23 ± 0.18; Vina 0.10 ±
  0.08). The authors explicitly frame MW regression as "a useful null model for
  determining how much additional ranking information is contributed by the
  docking calculations." This is our argument, applied one level down.
- **[DOCKSTRING, *JCIM* 2022](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334)**
  ([arXiv](https://arxiv.org/abs/2110.15486)) — docking scores gave the highest
  enrichment overall, but were surpassed by **logP** on a few targets.

---

## What appears genuinely open

Searches did not surface any of the following. This is suggestive, not proof of
absence (see caveats).

### Gap A — No descriptor null baseline in the ultra-large docking surrogate literature

The AmpC/D4 libraries from Lyu et al. are the standard benchmark for
ML-accelerated docking, and a substantial literature reports top-*k* retrieval
on them:

| Work | Reported result |
|---|---|
| [MolPAL (Graff et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8188596/) | D-MPNN + greedy: ~90% of top-50k AmpC, ~80% D4 |
| [Robustness assessment, *ACS Omega* 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12019444/) | top-100k retrieval ~60–90% across both targets |
| MEMES | 90% of top-1000 AmpC scanning 6% |
| GEM-Screen | up to 95% of virtual hits scanning 4% |

MolPAL's own ablations compare RF vs feedforward vs MPNN **on fingerprints** —
not against physicochemical descriptors. None of these papers, as far as the
search reached, reports what a trivial descriptor model recovers on the same
split. Given that our descriptors reach EF@1% = 40.7 on EnamineHTS, **a null
baseline for these published retrieval numbers is a concrete and missing
contribution.**

### Gap B — No size-decomposition protocol for ML docking surrogates

Size bias is known and ligand efficiency exists, but we found no work that
decomposes a *surrogate model's* performance into size-driven and
size-independent components. The nearest neighbours do something different:

- **[Scoreformer, arXiv:2406.09346](https://arxiv.org/abs/2406.09346)** trains
  on the lightest 90% by molecular weight and tests on the heaviest 10% — an
  out-of-distribution *generalization* test. It reports R² falling 0.52 → 0.33.
  That asks whether a model extrapolates across size, not how much of its
  in-distribution performance *is* size. It also uses no descriptor baseline.
- Ligand efficiency normalises the *label*. Our within-stratum protocol
  conditions the *evaluation*, which avoids LE's known normalisation pathologies.

### Gap C — Docking-surrogate-specific descriptor comparison at pool scale

Deng et al. and Praski et al. cover experimental property prediction on
datasets of 10²–10⁴ compounds. Nothing found covers docking-score regression at
10⁶–10⁸. The storage argument (TB of embeddings vs 12 free features) only
exists at this scale.

---

## Implications for framing — revised

**Drop:** "foundation-model embeddings don't beat trivial descriptors" as the
headline. Praski et al. and Deng et al. already own that claim in the general
case, and a reviewer who knows them will say so.

**Keep and sharpen:**

1. **A null baseline for the ultra-large docking surrogate benchmark.** The
   AmpC/D4 retrieval literature has no trivial control. Supplying one — and
   showing how much of the celebrated 90% top-50k retrieval a 12-feature model
   achieves — is specific, actionable and new. This should probably be the
   paper.
2. **The size-decomposition protocol.** Novel as applied to surrogates, and it
   avoids LE's known pathologies by conditioning the evaluation rather than
   normalising the label. Positions naturally against the MW-null-model
   argument in the JCAMD 2026 paper.
3. **Scale and cost.** Genuinely unexplored, and the practical decision
   (1.5 TB of GROVER vs 12 descriptors) is real.

**New risk identified:** Deng et al.'s finding that concatenating descriptors
onto GROVER inflates apparent advantage applies directly to our fusion arm. Any
fusion that mixes descriptors with embeddings must report the descriptor-only
and embedding-only arms alongside, or it reproduces the error they flag.

---

## Read in full before writing related work

Priority order. All conclusions above come from search results and abstracts,
not full texts.

1. Praski et al. arXiv:2508.06199 — closest prior art for the negative result
2. Deng et al. *Nat. Commun.* 2023 — descriptor baselines + the concatenation
   warning
3. Graff et al. MolPAL + supplementary — confirm no descriptor baseline exists
4. JCAMD 2026 Schrödinger evaluation — the MW null model argument
5. DOCKSTRING — the logP-beats-docking enrichment result
6. ACS Omega 2025 robustness assessment — the most directly comparable
   retrieval numbers on our exact pools

---

## Caveats on this review

- Based on web search and abstracts; two PDF fetches failed on size limits and
  the ACS DOCKSTRING page returned 403. **No full texts were read.**
- Absence of evidence is weak here: a negative search result does not establish
  that no such paper exists, particularly for a control that might appear as an
  unheadlined supplementary table.
- Several cited works are 2025–2026 and some are preprints. Peer-review status
  should be checked before citing.
- Searches were in English and keyword-driven; a paper using different
  terminology (e.g. "property-matched", "size-normalised enrichment") could
  have been missed.
