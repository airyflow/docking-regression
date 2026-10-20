# Cross-repo synthesis: `docking-regression` and `al-eval-framework`

Written 2026-10-16, after discovering that these two projects had been
answering the same question on different data and reaching opposite
conclusions.

A copy of the headline result is filed in `al-eval-framework/CROSS_REPO_FINDING.md`.

---

## The two projects

| | `docking-regression` | `al-eval-framework` |
|---|---|---|
| Data | EnamineHTS 2.1M, AmpC 98M, D4 116M | DOCKSTRING 50k × 4 targets |
| Oracle | DOCK3.7 | AutoDock Vina |
| Task | static regression | AL loop (MolPAL), AUDC |
| Representations | 5 pretrained + 12 descriptors | 4 + 2 fusion variants |
| Extras | descriptor baseline, size control | C1–C4 diagnostics, decoupling analysis |

They overlap on the central axis — *does the choice of frozen molecular
representation matter for predicting docking scores* — and they disagree.

---

## Finding 1 (CORRECTED): DOCKSTRING saturates, EnamineHTS does not

**An earlier version of this section claimed representation choice was
"irrelevant" on DOCKSTRING, with a spread of 0.019 against EnamineHTS's 0.349.
That comparison used 45,000 training rows. `al-eval-framework`'s `static_eval`
uses `train_frac=0.005` --- 250 molecules. At the right budget the conclusion
inverts.**

Spread across representations as a function of training budget, DOCKSTRING
PARP1 / Vina, protocol held fixed:

| Train rows | spread (Pearson) | best | worst |
|---:|---:|---|---|
| 250 | **0.297** | descriptors (0.845) | Morgan (0.548) |
| 1,000 | 0.171 | descriptors (0.871) | Morgan (0.700) |
| 5,000 | 0.064 | descriptors (0.877) | Morgan (0.813) |
| 45,000 | 0.019 | GROVER (0.910) | descriptors / Uni-Mol (0.891) |

At `al-eval-framework`'s actual budget the spread is 0.297, essentially
matching EnamineHTS's 0.349. The ordering also fully reverses with budget:
descriptors run 1st at 250 rows and last at 45k; Morgan runs last and then 2nd.

### The real contrast

| Dataset | spread @ ~50k | spread @ max | converges? |
|---|---:|---:|---|
| DOCKSTRING PARP1 / Vina | 0.019 (45k) | 0.019 | **yes, by 45k** |
| EnamineHTS / DOCK3.7 | 0.349 (50k) | 0.295 (1.9M) | **no** |

Every representation reaches 0.89--0.91 on Vina. On DOCK3.7, MolFormer and
SMI-TED plateau at 0.66--0.69 while Uni-Mol2 reaches 0.956 and the gap survives
1.9M rows.

**The Vina task has a low ceiling every representation reaches; the DOCK3.7
task has a higher ceiling only structure-aware representations reach.**

### Eliminated

- **Protocol.** Running this repo's pipeline on DOCKSTRING reproduces
  `al-eval-framework`'s ordering (MolFormer > Uni-Mol v1). Both pipelines agree
  on the same data.
- **Training-set size.** Swept directly above.
- **The size confound.** Heavy-atom correlation is -0.659 on EnamineHTS and
  -0.648 on PARP1 --- nearly identical size dependence, very different
  convergence behaviour.

### Still open

1. **Oracle.** Vina vs DOCK3.7 scoring.
2. **Molecule distribution.** ChEMBL-derived drug-like vs screening-diverse.
3. **Backbone version.** Uni-Mol v1 512d vs Uni-Mol2 1536d; GROVER 1600d vs
   3400d. Being tested on d4, which carries both Uni-Mol versions under one
   oracle --- the cheapest to settle, so it goes first.

### Per-target size dependence

| Target | heavy-atom vs score | variance explained |
|---|---:|---:|
| EnamineHTS / DOCK3.7 | -0.659 | 43.5% |
| PARP1 / Vina | -0.648 | 42.0% |
| F2 / Vina | -0.623 | 38.8% |
| **ESR2 / Vina** | **-0.199** | **4.0%** |

ESR2 is nearly size-free, making it the cleanest target for isolating genuine
chemical signal from bulk-property fitting.

---

## Finding 2: the decoupling hypothesis is not supported, and the power is low

`al-eval-framework/DESIGN.md` states the hypothesis as *"static prediction
performance does **not** reliably predict active-learning sample efficiency."*
The measured Spearman between static-RMSE rank and AL-AUDC rank:

| Subset | pooled | EGFR | ESR2 | F2 |
|---|---:|---:|---:|---:|
| all 6 (as published) | 0.886 | 1.000 | 0.943 | 0.943 |
| 4 independent (drop derived fusions) | 0.800 | 1.000 | 0.800 | 0.800 |
| 3 learned only (also drop Morgan) | 1.000 | 1.000 | 0.500 | 1.000 |

Ranks agree rather than decouple, and the agreement is **robust** to the two
design features that could have manufactured it: `fusional_lite` and
`fusional_full` are concatenations of the other representations and might have
anchored both rankings, and Morgan is a clear worst on both axes. Removing
either leaves the coupling intact.

`fusional_full` is rank 1 on *both* metrics, where DESIGN.md expected it to
"win under the AL evaluation despite not necessarily winning under static
RMSE."

**But the power is poor.** With 4 independent representations, rho = 0.8
corresponds to p ≈ 0.33. The honest statement is *the predicted decoupling did
not appear*, not *static reliably predicts AL*.

### A proposed mechanism, now withdrawn

An earlier version of this document argued that coupling was unsurprising
because representation choice "barely matters" on DOCKSTRING, and that the
hypothesis had therefore been tested on the dataset least able to falsify it.

**That argument was wrong and is withdrawn.** It rested on the 45k-row spread
of 0.019; at the 250-row budget `static_eval` actually uses, the spread is
0.297. There was ample room for the two rankings to diverge, and they did not.
The coupling result stands on its own terms and needs no explaining away ---
only more statistical power.

### An unresolved discrepancy

`al-eval-framework` reports static RMSE of 0.574–1.016 on DOCKSTRING; the
protocol-matched run here gets 0.495–0.540 on the same molecules — roughly
half. Most likely the static model there is trained at an AL-sized budget
rather than on 45k rows. The two projects are therefore not measuring "static
performance" identically, and the coupling claim depends on which definition is
used. **Confirm before building on it.**

---

## What follows

**The question to ask is not "which representation is best" but "when does
representation choice matter at all?"** Two datasets answer it oppositely under
one protocol, and that contrast is a sharper contribution than either project
had alone.

**Test decoupling where representations fail to converge.** Not because
DOCKSTRING is flat --- at AL-relevant budgets it is not --- but because it
converges by 45k while EnamineHTS does not. A regime where some
representations never catch up is a different test of whether AL efficiency
tracks static accuracy. This merges the two projects naturally:
`docking-regression` supplies the non-converging regime, `al-eval-framework`
supplies the AL loop and the C1–C4 diagnostics.

**Ordering.**

1. Rule out backbone version — run Uni-Mol2 and GROVER-3400 on DOCKSTRING, or
   Uni-Mol v1 on EnamineHTS.
2. Reconcile the two definitions of static performance.
3. Run the AL loop on EnamineHTS with the representations that actually differ.
4. If the spread gap survives (1), the oracle/distribution contrast becomes the
   paper.
