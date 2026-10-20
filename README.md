# docking-regression

Predicting docking scores from pre-computed molecular embeddings, across four
labelled pools and up to six backbones each.

Each pool is a single receptor, so there is no protein input: a row is one
molecule and the label is its docking score. This makes the task a plain
regression on molecule embeddings, which is why nothing here carries over from
the drug–target interaction codebases — the protein tower and the
knowledge-graph GNN both have nothing to relate when the target never varies.

What the repo asks: **which backbone carries the signal, how much does
embedding width cost you, and what does averaging backbones buy?** The model
head is held fixed across every backbone so those comparisons mean something.

## Pools

| Pool | Molecules | Target | Oracle | Backbones |
|---|---:|---|---|---:|
| `enhits` | 2,104,318 | EnamineHTS | DOCK3.7 | 5 |
| `ampc` | 98,489,350 | AmpC β-lactamase | DOCK3.7 | 6 |
| `d4` | 116,225,927 | D4 dopamine receptor | DOCK3.7 | 6 |
| `enamine_real` | 1,531,875,935 | SurA chaperone | Glide HTVS | not yet usable |

`enamine_real` is deliberately absent from the registry. Its embeddings exist
only as unstitched per-chunk `.npy` files (MHG-GED 339/1000, Uni-Mol 146/1000);
nothing can run on it until a backbone is complete and stitched.

### Backbone dimensions

| Backbone | `enhits` | `ampc` / `d4` |
|---|---:|---:|
| GROVER3400 | 3400 | 3400 |
| Uni-Mol2 | **768** | **1536** |
| MHG-GED | 1024 | 1024 |
| MolFormer | 768 | 768 |
| SMI-TED | 768 | 768 |
| Uni-Mol v1 | — | 512 |

Two traps the registry encodes so you don't have to remember them:

- `enhits` stores GROVER as `grover_embeddings.npy`, the other pools as
  `grover3400_embeddings.npy`.
- **`enhits` Uni-Mol2 is 768-dim, not 1536** — a different extraction from the
  one on `ampc`/`d4`. Do not compare `unimol2` numbers across those pools.
- `enhits` has no Uni-Mol v1 at all.

### Cross-oracle comparability

`enamine_real` uses Glide HTVS; the other three use DOCK3.7. Absolute scores
are not comparable across oracles — recall and enrichment are, raw RMSE and
score distributions are not.

## Row alignment

This is the part that silently ruins results, so it is verified in code rather
than assumed. Verified on disk 2026-10-16:

- **`ampc`, `d4`** — `dedup/<pool>_smiles.txt`, `dedup/<pool>_scores.csv.gz`
  and every `dedup/*_embeddings.npy` share one row order.
- **`enhits`** — the embeddings live in `embed_shuffled/`, whose row *i* is
  original row `perm[i]` of `shuffle_permutation.npy`. The scores come from
  `al-molecular/data/EnamineHTS_scores.csv.gz`, which is row-aligned with
  `embed/enhits_large_smiles.txt` (the *unshuffled* file) and is sorted by
  score. `prepare.py` applies the permutation so the stored score array always
  matches embedding row order.

`prepare.py` checks the SMILES column against the reference SMILES file row by
row and refuses to write on a mismatch. Keep that check on.

## Setup

Needs `torch`, `numpy`, `scipy`. The `py310` conda env on this machine has all
three (torch 2.3.1+cu121, CUDA available).

Derived files go to `work/` by default; override with `$DOCKREG_WORK`. Keep the
work directory on `/N/slate`, not in the repo's git history.

## Usage

```bash
# 1. scores CSV -> float32 .npy in embedding row order (once per pool)
python -m dockreg.prepare --pool enhits

# 2. train/val/test index splits, shared by every backbone (once per pool+seed)
python -m dockreg.splits --pool enhits --seed 0

# 3. train one backbone
python -m dockreg.train --pool enhits --backbone molformer \
    --max-train 200000 --max-eval 50000 --epochs 20

# 4. late fusion across trained backbones
python -m dockreg.fuse work/enhits/runs/*__n200000__s0 --all-subsets \
    --out work/enhits/fusion.json

# 5. collect every finished run into a table (grouped by training budget,
#    baseline pinned to the top of each group)
python -m dockreg.report --pool enhits --csv work/enhits/results.csv
```

A full learning-curve sweep over every backbone and four training budgets:

```bash
bash scripts/sweep_enhits.sh          # resumable; skips finished runs
tail -f work/enhits/sweep.log         # progress
```

`--max-train 0` uses the whole training split. On `ampc`/`d4` you almost
certainly want a cap: one epoch over 98M × 3400 floats is 1.3 TB of reads.

## Design notes

**Nothing is loaded whole.** `data.gather()` pulls an index set out of a memmap,
reading contiguous slabs wherever the sorted indices are dense enough to beat
per-row reads — Lustre punishes scattered access badly.

**Splits are materialised once per backbone** into a compact `.npy` under
`work/<pool>/cache/`, so training re-reads a few GB instead of walking the full
pool each epoch. The first run of a new (backbone, split) pays the gather cost;
later runs hit the cache. Expect the first epoch to be far slower than the rest
as the page cache warms.

**Failed docks are dropped at split time**, not at training time, so train, val
and test see the same universe and the filter is recorded next to the indices.
Default is `--drop-score-above 0.0`, meaning any score at or above zero goes.
That matters: `d4` has ~5% non-negative scores (max +328), `ampc` ~0.03%,
`enhits` 16 rows.

**The test split is read once**, after the best epoch is picked on validation.
Predictions and test indices are written to each run directory so `fuse.py` can
combine backbones without retraining.

## The baseline comes first

DOCK and Glide scores scale strongly with molecular size. On `enhits`, **heavy
atom count alone correlates −0.66 with the true score**. So a backbone that
reaches Pearson 0.65 has demonstrated that it encodes molecule size, and
nothing more.

`--backbone descriptors` runs ~12 RDKit descriptors (heavy atoms, MW, logP,
TPSA, rings, rotatable bonds, H-bond donors/acceptors, fraction sp³, halogens,
aromatic rings, heteroatoms) through the identical split, model and metrics.
It costs nothing to compute and no learned embedding is worth its storage
unless it clears this line.

```bash
python -m dockreg.train --pool enhits --backbone descriptors \
    --max-train 200000 --max-eval 50000 --epochs 10
```

### Preliminary numbers — `enhits`, 200k train, 50k test

| Backbone | Dims | Storage | Pearson | Spearman | EF@1% |
|---|---:|---:|---:|---:|---:|
| **descriptors** | 12 | 0 | **0.875** | **0.862** | **35.4** |
| MHG-GED | 1024 | 8.6 GB | 0.938 | 0.933 | 49.4 |
| SMI-TED | 768 | 6.5 GB | 0.648 | 0.634 | 17.4 |
| MolFormer | 768 | 6.5 GB | 0.629 | 0.615 | 16.6 |

Read this as a smoke test, not a result. **Caveat:** the descriptor model was
trained to convergence (10 epochs, loss flat); the three embedding models got
5 epochs and were all still improving when they stopped. MolFormer and SMI-TED
need a longer run before that gap can be called real.

Two things do already look solid, and both shape what to do next:

1. **The baseline is demanding.** Any backbone landing near Pearson 0.65 is
   plausibly just measuring size. Report the descriptor row beside every
   backbone comparison.
2. **Plain mean fusion is dragged down by its worst member.** Averaging
   MHG-GED with the two weaker backbones *lowers* Pearson from 0.938 to 0.839.
   Late fusion in the multimodality literature assumes roughly comparable
   members; that assumption fails here. Use `--mode rank`, weight by validation
   score, or fuse only models that clear the baseline.

## Metrics

Lower score = better binder, so "top-k" means the k most negative.

- Regression — `rmse`, `mae`, `r2`, `pearson`, `spearman`
- Screening — `recall@f` and `ef@f` (enrichment over random) at f = 0.001,
  0.01, 0.05

These answer different questions and routinely disagree. Pearson and RMSE are
dominated by the millions of non-binders; recall at 0.1% describes the only
part of the ranking anyone would ever synthesise. `--select-on` controls which
validation metric picks the epoch (default `rmse`, since regression is the
stated goal — switch to `spearman` or `recall@0.01` if ranking is what you
care about).

## Layout

```
dockreg/
  pools.py        pool registry: paths, backbones, dims, permutation
  prepare.py      scores CSV -> float32 .npy in embedding row order
  splits.py       deterministic train/val/test indices + failed-dock filter
  data.py         memmap gather, split caching, standardisation
  descriptors.py  RDKit descriptor baseline (pseudo-backbone)
  model.py        the MLP head (fixed across backbones)
  metrics.py      regression + screening metrics
  train.py        train one (pool, backbone)
  fuse.py         late fusion across trained backbones
  report.py       collect finished runs into a table / CSV
scripts/
  sweep_enhits.sh learning-curve sweep, resumable
```

## Documentation

- [docs/LOG.md](docs/LOG.md) — research log: what was done, why, and the dead
  ends
- [docs/FINDINGS.md](docs/FINDINGS.md) — distilled findings, with the
  superseded numbers clearly marked
- [docs/LITERATURE.md](docs/LITERATURE.md) — what is already published, and
  where the remaining gap actually is
- [docs/CROSS_REPO_SYNTHESIS.md](docs/CROSS_REPO_SYNTHESIS.md) — how this
  repo's results relate to `al-eval-framework`, and the contradiction between
  them
- [docs/SOURCE_PAPER.md](docs/SOURCE_PAPER.md) — analysis of the Nature
  Communications multimodality paper this project started from, including two
  design flaws worth not repeating

## Next steps

- Train every `enhits` backbone to convergence (20+ epochs) so the comparison
  above is fair, then re-run fusion.
- Sweep `--max-train` (50k / 200k / 1M / all) per backbone — the learning
  curve, not a single point, is what tells you whether 3400 dims pay for
  themselves.
- Carry the winning setup to `ampc` and `d4`, where the storage argument
  actually bites: GROVER3400 is 6.6× the bytes per molecule of Uni-Mol v1.
- Revisit `enamine_real` once a backbone is stitched past 1000 chunks.
