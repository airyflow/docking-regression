"""Registry of docking pools and their on-disk layout.

Every pool is a single receptor, so there is no protein input: a row is one
molecule, and the label is its docking score (more negative = better binder).

Row alignment, verified on disk 2026-10-16:

  ampc, d4     dedup/<pool>_smiles.txt, dedup/<pool>_scores.csv.gz and every
               dedup/*_embeddings.npy share one row order.

  enhits       embed/enhits_large_smiles.txt is row-aligned with
               al-molecular/data/EnamineHTS_scores.csv.gz, but the embeddings
               live in embed_shuffled/, where row i is original row perm[i].
               `perm_path` records that mapping; prepare.py applies it so the
               score array always matches the embedding row order.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("/N/project/SingleCell_Image/mengjing")
AL_MOLECULAR = Path("/N/slate/mengjing/repos/al-molecular")

# Default location for everything this repo derives (score arrays, splits,
# cached embedding subsets, runs). Override with $DOCKREG_WORK.
DEFAULT_WORK = Path("/N/slate/mengjing/repos/docking-regression/work")


@dataclass(frozen=True)
class Backbone:
    name: str
    filename: str
    dim: int


@dataclass(frozen=True)
class Pool:
    name: str
    target: str
    oracle: str
    n_rows: int
    embed_dir: Path
    smiles_path: Path
    scores_csv: Path
    backbones: dict
    # If set, embedding row i corresponds to scores_csv row perm[i].
    perm_path: Path = None
    # SMILES file that is row-aligned with scores_csv (may differ from the one
    # beside the embeddings when a permutation is in play).
    scores_smiles_path: Path = None

    def embedding_path(self, backbone: str) -> Path:
        if backbone not in self.backbones:
            raise KeyError(
                f"pool {self.name!r} has no backbone {backbone!r}; "
                f"available: {sorted(self.backbones)}"
            )
        return self.embed_dir / self.backbones[backbone].filename

    def dim(self, backbone: str) -> int:
        return self.backbones[backbone].dim


def _bb(*specs) -> dict:
    return {s.name: s for s in specs}


_ENHITS_EMBED = DATA_ROOT / "enhits_large" / "embed_shuffled"
_AMPC = DATA_ROOT / "ampc_99.5M" / "dedup"
_D4 = DATA_ROOT / "d4_138M" / "dedup"

POOLS = {
    "enhits": Pool(
        name="enhits",
        target="EnamineHTS",
        oracle="DOCK3.7",
        n_rows=2_104_318,
        embed_dir=_ENHITS_EMBED,
        smiles_path=_ENHITS_EMBED / "enhits_large_smiles.txt",
        scores_csv=AL_MOLECULAR / "data" / "EnamineHTS_scores.csv.gz",
        scores_smiles_path=(
            DATA_ROOT / "enhits_large" / "embed" / "enhits_large_smiles.txt"
        ),
        perm_path=_ENHITS_EMBED / "shuffle_permutation.npy",
        backbones=_bb(
            # NB: file is grover_embeddings.npy here, grover3400_embeddings.npy
            # on the other pools. Same 3400 dims.
            Backbone("grover3400", "grover_embeddings.npy", 3400),
            Backbone("mhgged", "mhgged_embeddings.npy", 1024),
            Backbone("molformer", "molformer_embeddings.npy", 768),
            Backbone("smited", "smited_embeddings.npy", 768),
            # NB: 768 dims here, 1536 on ampc/d4 -- a different extraction.
            # Do not compare unimol2 numbers across pools.
            Backbone("unimol2", "unimol2_embeddings.npy", 768),
            # Matched 1536-d extraction of the same pool, joined by SMILES
            # from al-molecular/results/embed/EnamineHTS. Its cache is built
            # by scripts/unimol2_1536_screen.py; the path below is never read
            # directly because the source file has a different row order.
            Backbone("unimol2_1536", "unimol2_embeddings.npy", 1536),
        ),
    ),
    "ampc": Pool(
        name="ampc",
        target="AmpC beta-lactamase",
        oracle="DOCK3.7",
        n_rows=98_489_350,
        embed_dir=_AMPC,
        smiles_path=_AMPC / "ampc_smiles.txt",
        scores_csv=_AMPC / "ampc_scores.csv.gz",
        backbones=_bb(
            Backbone("grover3400", "grover3400_embeddings.npy", 3400),
            Backbone("mhgged", "mhgged_embeddings.npy", 1024),
            Backbone("molformer", "molformer_embeddings.npy", 768),
            Backbone("smited", "smited_embeddings.npy", 768),
            Backbone("unimol2", "unimol2_embeddings.npy", 1536),
            Backbone("unimol", "unimol_embeddings.npy", 512),
        ),
    ),
    "d4": Pool(
        name="d4",
        target="D4 dopamine receptor",
        oracle="DOCK3.7",
        n_rows=116_225_927,
        embed_dir=_D4,
        smiles_path=_D4 / "d4_smiles.txt",
        scores_csv=_D4 / "d4_scores.csv.gz",
        backbones=_bb(
            Backbone("grover3400", "grover3400_embeddings.npy", 3400),
            Backbone("mhgged", "mhgged_embeddings.npy", 1024),
            Backbone("molformer", "molformer_embeddings.npy", 768),
            Backbone("smited", "smited_embeddings.npy", 768),
            Backbone("unimol2", "unimol2_embeddings.npy", 1536),
            Backbone("unimol", "unimol_embeddings.npy", 512),
        ),
    ),
}


def get_pool(name: str) -> Pool:
    if name not in POOLS:
        raise KeyError(f"unknown pool {name!r}; available: {sorted(POOLS)}")
    return POOLS[name]


def work_dir() -> Path:
    import os

    return Path(os.environ.get("DOCKREG_WORK", DEFAULT_WORK))


def pool_work(pool: str) -> Path:
    d = work_dir() / pool
    d.mkdir(parents=True, exist_ok=True)
    return d
