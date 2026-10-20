"""Cheap physicochemical descriptors, used as the baseline every backbone
must beat.

DOCK and Glide scores scale strongly with molecular size: on `enhits`, heavy
atom count alone reaches Pearson 0.66 against the true score. A learned
embedding that lands near that number has told you nothing about binding, only
that it encodes size. Treat this as a pseudo-backbone named `descriptors` and
run it through the same split, model and metrics as the real ones.

Uses RDKit when importable; otherwise falls back to a regex count of heavy
atoms and a few string-shape features, which is crude but still reproduces the
size effect.
"""

import re

import numpy as np

from .pools import Pool, pool_work

_HEAVY_RE = re.compile(r"Cl|Br|[BCNOSPFIbcnosp]")
_RING_RE = re.compile(r"[0-9]")
_HALO_RE = re.compile(r"Cl|Br|[FI]")

RDKIT_FEATURES = [
    "heavy_atoms", "mw", "logp", "tpsa", "n_rings", "n_rot",
    "n_hbd", "n_hba", "frac_csp3", "n_halogen", "n_arom_rings", "n_het",
]
FALLBACK_FEATURES = [
    "heavy_atoms", "smiles_len", "n_ring_digits", "n_halogen",
    "n_branch", "n_n", "n_o", "n_arom_lower",
]


def _have_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


def feature_names():
    return RDKIT_FEATURES if _have_rdkit() else FALLBACK_FEATURES


def _fallback_row(s: str):
    return [
        len(_HEAVY_RE.findall(s)),
        len(s),
        len(_RING_RE.findall(s)),
        len(_HALO_RE.findall(s)),
        s.count("("),
        s.count("N") + s.count("n"),
        s.count("O") + s.count("o"),
        sum(c.islower() for c in s),
    ]


def compute(smiles_list) -> np.ndarray:
    """(n, k) float32 descriptor matrix, in the order given."""
    if not _have_rdkit():
        return np.array([_fallback_row(s) for s in smiles_list],
                        dtype=np.float32)

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    out = np.zeros((len(smiles_list), len(RDKIT_FEATURES)), dtype=np.float32)
    n_bad = 0
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            n_bad += 1
            row = _fallback_row(s)
            out[i, 0] = row[0]
            continue
        out[i] = [
            mol.GetNumHeavyAtoms(),
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in mol.GetAtoms()),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            sum(a.GetSymbol() not in ("C", "H") for a in mol.GetAtoms()),
        ]
    if n_bad:
        print(f"    {n_bad:,} SMILES unparseable, fell back to atom count")
    return out


def read_smiles_at(path, idx: np.ndarray):
    """Read the lines at row positions `idx`, returned in `idx` order.

    One sequential pass -- random seeks into a 5 GB text file are far worse.
    """
    order = np.argsort(idx, kind="stable")
    wanted = np.asarray(idx)[order]
    out = [None] * len(idx)
    j = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if j >= wanted.size:
                break
            if i == wanted[j]:
                s = line.rstrip("\n")
                while j < wanted.size and wanted[j] == i:
                    out[order[j]] = s
                    j += 1
    if j < wanted.size:
        raise ValueError(
            f"{path} ended at row {j} of {wanted.size} requested "
            f"(max index {int(wanted[-1])})"
        )
    return out


def descriptor_path(pool: Pool, split_tag: str, part: str):
    d = pool_work(pool.name) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"descriptors__{split_tag}__{part}.npy"


def ensure_descriptors(
    pool: Pool, idx: np.ndarray, split_tag: str, part: str,
    rebuild: bool = False, progress: bool = True,
):
    path = descriptor_path(pool, split_tag, part)
    if path.exists() and not rebuild:
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] == idx.size:
            if progress:
                print(f"  cache hit: {path.name} ({arr.shape[0]:,} rows)",
                      flush=True)
            return arr
    if progress:
        print(f"  computing {len(feature_names())} descriptors for "
              f"{idx.size:,} molecules (rdkit={_have_rdkit()})", flush=True)
    smiles = read_smiles_at(pool.smiles_path, idx)
    X = compute(smiles)
    np.save(path, X)
    return np.load(path, mmap_mode="r")
