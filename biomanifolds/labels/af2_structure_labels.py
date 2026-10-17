"""Per-residue labels from the AlphaFold DB models of Swiss-Prot proteins.

Every label is a function of a `Structure` registered with `@label`. Two sources:
  * "afdb": read from the model as it is (pLDDT);
  * "computed": calculated from the model coordinates.

AFDB models are full length and continuous, and residue i of the model is residue i of the
UniProt sequence, so there are no gaps or numbering problems to handle. Models whose sequence
differs from the current UniProt sequence are skipped.

pLDDT masks (set per label):
  * "residue": NaN where pLDDT < PLDDT_MIN;
  * "element": also NaN for every residue of a secondary-structure element (a run of one ss3
    state) that contains a residue with pLDDT < PLDDT_MIN, because its extent is not reliable;
  * None: never masked.

Angles are in degrees in (-180, 180]. Undefined values are NaN.
"""

import gzip
import io
import multiprocessing
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property

import biotite.structure as struc
import numpy as np
import polars as pl
from biotite.structure.io import pdb
from tqdm import tqdm

from biomanifolds import config
from biomanifolds.datasets import alphafold, swissprot
from biomanifolds.labels import label_metadata, runner
from biomanifolds.labels._protor_radii import PROTOR_RADII

PLDDT_MIN = 70.0
CONTACT_CUTOFF = 8.0  # virtual-CB distance (A)
MIN_CONTACT_SEP = 3  # contacts need |i - j| >= 3
NONLOCAL_SEP = 6  # "non-local" contacts need |i - j| >= 6
LONG_RANGE = 24  # "long-range" contacts need |i - j| >= 24
HSE_RADIUS = 13.0  # CA distance for half-sphere exposure (A)
HELIX_WINDOW = 3  # half-width of the CA window used to fit a local helix axis
STRAND_PAIR_CUTOFF = 5.5  # CA distance between paired strand residues (A)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
# Theoretical maximum accessible surface area, Tien et al. 2013 (A^2)
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225, "G": 104,
    "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240, "P": 159, "S": 155,
    "T": 172, "W": 285, "Y": 263, "V": 174,
}
# biotite P-SEA states -> DSSP-like 3-state code
SSE_TO_SS3 = {"a": "H", "b": "E", "c": "-"}
_MODEL_NAME = re.compile(r"AF-([A-Z0-9]+)-F1-model_v\d+\.pdb\.gz$")


def output_dir():
    return config.data_path("af2_structure_labels")


# --- Geometry helpers ------------------------------------------------------------------------


def _angle(p0, p1, p2):
    """Angle p0-p1-p2 (degrees) for arrays of points of shape (n, 3)."""
    a, b = p0 - p1, p2 - p1
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _dihedral(p0, p1, p2, p3):
    """Dihedral angle p0-p1-p2-p3 (degrees) for arrays of points of shape (n, 3)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def _signed_angle(u, v, axis):
    """Angle from u to v around axis (degrees)."""
    return np.degrees(np.arctan2(np.dot(np.cross(u, v), axis), np.dot(u, v)))


# --- Structure -------------------------------------------------------------------------------


class Structure:
    """One AFDB model. Quantities that several labels share are cached properties, so they
    are calculated once, and only when a label needs them."""

    def __init__(self, atoms):
        self.atoms = atoms
        starts = struc.get_residue_starts(atoms)
        self.seq = "".join(THREE_TO_ONE.get(r, "X") for r in atoms.res_name[starts])
        self.n = len(self.seq)

    @classmethod
    def from_pdb_gz(cls, data: bytes):
        f = pdb.PDBFile.read(io.StringIO(gzip.decompress(data).decode()))
        atoms = f.get_structure(model=1, extra_fields=["b_factor"])
        # standard amino acids only; struc.filter_amino_acids would load biotite's chemical
        # component dictionary (~0.5 GB per process)
        return cls(atoms[(atoms.element != "H") & np.isin(atoms.res_name, list(THREE_TO_ONE))])

    def _backbone(self, name):
        coords = self.atoms.coord[self.atoms.atom_name == name]
        assert len(coords) == self.n, f"{name}: {len(coords)} atoms for {self.n} residues"
        return coords.astype(float)

    # Atoms

    @cached_property
    def N(self):
        return self._backbone("N")

    @cached_property
    def CA(self):
        return self._backbone("CA")

    @cached_property
    def C(self):
        return self._backbone("C")

    @cached_property
    def CB(self):
        """Virtual CB (ideal geometry), also defined for glycine."""
        b, c = self.CA - self.N, self.C - self.CA
        return -0.58273431 * np.cross(b, c) + 0.56802827 * b - 0.54067466 * c + self.CA

    @cached_property
    def centroid(self):
        return self.CA.mean(0)

    # Pairwise quantities

    @cached_property
    def separation(self):
        """|i - j| for every residue pair."""
        idx = np.arange(self.n)
        return np.abs(idx[:, None] - idx[None])

    @cached_property
    def ca_distances(self):
        return np.linalg.norm(self.CA[:, None] - self.CA[None], axis=-1)

    @cached_property
    def contacts(self):
        """Virtual-CB distance < CONTACT_CUTOFF and |i - j| >= MIN_CONTACT_SEP."""
        cb_distances = np.linalg.norm(self.CB[:, None] - self.CB[None], axis=-1)
        return (cb_distances < CONTACT_CUTOFF) & (self.separation >= MIN_CONTACT_SEP)

    # Per-residue quantities

    @cached_property
    def plddt(self):
        return self.atoms.b_factor[self.atoms.atom_name == "CA"].astype(float)

    @cached_property
    def ss3(self):
        """Secondary structure: H (helix), E (strand), - (other), from biotite P-SEA."""
        return np.array([SSE_TO_SS3[s] for s in struc.annotate_sse(self.atoms)])

    @cached_property
    def elements(self):
        """Runs of one ss3 state: list of (state, start, end_exclusive)."""
        change = np.flatnonzero(self.ss3[1:] != self.ss3[:-1]) + 1
        bounds = np.r_[0, change, self.n]
        return [(self.ss3[s], s, e) for s, e in zip(bounds[:-1], bounds[1:])]

    @cached_property
    def sasa(self):
        """Solvent accessible surface area of each residue in the isolated chain (A^2)."""
        radii = np.array([PROTOR_RADII[r][a] for r, a in zip(self.atoms.res_name, self.atoms.atom_name)])
        atom_sasa = struc.sasa(self.atoms, point_number=200, vdw_radii=radii, ignore_ions=False)
        return struc.apply_residue_wise(self.atoms, atom_sasa, np.nansum)

    @cached_property
    def backbone_dihedrals(self):
        """(phi, psi, omega) in degrees; NaN at the termini."""
        return tuple(np.degrees(a) for a in struc.dihedral_backbone(self.atoms))


# --- Label registry --------------------------------------------------------------------------

LABELS: dict[str, label_metadata.Label] = {}
_register = label_metadata.decorator(LABELS)


def label(kind, source="computed", mask="residue"):
    """Register a function `Structure -> np.ndarray` (one value per residue) as a label."""
    return _register(source, kind, mask=mask)


# --- Labels: read from the model -------------------------------------------------------------


@label("continuous", source="afdb", mask=None)
def plddt(s: Structure):
    """AlphaFold per-residue confidence (0-100)."""
    return s.plddt


@label("binary", source="afdb", mask=None)
def confident(s: Structure):
    """pLDDT >= PLDDT_MIN."""
    return s.plddt >= PLDDT_MIN


# --- Labels: secondary structure -------------------------------------------------------------


@label("categorical")
def ss3(s: Structure):
    """Secondary structure: H, E or -."""
    return s.ss3


@label("count", mask="element")
def seg_len(s: Structure):
    """Length of the secondary-structure element that contains the residue."""
    out = np.empty(s.n)
    for _, start, end in s.elements:
        out[start:end] = end - start
    return out


@label("count", mask="element")
def pos_in_seg(s: Structure):
    """0-based position of the residue in its secondary-structure element."""
    out = np.empty(s.n)
    for _, start, end in s.elements:
        out[start:end] = np.arange(end - start)
    return out


@label("continuous", mask="element")
def seg_frac(s: Structure):
    """Relative position in the element: 0 = first residue, 1 = last, 0.5 for one residue."""
    length, pos = seg_len(s), pos_in_seg(s)
    return np.where(length > 1, pos / np.maximum(length - 1, 1), 0.5)


@label("binary", mask="element")
def n_cap(s: Structure):
    """The residue just before a helix."""
    out = np.zeros(s.n, bool)
    for state, start, _ in s.elements:
        if state == "H" and start > 0:
            out[start - 1] = True
    return out


@label("binary", mask="element")
def c_cap(s: Structure):
    """The residue just after a helix."""
    out = np.zeros(s.n, bool)
    for state, _, end in s.elements:
        if state == "H" and end < s.n:
            out[end] = True
    return out


@label("angle", mask="element")
def helix_face(s: Structure):
    """Angle of the residue around the local helix axis; 0 = pointing toward the protein
    centroid (the buried face). Only for helices of at least 2 * HELIX_WINDOW + 1 residues."""
    out = np.full(s.n, np.nan)
    width = 2 * HELIX_WINDOW + 1
    for state, start, end in s.elements:
        if state != "H" or end - start < width:
            continue
        direction = s.CA[end - 1] - s.CA[start]
        for i in range(start, end):
            lo = min(max(start, i - HELIX_WINDOW), end - width)
            window = s.CA[lo:lo + width]
            mid = window.mean(0)
            axis = np.linalg.svd(window - mid)[2][0]
            axis = axis if np.dot(axis, direction) > 0 else -axis
            radial = s.CA[i] - mid
            radial -= np.dot(radial, axis) * axis
            inward = s.centroid - mid
            inward -= np.dot(inward, axis) * axis
            out[i] = _signed_angle(inward, radial, axis)
    return out


def _strand_partners(s: Structure):
    """For each E residue, its paired E residues (CA < STRAND_PAIR_CUTOFF, |i - j| >= 3),
    closest first, at most one per neighbouring strand."""
    is_e = s.ss3 == "E"
    pairs = is_e[:, None] & is_e[None] & (s.ca_distances < STRAND_PAIR_CUTOFF) & (s.separation >= 3)
    partners = {}
    for i in np.flatnonzero(is_e):
        kept = []
        for j in sorted(np.flatnonzero(pairs[i]), key=lambda j: s.ca_distances[i, j]):
            if all(abs(j - k) > 2 for k in kept):
                kept.append(j)
        partners[i] = kept
    return partners


@label("count")
def strand_partners(s: Structure):
    """Number of paired strand residues (0, 1 or 2)."""
    out = np.zeros(s.n)
    for i, kept in _strand_partners(s).items():
        out[i] = min(len(kept), 2)
    return out


@label("binary")
def strand_parallel(s: Structure):
    """1 if the residue's closest strand partner runs parallel, 0 if antiparallel."""
    out = np.full(s.n, np.nan)
    for i, kept in _strand_partners(s).items():
        if not kept:
            continue
        j = kept[0]
        if i + 1 < s.n and 0 < j < s.n - 1:
            same = np.linalg.norm(s.CA[i + 1] - s.CA[j + 1])
            opposite = np.linalg.norm(s.CA[i + 1] - s.CA[j - 1])
            out[i] = float(same < opposite)
    return out


# --- Labels: backbone angles -----------------------------------------------------------------


@label("angle")
def phi(s: Structure):
    """Backbone dihedral C(i-1)-N-CA-C."""
    return s.backbone_dihedrals[0]


@label("angle")
def psi(s: Structure):
    """Backbone dihedral N-CA-C-N(i+1)."""
    return s.backbone_dihedrals[1]


@label("angle")
def omega(s: Structure):
    """Backbone dihedral CA-C-N(i+1)-CA(i+1)."""
    return s.backbone_dihedrals[2]


@label("continuous")
def theta(s: Structure):
    """CA virtual bond angle CA(i-1)-CA(i)-CA(i+1)."""
    out = np.full(s.n, np.nan)
    out[1:-1] = _angle(s.CA[:-2], s.CA[1:-1], s.CA[2:])
    return out


@label("angle")
def tau(s: Structure):
    """CA virtual dihedral CA(i-1)-CA(i)-CA(i+1)-CA(i+2)."""
    out = np.full(s.n, np.nan)
    out[1:-2] = _dihedral(s.CA[:-3], s.CA[1:-2], s.CA[2:-1], s.CA[3:])
    return out


# --- Labels: burial and contacts -------------------------------------------------------------


@label("continuous")
def sasa(s: Structure):
    """Solvent accessible surface area (A^2)."""
    return s.sasa


@label("continuous")
def rsa(s: Structure):
    """Relative solvent accessibility: SASA / maximum ASA of the amino acid."""
    return s.sasa / np.array([MAX_ASA.get(a, np.nan) for a in s.seq])


@label("count")
def contact_number(s: Structure):
    """Number of contacts."""
    return s.contacts.sum(1).astype(float)


@label("count")
def long_range_contacts(s: Structure):
    """Number of contacts with |i - j| >= LONG_RANGE."""
    return (s.contacts & (s.separation >= LONG_RANGE)).sum(1).astype(float)


@label("count")
def nearest_nonlocal_sep(s: Structure):
    """Smallest |i - j| of the contacts with |i - j| >= NONLOCAL_SEP."""
    nonlocal_sep = np.where(s.contacts & (s.separation >= NONLOCAL_SEP), s.separation, np.inf).min(1)
    return np.where(np.isinf(nonlocal_sep), np.nan, nonlocal_sep)


@label("continuous")
def mean_contact_sep(s: Structure):
    """Mean |i - j| of the contacts."""
    count = s.contacts.sum(1)
    total = np.where(s.contacts, s.separation, 0).sum(1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _half_sphere(s: Structure):
    """For each residue, which CA atoms are within HSE_RADIUS (near) and on the side its CB
    points to (up)."""
    to_other = s.CA[None] - s.CA[:, None]  # (i, j, 3): vector CA(i) -> CA(j)
    up = (to_other * (s.CB - s.CA)[:, None]).sum(-1) > 0
    near = (s.ca_distances < HSE_RADIUS) & (s.ca_distances > 0)
    return near, up


@label("count")
def hse_up(s: Structure):
    """Half-sphere exposure: CA atoms within HSE_RADIUS on the CB side."""
    near, up = _half_sphere(s)
    return (near & up).sum(1).astype(float)


@label("count")
def hse_down(s: Structure):
    """Half-sphere exposure: CA atoms within HSE_RADIUS on the side opposite to CB."""
    near, up = _half_sphere(s)
    return (near & ~up).sum(1).astype(float)


@label("continuous")
def depth(s: Structure):
    """Distance of CA to the protein centroid, divided by the radius of gyration."""
    distance = np.linalg.norm(s.CA - s.centroid, axis=1)
    return distance / np.sqrt((distance**2).mean())


# --- Runner ----------------------------------------------------------------------------------


def _mask(s: Structure, values, mask):
    if mask is None:
        return values
    low = s.plddt < PLDDT_MIN
    if mask == "element":
        for _, start, end in s.elements:
            if low[start:end].any():
                low[start:end] = True
    values = values.astype(object if values.dtype.kind in "bUO" else float)
    values[low] = None if values.dtype == object else np.nan
    return values


def compute_labels(s: Structure) -> tuple[dict[str, np.ndarray], str]:
    """Run every registered label on one structure. Returns (labels, status); a label that
    fails is left out and named in the status."""
    out, failed = {}, []
    for lab in LABELS.values():
        try:
            out[lab.name] = _mask(s, np.asarray(lab.fn(s)), lab.mask)
        except Exception as exc:
            failed.append(f"{lab.name}: {type(exc).__name__}: {exc}")
    return out, "; ".join(failed)[:500] or "ok"


SCHEMA = {"accession": pl.String, "i": pl.Int32, **{name: lab.dtype for name, lab in LABELS.items()}}


def _one(args):
    """Labels of one protein, or None, and its status."""
    accession, sequence, data = args
    try:
        s = Structure.from_pdb_gz(data)
        if s.seq != sequence:
            return None, (accession, "sequence differs from UniProt")
        labels, status = compute_labels(s)
        columns = {k: v.tolist() if v.dtype == object else v for k, v in labels.items()}
        df = pl.DataFrame({"accession": [accession] * s.n, "i": np.arange(s.n), **columns}, nan_to_null=True)
        df = df.with_columns(pl.lit(None, SCHEMA[c]).alias(c) for c in SCHEMA if c not in df.columns)
        return df.select(pl.col(c).cast(t) for c, t in SCHEMA.items()), (accession, status)
    except Exception as exc:
        return None, (accession, f"failed: {type(exc).__name__}: {exc}"[:500])


def _models(tar_path, sequences):
    """(accession, sequence, pdb_gz) for every Swiss-Prot protein in the AFDB tar, in tar order."""
    with tarfile.open(tar_path, mode="r|") as tar:
        for member in tar:
            match = _MODEL_NAME.search(member.name)
            if match and match.group(1) in sequences:
                yield match.group(1), sequences[match.group(1)], tar.extractfile(member).read()


def is_prepared():
    return runner.is_done(output_dir())


def prepare_data(workers=os.cpu_count()):
    """Labels of every Swiss-Prot protein (swissprot.parquet) that has an AFDB model.

    Writes shards of runner.SHARD proteins, in tar order, to part-NNNN.parquet and the status of
    each protein to status-NNNN.parquet (proteins without a model: status-missing.parquet).
    Finished shards are skipped on a restart; _SUCCESS is written last.
    """
    out_dir = output_dir()
    proteins = pl.read_parquet(swissprot.swissprot_dir() / "swissprot.parquet", columns=["accession", "sequence"])
    sequences = dict(proteins.iter_rows())
    seen = set()

    # forkserver: workers start clean instead of copying this process's memory
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        bar = tqdm(total=len(sequences), unit=" proteins", desc="AF2 structure labels")
        for k, batch in enumerate(runner.batches(_models(alphafold.download_data(), sequences), runner.SHARD)):
            seen.update(accession for accession, _, _ in batch)
            part, status = out_dir / f"part-{k:04d}.parquet", out_dir / f"status-{k:04d}.parquet"
            if not status.exists():
                results = list(pool.map(_one, batch, chunksize=16))
                frames = [df for df, _ in results if df is not None]
                pl.concat(frames).write_parquet(part)
                pl.DataFrame([st for _, st in results], schema=["accession", "status"], orient="row").write_parquet(status)
            bar.update(len(batch))
        bar.close()

    missing = sorted(set(sequences) - seen)
    pl.DataFrame({"accession": missing, "status": "no AFDB model"}).write_parquet(out_dir / "status-missing.parquet")
    runner.mark_done(out_dir)


def load_labels(columns=None) -> pl.LazyFrame:
    """All label shards as one lazy frame."""
    return pl.scan_parquet(output_dir() / "part-*.parquet").select(columns or pl.all())
