"""Labels that depend only on the protein sequence.

Every label is a function of a `Protein` registered with `@label`. Proteins are the Swiss-Prot
entries of swissprot.parquet (length <= 1022, no fragments).
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from biomanifolds import config
from biomanifolds.datasets import swissprot
from biomanifolds.labels import label_metadata, runner


def output_dir():
    return config.data_path("sequence_labels")


@dataclass
class Protein:
    accession: str
    sequence: str

    @property
    def length(self):
        return len(self.sequence)

    @property
    def positions(self):
        """0-based residue positions."""
        return np.arange(self.length)


# --- Label registry --------------------------------------------------------------------------

LABELS: dict[str, label_metadata.Label] = {}
_register = label_metadata.decorator(LABELS)


def label(kind, level="residue"):
    """Register a function `Protein -> value(s)` as a label."""
    return _register("sequence", kind, level=level)


# --- Labels ----------------------------------------------------------------------------------


@label("categorical")
def aa(p: Protein):
    """Amino acid (one-letter code)."""
    return np.array(list(p.sequence))


@label("continuous")
def rel_pos(p: Protein):
    """Relative position: 0 = N terminus, 1 = C terminus."""
    return p.positions / max(p.length - 1, 1)


@label("count")
def dist_n(p: Protein):
    """Residues to the N terminus."""
    return p.positions


@label("count")
def dist_c(p: Protein):
    """Residues to the C terminus."""
    return p.length - 1 - p.positions


@label("count", level="protein")
def length(p: Protein):
    """Number of residues."""
    return p.length


# --- Runner ----------------------------------------------------------------------------------


def is_prepared():
    return runner.is_done(output_dir())


def prepare_data():
    proteins = pl.read_parquet(swissprot.swissprot_dir() / "swissprot.parquet", columns=["accession", "sequence"])
    runner.run(LABELS, [Protein(*row) for row in proteins.iter_rows()], output_dir())


def load_labels(columns=None) -> pl.LazyFrame:
    """All residue-level shards as one lazy frame."""
    return pl.scan_parquet(output_dir() / "part-*.parquet").select(columns or pl.all())


def load_protein_labels() -> pl.LazyFrame:
    return pl.scan_parquet(output_dir() / "proteins.parquet")
