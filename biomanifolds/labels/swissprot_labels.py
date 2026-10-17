"""Labels from the Swiss-Prot (UniProtKB) annotations of each protein.

Every label is a function of a `Protein` registered with `@label`. Proteins are the Swiss-Prot
entries of swissprot.parquet (length <= 1022, no fragments).

Residue-level labels come from the feature annotations (signal peptides, transmembrane
segments, binding sites, ...); protein-level labels from the taxonomy, the protein existence
level and the subcellular location.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from biomanifolds import config
from biomanifolds.datasets import swissprot
from biomanifolds.labels import label_metadata, runner

# UniProt feature type -> label name. Intervals cover start..end; sites cover single residues
# (a disulfide bond a..b covers its two cysteines). REGION is kept only for "Disordered".
INTERVAL_FEATURES = {
    "SIGNAL": "signal", "TRANSIT": "transit", "PROPEP": "propep", "CHAIN": "chain",
    "PEPTIDE": "peptide", "TRANSMEM": "transmem", "INTRAMEM": "intramem", "REGION": "disordered",
    "COILED": "coiled", "COMPBIAS": "compbias",
}
SITE_FEATURES = {"DISULFID": "disulfide", "ACT_SITE": "act_site", "BINDING": "binding", "LIPID": "lipid", "CARBOHYD": "glyco"}
# label name -> description of the in_<name> label
IN_FEATURE = {
    "signal": "In a signal peptide.",
    "transit": "In a transit peptide.",
    "propep": "In a propeptide.",
    "peptide": "In a released active peptide.",
    "transmem": "In a transmembrane segment.",
    "intramem": "In a segment inside the membrane that does not cross it.",
    "disordered": "In a disordered region.",
    "coiled": "In a coiled coil.",
    "compbias": "In a region of compositional bias.",
    "disulfide": "A cysteine in a disulfide bond.",
    "act_site": "An active-site residue.",
    "binding": "A ligand-binding residue.",
    "lipid": "A lipidated residue.",
    "glyco": "A glycosylated residue.",
}
# coarse subcellular location: the first location named in the annotation
LOCATIONS = ["Secreted", "Cell membrane", "Membrane", "Cytoplasm", "Nucleus", "Mitochondrion",
             "Plastid", "Endoplasmic reticulum", "Golgi apparatus", "Periplasm", "Cell outer membrane",
             "Virion", "Host"]

_FEATURE = re.compile(r'([A-Z_]+) ([<>?]?\d+|\?)(?:\.\.([<>?]?\d+|\?))?;?((?: /[a-z_]+="[^"]*";?)*)')
_LOCATION = re.compile(r"SUBCELLULAR LOCATION: (?:\[[^\]]*\]: )?([^.;{,]+)")


def output_dir():
    return config.data_path("swissprot_labels")


def _position(text):
    """1-based position, or None when unknown ("?")."""
    text = text.strip("<>")
    return int(text) if text.isdigit() else None


def parse_features(text):
    """UniProt feature columns -> {label name: [(start, end)]} (1-based, inclusive)."""
    features = {}
    for kind, a, b, qualifiers in _FEATURE.findall(text):
        name = INTERVAL_FEATURES.get(kind) or SITE_FEATURES.get(kind)
        if name is None or (kind == "REGION" and '/note="Disordered"' not in qualifiers):
            continue
        start, end = _position(a), _position(b) if b else _position(a)
        if start is None or end is None:
            continue
        if kind == "DISULFID" and b:  # a bond: the two cysteines, not the span
            features.setdefault(name, []).extend([(start, start), (end, end)])
        else:
            features.setdefault(name, []).append((start, end))
    return features


@dataclass
class Protein:
    accession: str
    length: int
    features: dict = field(default_factory=dict)  # label name -> [(start, end)], 1-based
    existence: str = ""
    location_text: str = ""
    kingdom: str = "other"
    group: str = "other"

    @property
    def positions(self):
        """0-based residue positions."""
        return np.arange(self.length)

    def covered(self, name):
        """Residues inside any `name` feature."""
        out = np.zeros(self.length, bool)
        for start, end in self.features.get(name, []):
            if 1 <= start <= end <= self.length:
                out[start - 1:end] = True
        return out

    @property
    def mature_chain(self):
        """(start, end) of the first mature chain (1-based), or None."""
        chains = [c for c in self.features.get("chain", []) if 1 <= c[0] <= c[1] <= self.length]
        return min(chains) if chains else None


# --- Label registry --------------------------------------------------------------------------

LABELS: dict[str, label_metadata.Label] = {}
_register = label_metadata.decorator(LABELS)


def label(kind, level="residue"):
    """Register a function `Protein -> value(s)` as a label."""
    return _register("swissprot", kind, level=level)


# --- Labels: features ------------------------------------------------------------------------


def _in_feature(name, description):
    def fn(p: Protein):
        return p.covered(name)

    fn.__name__, fn.__doc__ = f"in_{name}", description
    return label("binary")(fn)


for _name, _description in IN_FEATURE.items():
    _in_feature(_name, _description)


@label("count")
def dist_mature_n(p: Protein):
    """Residues to the N-terminal end of the first mature chain (null outside it)."""
    out = np.full(p.length, np.nan)
    if p.mature_chain:
        start, end = p.mature_chain
        out[start - 1:end] = np.arange(end - start + 1)
    return out


@label("count")
def dist_mature_c(p: Protein):
    """Residues to the C-terminal end of the first mature chain (null outside it)."""
    out = np.full(p.length, np.nan)
    if p.mature_chain:
        start, end = p.mature_chain
        out[start - 1:end] = np.arange(end - start, -1, -1)
    return out


# --- Labels: protein level -------------------------------------------------------------------


@label("categorical", level="protein")
def kingdom(p: Protein):
    """Bacteria, Archaea, Eukaryota, Viruses or other."""
    return p.kingdom


@label("categorical", level="protein")
def group(p: Protein):
    """Major taxonomic group, e.g. Pseudomonadota, Metazoa, Fungi."""
    return p.group


@label("categorical", level="protein")
def existence(p: Protein):
    """UniProt protein existence level, e.g. "Evidence at protein level"."""
    return p.existence


@label("categorical", level="protein")
def location(p: Protein):
    """Coarse subcellular location: the first location named in the annotation."""
    match = _LOCATION.search(p.location_text)
    if not match:
        return "unknown"
    first = match.group(1).strip()
    return next((loc for loc in LOCATIONS if first.startswith(loc)), "other")


# --- Runner ----------------------------------------------------------------------------------


def read_proteins():
    """Proteins of swissprot.parquet with their annotations and taxonomy."""
    annotations = pl.read_csv(swissprot.download_annotations(), separator="\t", infer_schema=False)
    feature_columns = [c for c in annotations.columns if c not in ("Entry", "Protein existence", "Subcellular location [CC]")]
    annotations = annotations.select(
        accession="Entry",
        existence="Protein existence",
        location_text=pl.col("Subcellular location [CC]").fill_null(""),
        features=pl.concat_str(feature_columns, separator=" ", ignore_nulls=True),
    )
    proteins = (
        pl.read_parquet(swissprot.swissprot_dir() / "swissprot.parquet", columns=["accession", "length"])
        .join(annotations, on="accession", how="left")
        .join(swissprot.read_taxonomy(), on="accession", how="left")
    )
    return [
        Protein(r["accession"], r["length"], parse_features(r["features"] or ""), r["existence"] or "",
                r["location_text"] or "", r["kingdom"] or "other", r["group"] or "other")
        for r in proteins.iter_rows(named=True)
    ]


def is_prepared():
    return runner.is_done(output_dir())


def prepare_data():
    runner.run(LABELS, read_proteins(), output_dir())


def load_labels(columns=None) -> pl.LazyFrame:
    """All residue-level shards as one lazy frame."""
    return pl.scan_parquet(output_dir() / "part-*.parquet").select(columns or pl.all())


def load_protein_labels() -> pl.LazyFrame:
    return pl.scan_parquet(output_dir() / "proteins.parquet")
