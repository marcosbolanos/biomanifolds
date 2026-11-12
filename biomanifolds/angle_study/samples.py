"""The three samples of the study: train proteins (bins), test helix residues (test bins and ablation), steering residues."""

import hashlib

import numpy as np
import polars as pl

from biomanifolds import config
from biomanifolds.angle_study.constants import SOURCE_ANGLES, SOURCE_BIN_WIDTH
from biomanifolds.datasets import swissprot
from biomanifolds.labels import af2_structure_labels


def sequences():
    return pl.read_parquet(swissprot.swissprot_dir() / "swissprot.parquet", columns=["accession", "sequence"])


def split_proteins(split):
    """accession, sequence of the proteins of a split, sorted by accession."""
    splits = pl.read_parquet(swissprot.swissprot_dir() / "splits" / "splits.parquet", columns=["accession", "split"])
    return sequences().join(splits.filter(pl.col("split") == split), on="accession").select("accession", "sequence").sort("accession")


def in_md5_sample(accession, fraction):
    """The MD5 rule of notebook 2: the first 8 hex digits of MD5(accession), as a fraction, below `fraction`."""
    return int(hashlib.md5(accession.encode()).hexdigest()[:8], 16) / 16**8 < fraction


def train_proteins(fraction):
    """Train proteins of the MD5 sample, with all their residues: accession, sequence, positions."""
    return (
        split_proteins("train")
        .filter(pl.col("accession").map_elements(lambda a: in_md5_sample(a, fraction), return_dtype=pl.Boolean))
        .with_columns(positions=pl.int_ranges(pl.col("sequence").str.len_chars()))
    )


def test_helix_residues(test_proteins):
    """All helix residues (with a helix face) of the test proteins: accession, i, helix_face, sorted."""
    return (
        af2_structure_labels.load_labels(["accession", "i", "helix_face"])
        .filter(pl.col("helix_face").is_not_null())
        .join(test_proteins.select("accession").lazy(), on="accession")
        .collect(engine="streaming")
        .sort("accession", "i")  # fixed order, so that the sample is reproducible
    )


def test_residues(test_proteins, helix_residues, n, seed):
    """n random test helix residues (notebook 3): accession, i, helix_face, sequence, sorted by protein."""
    return (
        helix_residues.sample(min(n, len(helix_residues)), seed=seed)
        .sort("accession", "i")
        .join(test_proteins, on="accession", maintain_order="left")
    )


def residue_proteins(residues):
    """The proteins of a residue sample, with the sampled positions: accession, sequence, positions."""
    return (
        residues.group_by("accession").agg(positions=pl.col("i").sort(), sequence=pl.col("sequence").first())
        .select("accession", "sequence", "positions").sort("accession")
    )


def steering_residues(test_proteins, helix_residues, n_per_bin, seed):
    """Per source bin (5 degrees): one random residue per protein, n_per_bin random proteins (notebook 5).

    Returns the residues (accession, i, helix_face, source_angle, sequence) and the counts per bin."""
    samples, counts = [], []
    for source in SOURCE_ANGLES:
        # angular distance below half a bin, with the wrap-around at +-180 degrees
        candidates = helix_residues.filter((pl.col("helix_face") - source).radians().cos() > np.cos(np.deg2rad(SOURCE_BIN_WIDTH / 2)))
        chosen = (
            candidates.sample(fraction=1.0, shuffle=True, seed=seed)  # random order
            .unique("accession", keep="first", maintain_order=True)  # one random residue per protein
            .head(n_per_bin)  # random proteins
            .with_columns(source_angle=pl.lit(source))
        )
        samples.append(chosen)
        counts.append({"source_angle": source, "helix_residues_in_bin": len(candidates), "proteins_in_bin": candidates["accession"].n_unique(),
                       "chosen_residues": len(chosen), "chosen_proteins": chosen["accession"].n_unique()})
    residues = pl.concat(samples).join(test_proteins, on="accession", maintain_order="left")
    return residues, pl.DataFrame(counts)


def train_helix_residues(fraction=0.10):
    """The helix residues (with a helix face) of the train proteins of the MD5 sample: accession, i, helix_face, row; sorted by row.

    row: the row of the residue in the train embedding files of notebook 2 (all the residues of the train proteins, in order)."""
    rows = train_proteins(fraction).select("accession", i="positions").explode("i").with_row_index("row")
    return (
        af2_structure_labels.load_labels(["accession", "i", "helix_face"])
        .filter(pl.col("helix_face").is_not_null())
        .join(rows.lazy(), on=["accession", "i"])  # the label table is streamed on the left, only the rows are held
        .collect(engine="streaming")
        .sort("row")
    )


def helix_proteins(helix_residues, proteins):
    """The proteins of `helix_residues` with their helix positions, in the row order of the residues: accession, sequence, positions."""
    return (
        helix_residues.group_by("accession", maintain_order=True).agg(positions=pl.col("i"))
        .join(proteins.select("accession", "sequence"), on="accession", maintain_order="left")
        .select("accession", "sequence", "positions")
    )


def train_helix_embeddings(layer, train_helix):
    """h (residues, hidden), float16, of the train helix residues at a layer, in the row order of `train_helix`.

    From the file of notebook 2 (all the residues of the train proteins) if there is one: the helix rows, in memory.
    If not, from the file with the helix residues only (helix_train_..., made with embed_sequences on helix_proteins): a memory map."""
    directory = config.data_path("esm2_embeddings")
    full = directory / f"sampled_train_residue_embeddings_L{layer}.npy"
    if full.exists():
        return np.load(full, mmap_mode="r")[train_helix["row"].to_numpy()]
    h = np.load(directory / f"helix_train_residue_embeddings_L{layer}.npy", mmap_mode="r")
    assert len(h) == len(train_helix), f"the layer-{layer} file does not have the train helix residues"
    return h
