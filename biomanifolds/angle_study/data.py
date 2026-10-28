"""Splits and labels (prepared if missing) and their figures (notebook 1). The same for every model."""

import matplotlib.pyplot as plt
import polars as pl

from biomanifolds import plots
from biomanifolds.angle_study.figures import save
from biomanifolds.datasets import swissprot
from biomanifolds.labels import af2_structure_labels, sequence_labels, swissprot_labels


def run(out):
    figures = out / "figures"
    splits_dir = swissprot.swissprot_dir() / "splits"
    if not (splits_dir / "splits.parquet").exists():
        swissprot.prepare_data()
    for labels in [sequence_labels, swissprot_labels, af2_structure_labels]:
        if not labels.is_prepared():
            labels.prepare_data()

    # sequence identity between the test proteins and their closest train proteins
    splits = pl.read_parquet(splits_dir / "splits.parquet")
    print(splits["split"].value_counts(), flush=True)
    hits = pl.read_csv(splits_dir / "test_vs_train.m8", separator="\t", has_header=False,
                       new_columns=["query", "target", "fident", "qcov", "tcov", "evalue"])
    identity = hits.filter(pl.col("query").is_in(splits.filter(pl.col("split") == "test")["accession"].implode()))["fident"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(identity, bins=50)
    ax.set_title("train-test sequence identities")
    ax.set_xlabel("identity")
    ax.set_ylabel("count")
    fig.tight_layout()
    save(fig, figures, "data_train_test_identity")

    # proteins of Swiss-Prot and of each split
    raw = swissprot.read_fasta()
    taxonomy = swissprot.read_taxonomy()
    groups = {
        "Swiss-Prot (all)": raw.join(taxonomy, on="accession"),
        **{f"split: {name}": splits.filter(pl.col("split") == name).join(taxonomy, on="accession") for name in ["train", "test", "removed"]},
    }
    save(plots.protein_summary(groups), figures, "data_splits")
