import os
import subprocess
from pathlib import Path

import polars as pl
import pymmseqs

MMSEQS = Path(pymmseqs.__file__).parent / "bin" / "mmseqs"


def _mmseqs(*args):
    subprocess.run([str(MMSEQS), *map(str, args)], check=True)


def _write_fasta(df, path):
    with open(path, "w") as f:
        for acc, seq in df.select("accession", "sequence").iter_rows():
            f.write(f">{acc}\n{seq}\n")


def make_splits(
    df,
    out_dir,
    min_seq_id=0.3,
    coverage=0.8,
    train_frac=0.75,
    sensitivity=7.5,
    seed=0,
    threads=os.cpu_count(),
    memory_limit="10G",
):
    """Split `df` (columns: accession, sequence) into train/test with at most
    `min_seq_id` identity between splits, as measured by MMseqs2.

    1. Cluster all sequences at `min_seq_id` / `coverage` and assign whole
       clusters to train until `train_frac` of the sequences are in train.
    2. Search test against train and mark every test sequence with a hit at
       `min_seq_id` or more (coverage of both sequences >= `coverage`) as "removed".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    common = ["-c", coverage, "--cov-mode", 0, "--threads", threads, "--split-memory-limit", memory_limit]

    # 1. Cluster, then assign whole clusters to train or test
    _write_fasta(df, out_dir / "all.fasta")
    _mmseqs("easy-cluster", out_dir / "all.fasta", out_dir / "clu", tmp, *common, "-s", sensitivity,
            "--min-seq-id", min_seq_id)
    clusters = pl.read_csv(
        out_dir / "clu_cluster.tsv", separator="\t", has_header=False, new_columns=["cluster", "accession"]
    )

    sizes = clusters.group_by("cluster").len().sample(fraction=1.0, shuffle=True, seed=seed)
    train_clusters = sizes.filter(pl.col("len").cum_sum() <= train_frac * len(clusters))["cluster"]
    clusters = clusters.with_columns(
        split=pl.when(pl.col("cluster").is_in(train_clusters.implode())).then(pl.lit("train")).otherwise(pl.lit("test"))
    )
    df = df.join(clusters, on="accession")

    # 2. Homology filter: mark test sequences with a hit in train as removed.
    #    No identity threshold in the search, so test_vs_train.m8 keeps all
    #    train-test identities for verification.
    _write_fasta(df.filter(pl.col("split") == "train"), out_dir / "train.fasta")
    _write_fasta(df.filter(pl.col("split") == "test"), out_dir / "test.fasta")
    _mmseqs(
        "easy-search", out_dir / "test.fasta", out_dir / "train.fasta", out_dir / "test_vs_train.m8", tmp,
        *common, "-s", sensitivity, "-e", 10, "--max-seqs", 1000,
        "--format-output", "query,target,fident,qcov,tcov,evalue",
    )
    hits = pl.read_csv(
        out_dir / "test_vs_train.m8", separator="\t", has_header=False,
        schema={"query": pl.String, "target": pl.String, "fident": pl.Float64,
                "qcov": pl.Float64, "tcov": pl.Float64, "evalue": pl.Float64},
    )
    leaked = hits.filter(pl.col("fident") >= min_seq_id)["query"].unique()
    df = df.with_columns(
        split=pl.when(pl.col("accession").is_in(leaked.implode())).then(pl.lit("removed")).otherwise("split")
    )

    df.write_parquet(out_dir / "splits.parquet")
    print(df.group_by("split").len())
    return df
