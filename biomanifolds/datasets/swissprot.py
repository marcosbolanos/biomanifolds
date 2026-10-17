import gzip
import tomllib
import urllib.request
from pathlib import Path

import polars as pl
from tqdm import tqdm

from biomanifolds.datasets.clustering import make_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWISSPROT_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
TAXONOMY_URL = "https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,lineage_ids&query=reviewed:true&compressed=true"
ANNOTATION_FIELDS = (
    "accession,protein_existence,ft_signal,ft_transit,ft_propep,ft_chain,ft_peptide,ft_transmem,"
    "ft_intramem,ft_region,ft_coiled,ft_compbias,ft_disulfid,ft_act_site,ft_binding,ft_lipid,"
    "ft_carbohyd,cc_subcellular_location"
)
ANNOTATIONS_URL = f"https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields={ANNOTATION_FIELDS}&query=reviewed:true&compressed=true"

# NCBI taxon id -> name, checked in order against each protein's lineage
KINGDOMS = {2: "Bacteria", 2157: "Archaea", 2759: "Eukaryota", 10239: "Viruses"}
GROUPS = {
    1224: "Pseudomonadota", 1239: "Bacillota", 201174: "Actinomycetota", 2: "other Bacteria",
    2157: "Archaea",
    33208: "Metazoa", 4751: "Fungi", 33090: "Viridiplantae", 2759: "other Eukaryota",
    10239: "Viruses",
}


def swissprot_dir():
    with open(PROJECT_ROOT / "project-config.toml", "rb") as f:
        config = tomllib.load(f)

    path = PROJECT_ROOT / config["paths"]["swissprot"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url, path, desc):
    part_path = path.with_name(path.name + ".part")  # renamed only when complete
    with tqdm(unit="B", unit_scale=True, desc=desc) as bar:

        def hook(blocks, block_size, total_size):
            bar.total = total_size if total_size > 0 else None
            done = blocks * block_size
            if total_size > 0:
                done = min(done, total_size)
            bar.update(done - bar.n)

        urllib.request.urlretrieve(url, part_path, reporthook=hook)
    part_path.rename(path)


def download_data():
    fasta_path = swissprot_dir() / "uniprot_sprot.fasta.gz"
    if not fasta_path.exists():
        _download(SWISSPROT_URL, fasta_path, "Downloading SwissProt")
    return fasta_path


def download_taxonomy():
    tsv_path = swissprot_dir() / "uniprot_sprot_lineage.tsv.gz"
    if not tsv_path.exists():
        _download(TAXONOMY_URL, tsv_path, "Downloading SwissProt taxonomy")
    return tsv_path


def download_annotations():
    """UniProt feature annotations, existence and subcellular location of every entry."""
    tsv_path = swissprot_dir() / "uniprot_sprot_annotations.tsv.gz"
    if not tsv_path.exists():
        _download(ANNOTATIONS_URL, tsv_path, "Downloading SwissProt annotations")
    return tsv_path


def read_fasta():
    """All SwissProt entries, unfiltered."""
    fasta_path = download_data()

    rows, acc, desc, seq = [], None, None, []
    with gzip.open(fasta_path, "rt") as fh:
        for line in tqdm(fh, unit=" lines", desc="Parsing FASTA"):
            if line.startswith(">"):
                if acc:
                    rows.append((acc, desc, "".join(seq)))
                _, acc, rest = line[1:].split("|", 2)  # >sp|P12345|NAME_SPECIES description OS=... OX=...
                desc, seq = rest.strip(), []
            else:
                seq.append(line.strip())
    rows.append((acc, desc, "".join(seq)))

    return pl.DataFrame(rows, schema=["accession", "description", "sequence"], orient="row").with_columns(
        length=pl.col("sequence").str.len_chars(),
        fragment=pl.col("description").str.contains(r"\(Fragment\)"),
        taxid=pl.col("description").str.extract(r"OX=(\d+)").cast(pl.Int64),
    )


def read_taxonomy():
    """Kingdom and major group of every SwissProt entry."""
    lineage = (
        pl.read_csv(download_taxonomy(), separator="\t")
        .rename({"Entry": "accession", "Taxonomic lineage (Ids)": "lineage"})
        .with_columns(
            pl.col("lineage").str.extract_all(r"\d+ \(").list.eval(pl.element().str.strip_chars(" (").cast(pl.Int64))
        )
    )

    def first_match(names, default):
        return pl.coalesce(
            [pl.when(pl.col("lineage").list.contains(taxid)).then(pl.lit(name)) for taxid, name in names.items()]
            + [pl.lit(default)]
        )

    return lineage.select(
        "accession", kingdom=first_match(KINGDOMS, "other"), group=first_match(GROUPS, "other")
    )


def prepare_data():
    df = read_fasta().filter((pl.col("length") <= 1022) & ~pl.col("fragment"))
    df.write_parquet(swissprot_dir() / "swissprot.parquet")
    return make_splits(df, swissprot_dir() / "splits")
