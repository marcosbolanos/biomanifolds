import tomllib
import urllib.request
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALPHAFOLD_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar"


def alphafold_dir():
    with open(PROJECT_ROOT / "project-config.toml", "rb") as f:
        config = tomllib.load(f)

    path = PROJECT_ROOT / config["paths"]["alphafold"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_data():
    tar_path = alphafold_dir() / "swissprot_pdb_v6.tar"
    if not tar_path.exists():
        part_path = tar_path.with_suffix(".tar.part")  # renamed only when complete
        with tqdm(unit="B", unit_scale=True, desc="Downloading AlphaFold SwissProt") as bar:

            def hook(blocks, block_size, total_size):
                bar.total = total_size
                done = blocks * block_size
                if total_size > 0:
                    done = min(done, total_size)
                bar.update(done - bar.n)

            urllib.request.urlretrieve(ALPHAFOLD_URL, part_path, reporthook=hook)
        part_path.rename(tar_path)
    return tar_path
