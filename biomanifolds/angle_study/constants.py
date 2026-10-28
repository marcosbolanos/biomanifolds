"""Settings of the angle study that are the same for every model."""

from dataclasses import dataclass, replace

AA = "ACDEFGHIKLMNPQRSTVWY"  # order of the saved amino-acid log-probabilities
HYDROPHOBIC = "ACGILMFPWYV"  # sign of the Eisenberg consensus scale
EISENBERG = dict(zip("ARNDCQEGHILKMFPSTWYV", [0.62, -2.53, -0.78, -0.90, 0.29, -0.85, -0.74, 0.48, -0.40, 1.38,
                                                1.06, -1.50, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08]))

METHODS = ["bin_means", "linear", "harmonic"]  # approximations of the mean residual stream at an angle
CONTROLS = ["", " | random angle", " | random vector"]
EDITS = [method + control for method in METHODS for control in CONTROLS]  # ablation edits, without "none"


@dataclass(frozen=True)
class Sizes:
    """Sample sizes and seeds."""
    train_fraction: float = 0.10  # train proteins, MD5 rule on the accession
    test_residues: int = 1_000_000  # helix residues of the test set, for the test bins and the ablation
    test_seed: int = 0
    steering_per_bin: int = 5000  # residues per source bin, one per protein
    steering_seed: int = 0
    n_boot: int = 2000  # bootstrap samples (resampling proteins)
    min_count: int = 200  # bins with fewer residues are left out of the PCA

    def smoke(self):
        """Small sizes, to test the whole pipeline in a few minutes."""
        return replace(self, train_fraction=0.002, test_residues=2000, steering_per_bin=50, n_boot=200, min_count=1)


SOURCE_ANGLES = [0, 90, 180, -90]  # steering: centres of the source bins, degrees
SOURCE_BIN_WIDTH = 5  # degrees
TARGETS = list(range(-180, 180, 5))  # steering: target angles, degrees
CONTROL_OFFSETS = [45, 90, 180, -90]  # steering: random vectors with the norm of the edit to source + offset
