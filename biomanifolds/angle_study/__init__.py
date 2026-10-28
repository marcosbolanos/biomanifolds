"""The helix face angle study of notebooks 1-5, for any model with an adapter (models.py).

Stages, in order (scripts/run_full_angle_study.py runs them):
  data      splits and labels (prepared if missing), and their figures (notebook 1)
  bins      mean residual stream per helix-face and pLDDT bin, PCA explained variance, linear and harmonic fits (notebook 2)
  ablation  ablation of the residue's own angle at the masked residue, for 1M test residues (notebook 3)
  analysis  statistics and figures of the ablation (notebook 4)
  steering  steering from 4 source angles to 72 target angles (notebook 5)
"""
