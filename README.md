# METHOD FOR ENSURING THE SURVIVABILITY OF AN INFORMATION SYSTEM ON A MOBILE PLATFORM WITH ACCOUNT FOR THE FEASIBILITY OF REDUNDANT RESOURCE-PROVISIONING PROFILES

Experimental simulation code and reproducible materials for the article.

## Overview

This repository contains an experimental Python script used to generate Section 5 simulation results for a survivability-oriented method for an information system on a mobile platform, with explicit account for the feasibility limits of redundant resource-provisioning profiles.

## Repository contents

- `exp_section5_realistic_.py` — main experimental simulation script
- `requirements.txt` — Python dependencies
- `CITATION.cff` — citation metadata for the repository
- `LICENSE` — MIT License

## Requirements

- Python 3.10+ (recommended)
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```
## Run

```bash
python exp_section5_realistic_.py
```
##  Output

The script generates figures and tabular outputs (CSV) for analysis of:
- resource dynamics
- connectivity degradation
- reconfiguration behavior
- feasibility-aware profile switching
- survivability-related performance indicators

## Reproducibility notes

- Use the fixed random seed defined in the script (if applicable).
- Record Python and package versions for exact result matching.
- For publication use, cite a tagged release of this repository.

## Citation

If you use this code or reproduce the experiments, please cite this repository (see CITATION.cff) and the corresponding article.

## License

MIT License.
