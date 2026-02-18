# astro-utils

Custom Python utilities for astrophysical data analysis and visualisation

## Overview

**astro-utils** is a personal collection of Python functions designed to streamline common tasks in galaxy and cosmological data analysis. The package includes tools for data extraction, filtering, kinematic analysis, and structural decomposition, as well as routines for producing publication-quality plots and radial profiles.

## Features

* **Data handling:** Functions for extracting, cleaning, and filtering simulation or observational catalogues.
* **Kinematic analysis:** Utilities for computing and visualising velocity anisotropies, rotation curves, and related diagnostics.
* **Density and radial profiles:** Custom routines for generating 1D and 2D profiles of density, mass, or other quantities.
* **Morphological decomposition:** Functions for separating components such as bulge and disk or analysing structural parameters.
* **Dust and radiative transfer tools:** Helpers for working with *SKIRT* maps and other dust attenuation or emission outputs.
* **Plotting utilities:** Quick routines for consistent figure formatting and multi-panel visualisations.


## Installation

Clone the repository and add it to your Python path:

```bash
git clone https://github.com/PedroCataldi/astro-utils.git
cd astro-utils
pip install -e .
```

## Requirements

* Python 3.8+
* NumPy
* SciPy
* Matplotlib
* Astropy
* h5py (for simulation data handling)

## Author

Pedro Cataldi
