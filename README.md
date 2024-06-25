# FragmentFactory

Devising human interpretable diagnostic glycan fragments from MS/MS-spectra.

## Abstract

tbd

## Setup

Download the code from github via

```bash
git clone git@github.com:BojarLab/FragmentFactory.git
```

Then, setup an environment via

```bash
conda create -y -n ff python=3.9
conda activate ff
mamba install -c conda-forge -c bioconda -c kalininalab datasail-lite
pip install -r requirements.txt
```

## Usage

Inside the FragmentFactory folder, one can run 

```bash
python train.py <path/to/spectra_df.pkl> <output-prefix> --weighting --GPID_SIM <val>
```

to create custom trees and a rough visualization thereof. 