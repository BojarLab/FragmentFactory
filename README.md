# FragmentFactory

Devising human interpretable diagnostic glycan fragments from MS/MS-spectra.

## Abstract

Structural details of oligosaccharides, or glycans, often carry biological relevance, which is why they are typically elucidated using tandem mass spectrometry. Common approaches to distinguish isomers rely on diagnostic glycan fragments for annotating topologies or linkages. Diagnostic fragments are often only known informally among practitioners or stem from individual studies, with unclear validity or generalizability, causing annotation heterogeneity and hampering new analysts. Drawing on a curated set of 237,000 O-glycomics spectra, we here present a rule-based machine learning workflow to uncover quantifiably valid and generalizable diagnostic fragments. This results in fragmentation rules to robustly distinguish common O-glycan isomers. We envision this resource to improve glycan annotation accuracy and concomitantly make annotations more transparent and homogeneous across analysts.

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

Download the dataset from Zenodo here:
[https://doi.org/10.5281/zenodo.7940046](https://zenodo.org/records/12177171)


## Usage
Preprocess the downloaded dataset `FragmentFactory_dataset.pkl`
> [!NOTE]  
> Make sure the dataset is in the FragmentFactory folder or the path is set correctly within `FF_data_preprocessing.py`

```bash
python FF_data_preprocessing.py
```

Inside the FragmentFactory folder, one can run 

```bash
python train.py <path/to/spectra_df_processed.pkl> <output-prefix> --weighting --GPID_SIM <val>
```

to create custom trees and a rough visualization thereof. 
