# Baymer
Bayesian MCMC to estimate hierarchical tree-based sequence context models

## Installation

Baymer requires mini-conda for installation. For information on installing mini-conda for your operating system, please view https://docs.conda.io/en/latest/miniconda.html.

After installing mini-conda, run the following commands to install Baymer:
1) git clone https://github.com/bvoightlab/Baymer
2) cd Baymer
3) source install.sh

Installation should take less than 5 minutes. After installing, you can activate the conda environment containing the installed Baymer package:
conda activate baymer

## Running Baymer

Note that all scripts can be run as standalone programs or imported and collapsed into a single script, as demonstrated in the tutorial jupyter notebook.

### Generating Baymer Input Files

Baymer expects jsons of the following format as as input files:

{Context: [A polymorphisms, C polymorphisms, G polymorphisms, T polymorphisms, total contexts, context reference index, list reference index], ...}

where the "context reference index" is the 0-indexed position of the reference nucleotide in the context and "list reference index" is the position in the list of the reference nucleotide. Therefore, for the 3-mer context "AAA", the context reference index would be 1 and the list reference index would be 0.

Note that the input json should only contain a single context reference index and sequence context length.

Generating these files necessitates counting both the contexts and the polymorphism counts. We have provided scripts to count these quantities and format the proper json if desired.

#### Counting Contexts


