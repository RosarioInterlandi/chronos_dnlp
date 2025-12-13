# chronos_dnlp


## Setup

### Clone and Install
Clone repo: 
```bash
# Clone this repo with submodules
git clone --recurse-submodules git@github.com:fpgmina/chronos_dnlp.git
cd chronos_dnlp

# If you already cloned without submodules:
git submodule update --init --recursive
```

To create conda environment run:
```bash
conda env create -f environment.yml
```
To activate conda environment run:
```bash
conda activate chronos_dnlp
```

Before pushing a change remember to run: 
* ```pytest```
* ```black .```
* ```ruff check . --fix```
