# chronos_dnlp


## Setup

### Clone and Install
Clone repo: 
```bash
# Clone this repo 
git clone git@github.com:fpgmina/chronos_dnlp.git
cd chronos_dnlp
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
