# chronos_dnlp

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
