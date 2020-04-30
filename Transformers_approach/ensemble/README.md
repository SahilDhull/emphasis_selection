# ENSEMBLE

## Directory Structure

- config.py -> contains configuration parameters like save paths, input file paths, etc

- run.py -> script for ensembling of various runs of different models

- output/ -> output of emsembles are stored in this directory in the format: `bert_n1_roberta_n2_xlnet_n3.txt`, where n1 is number of runs of bert, n2 is number of runs of roberta and n3 is number of runs of xlnet used for ensemble

## Running the ensemble

```python run.py n1 n2 n3```

E.g. `python run.py 1 2 3` runs the ensemble for 1 run of BERT, 2 runs of RoBERTa and 3 runs of XLNet