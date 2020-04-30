# SemEval 2020 - Task 10: Emphasis Selection For Written Text in Visual Media

### Team 'Corona Survivors':
- Rishabh Agarwal
- [Sahil Dhull](https://sahildhull.github.io/)
- Vipul Singhal

#### Under the guidance of:
[Prof. Ashutosh Modi](https://ashutosh-modi.github.io/)

## Installation
```
git clone https://github.com/SahilDhull/emphasis_selection.git
cd emphasis_selection
virtualenv project
pip install -r requirements.txt
```

Note: If not having virtualenv, install it using: `pip install virtualenv`.
Download pre-trained GloVe embeddings from `http://nlp.stanford.edu/data/glove.6B.zip`

## Problem Statement
The problem statement ([Task 10 in SemEval-2020](https://competitions.codalab.org/competitions/20815)) is designing automatic methods for emphasis selection i.e. choosing candidates for emphasis in short written text.

More formally,
Given a sequence of words or tokens C = <img src="https://render.githubusercontent.com/render/math?math=\{ x_1, x_2, ..., x_n \}">, we want to compute a score <img src="https://render.githubusercontent.com/render/math?math=S_i"> for each <img src="https://render.githubusercontent.com/render/math?math=x_i"> which indicates the degree of emphasis to be laid on the word.

## Directory Structure
- datasets/ -> contains the train and development sets

- BiLSTM_Attention_approach/ -> contains the model for 'BiLSTM + Attention' approach

- Transformers_approach/ -> contains model and ensemble files for Transformers approach

- requirements.txt -> contains versions of packages required

- eval_metric.py -> contains the function for Match_m as defined in task
