# Emphasis Selection

This is our project for the course CS698O (Topics in NLP).

### Group Members:
- Rishabh Agarwal
- Sahil Dhull
- Vipul Singhal

## Problem Statement
The problem statement ([Task 10 in SemEval-2020](https://competitions.codalab.org/competitions/20815)) is designing automatic methods for emphasis selection i.e. choosing candidates for emphasis in short written text.

More formally,
Given a sequence of words or tokens C = $`\{ x_1, x_2, ..., x_n \}`$, we want to compute a score $`S_i`$ for each $`x_i`$ which indicates the degree of emphasis to be laid on the word.

## Proposed Approach
We propose an end-to-end model which takes as input the words in the text and corresponding to each word, gives a score describing the degree of emphasis to be laid on the word. We plan to try at least two different types of sequence labeling model to learn emphasis patterns.

The first approach (Akhundov et al., 2018) involves character-level or byte-level embeddings of each word of a entence computed using a BiLSTM layer, concatenated with word embeddings computed using pre-trained embeddings which is further passed through a pair of BiLSTM Layers.
We will finally add a fully connected layer to predict a score for each word. The byte-level (or the character-level) embeddings will capture the morphological information of the words in the sentence, whereas word embeddings capture the semantic information of the words. Another variant of the above approach can be using GRU layer instead of BiLSTM layer (Yang et al., 2016).

The second approach (Emelyanov and Artemova, 2019) uses the BERT Language Model as embeddings with bidirectional recurrent network, attention, and CRF layer on the top. If the time and machines permit, we can even fine-tune the BERT Language Model.

## Corpus Description
The dataset taken is the dataset provided in SemEval-2020 Task-10: Emphasis Selection for Written Text in Visual Media. The dataset consists of 1,206 short text instances obtained from Adobe Spark. The dataset contains 7,550 tokens, and the average number of tokens per instance is 6.16, ranging from 2 to 25 tokens. On average, each instance contains 2.38 emphases, and the ratio of non-emphasis to emphasis tokens is 1.61.

## Directory Structure
datasets/ - contains the train and development set
glove/ - contains the script for preprocessing GloVe embeddings
model/ - contains the codes for all the models used
tests/ - contains the results/logs of the tests run

### References
- Adnan Akhundov, Dietrich Trautmann, and Georg Groh. 2018. Sequence labeling: A practical approach. arXiv preprint arXiv:1808.03926.
- Anton A Emelyanov and Ekaterina Artemova. 2019. Multilingual named entity recognition using pretrained embeddings, attention mechanism and ncrf. arXiv preprint arXiv:1906.09978.
- Amirreza Shirani, Franck Dernoncourt, Paul Asente, Nedim Lipka, Seokhwan Kim, Jose Echevarria, and Thamar Solorio. 2019. Learning emphasis selection for written text in visual media from crowd-sourced label distributions. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1167–1172.
- Zhilin Yang, Ruslan Salakhutdinov, and William Cohen. 2016. Multi-task cross-lingual sequence tagging from scratch. arXiv preprint arXiv:1603.06270.