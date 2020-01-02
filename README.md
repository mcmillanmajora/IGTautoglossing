# Dependecies
For part of speech and lemmatization https://github.com/stanfordnlp/python-stanford-corenlp 

For handling ODIN corpora https://github.com/xigt/xigt 

For the CRF models https://sklearn-crfsuite.readthedocs.io/en/latest/ 

For saving the models https://github.com/joblib/joblib

# Data
Download the ODIN data from http://depts.washington.edu/uwcl/odin/

# Command Line Usage
For first time usage:
```
python3 ./main.py ./path_to_corpus_file ./path_to_output_file
```
If you want to reload already trained models:
```
python3 ./main.py ./path_to_corpus_file ./path_to_output_file ./path_to_saved_models_file
```
