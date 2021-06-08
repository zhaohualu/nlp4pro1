Code for the BERT classification in ''Natural Language Processing and Machine Learning Methods to Characterize Unstructured Patient-Reported Outcomes: A Validation Study''


# Pytorch-Transformers-Classification

The code is derived from [this repo](https://github.com/ThilinaRajapakse/pytorch-transformers-classification).

This repository is based on the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library by HuggingFace. 

The code was run in an anaconda environment, the details of which is shown in the environment.yml.


# Datasets format

The data are stored in the data folder.

The data needs to be in `tsv` format, with four columns, and no header.

This is the required structure.

- `guid`: An ID for the row.
- `label`: The label for the row (should be an int).
- `alpha`: A column of the same letter for all rows. Not used in classification but still expected by the `DataProcessor`.
- `text`: The sentence or sequence of text.

An example of creating data files for cross-validated samples is data_prep.py. Two variables to set, the file name of the complete data file "DataFilename" and the index of the CV batch used to generate file names.
```python
DataFilename = sys.argv[1]
CVBatch = int(sys.argv[2])
```

# Run model

Please run run_model.py and set the pre-trained model below before running the code

## Set hyperparameters

Hyperparameters can be changed in the args object in run_model.py.

## Set pre-trained models


### BERT

BERT is downloaded from the https://huggingface.co/models automatically.

```python
NEpochs = 10
ModelType = 'bert'
ModelName = 'bert-base-uncased'
CVBatch = 1

configvar=ModelName
tokenvar=ModelName
modelvar=ModelName
from_tf_flag=False
```


### BioBERT

please download the pretrained model BioBERT-Base v1.1 (+ PubMed 1M) from https://github.com/dmis-lab/biobert
and change the path below to the folder storing the model

```python
NEpochs = 10
ModelType = 'bert'
ModelName = 'bert-base-cased'
CVBatch = 1


configvar='path/'
tokenvar='path/'
modelvar='path/model.ckpt-1000000'
from_tf_flag=True
```

### BlueBERT

Please download the pretrained model from https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT and change the path below to the folder storing the model

```python
NEpochs = 10
ModelType = 'bert'
ModelName = 'bert-base-uncased'
CVBatch = 1


configvar='path/'
tokenvar='path/'
modelvar='path/'
from_tf_flag=False
```


### Clinical BERT

please download the pretrained model BlueBERT-Base, Uncased, PubMed from https://github.com/ncbi-nlp/bluebert and change the path below to the folder storing the model

```python
NEpochs = 10
ModelType = 'bert'
ModelName = 'bert-base-cased'
CVBatch = 1


configvar='path/'
tokenvar='path/'
modelvar='path/'
from_tf_flag=False
```
