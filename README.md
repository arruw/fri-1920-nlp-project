# Aspect-based sentiment analysis
- Authors: Žan Jaklič, Iztok Ramovš, Matjaž Mav
- Draft preview: [here](https://www.overleaf.com/8497658145vrdhbgxccgsd)

## Folder structure
```
**code** : IPython Notebooks and Python files
**data** : required datasets and saved pickle DataFrames for faster execution
**models** : saved Neural Net Models 
**old** : deprecated, unrunnable files from the project's 1st phase
```

## Model order
Run models in the given order:
```
parsing.ipynb
context_extraction.ipynb
feature_expansion.ipynb
modelling.ipynb
```

## Installation help
```bash
# Download Anaconda for your OS [here](https://www.anaconda.com/products/individual)

# Install PyTorch
$ conda install pytorch torchvision -c pytorch

# Install Stanza
$ conda install -c stanfordnlp stanza

#Open Anaconda Prompt and download Slovene Stanza modelling
stanza.download('sl')
```