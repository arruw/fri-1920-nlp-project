# Aspect-based sentiment analysis
- Authors: Žan Jaklič, Iztok Ramovš, Matjaž Mav
- Draft preview: [here](https://www.overleaf.com/8497658145vrdhbgxccgsd)

## Folder structure
**code** : IPython Notebooks and Python files  
**data** : required datasets and saved pickle DataFrames for faster execution  
**models** : saved Neural Net Models  
**old** : deprecated, unrunnable files from the project's 1st phase  


## File order
Run notebooks in the given order:
```
parsing.ipynb
context_extraction.ipynb
feature_expansion.ipynb
modelling.ipynb
```

For evaluating saved FFN model run following:
```
python code/nn_from_feature_vector.py
```

To retrain FFN model run following:
```
python code/nn_from_feature_vector.py -train
```

## Installation help
```bash
# Download Anaconda (Python 3.7) for your OS
https://www.anaconda.com/products/individual

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install Stanza
conda install -c stanfordnlp stanza

# Open Python in Anaconda Prompt and download Slovene Stanza modelling
import stanza
stanza.download('sl')

# Install Keras and Tensorflow (gpu or cpu, doesnt't matter)
conda install -c conda-forge tensorflow
conda install keras

# Install h5py for saving and reading Keras models to disk
conda install h5py 

# Install Pandas
conda install -c anaconda pandas
```