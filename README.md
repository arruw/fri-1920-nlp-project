# Aspect-based sentiment analysis
- Authors: Žan Jaklič, Iztok Ramovš, Matjaž Mav
- Draft preview: [here](https://www.overleaf.com/read/qrxfxxwggtpj)

## Folder structure
```
TODO
```

## Helper commands
```bash
# Install specific dependencies
$ sudo apt-get install libicu-dev

# Create local Conda environment and install required depandencies
$ make conda-install

# Activate Conda environment in bash
$ conda activate ./.env

# Export Conda depandencies
$ make conda-export

# Download and extract dataset into ./dataset
$ make dataset-download

# Clean workspace (delete ignored files)
$ make git-clean
```

## Links
- [Sentiment Analysis: Does Coreference Matter?](https://pdfs.semanticscholar.org/041e/0a842a9d039c14f03ff21dafa82cca202f50.pdf)