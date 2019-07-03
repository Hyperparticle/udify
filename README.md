# UDify

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

UDify is a single model that parses Universal Dependencies (UPOS, UFeats, Lemmas, Deps) jointly, accepting any of 75 
supported languages as input (UD v2.3). This repository accompanies the paper, 
"[75 Languages, 1 Model: Parsing Universal Dependencies Universally](https://arxiv.org/abs/1904.02099)," 
providing tools to train a multilingual model capable of parsing any Universal Dependencies treebank with high 
accuracy. This project also supports training and evaluating for the 
[SIGMORPHON 2019 Shared Task #2](https://sigmorphon.github.io/sharedtasks/2019/task2/).

The project is built using [AllenNLP](https://allennlp.org/) and [PyTorch](https://pytorch.org/).

**Note that this repository is still a work in progress** (sorry, I'm traveling a lot :airplane:).

## Getting Started

Install the Python packages in `requirements.txt`. UDify depends on AllenNLP and PyTorch. For Windows OS, use 
[WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10). Optionally, install TensorFlow to get access to 
TensorBoard to get a rich visualization of model performance on each UD task.

```bash
pip install -r ./requirements.txt
```

Download the UD corpus by running the script

```bash
bash ./scripts/download_ud_data.sh
```

or alternatively download the data from [universaldependencies.org](https://universaldependencies.org/) and extract 
into `data/ud-treebanks-v2.3/`, then run `scripts/concat_ud_data/sh` to generate the multilingual UD dataset.

### Training the Model

To train the multilingual model (fine-tune UD on BERT), run the command

```bash
python train.py --config config/ud/multilingual/udify_bert_finetune_multilingual.json --name multilingual
```

which will begin loading the dataset and model before training the network. The model metrics, vocab, and weights will
be saved under `logs/multilingual`. Note that this process is highly memory intensive and requires 16+ GB of RAM and 
12+ GB of GPU memory. The training may take 20 or more days to complete all 80 epochs depending on the type of your 
GPU.

### Viewing Model Performance

One can view how well the models are performing by running TensorBoard

```bash
tensorboard --logdir logs
```

This should show the currently trained model as well as any other previously trained models. The model will be stored 
in a folder specified by the `--name` parameter as well as a date stamp, e.g., `logs/multilingual/2019.07.03_11.08.51`.

### Predicting Universal Dependencies from a Trained Model

To predict UD annotations, one can supply the path to the trained model and an input `conllu`-formatted file:

```bash
python predict.py <archive_dir> <input.conllu> [--pred_file <output.conllu>]
```

For instance, predicting the dev set of English EWT with the trained model saved under 
`logs/multilingual/2019.07.03_11.08.51` can be done with

```bash
python predict.py logs/multilingual/2019.07.03_11.08.51 data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllu --pred_file en-ewt_dev.conllu
```

and will save the output predictions to `logs/multilingual/2019.07.03_11.08.51/en-ewt_dev.conllu`.

## Configuration Options

One can specify the type of device to run on. For a single GPU, use the flag `--device 0`, or `--device -1` for CPU.

## Pretrained Models [Coming Soon]

The pretrained multilingual UDify model can drastically speed up fine-tuning on any UD treebank. A link to download the 
pretrained model is coming soon (need to find a proper hosting site).

## SIGMORPHON 2019 Shared Task [Coming Soon]

A modification to the basic UDify model is available for parsing morphology in the 
[SIGMORPHON 2019 Shared Task #2](https://sigmorphon.github.io/sharedtasks/2019/task2/). The configuration will be made 
available soon.
