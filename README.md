 # README.md

This project is a PyTorch implementation of a Machine Translation system, from German to English, using the Marian model provided by Hugging Face.

## Overview

The main script loads German and English parallel texts, tokenizes them using a pre-trained tokenizer, and then trains a translation model for a specified number of epochs. The trained model is then saved together with the tokenizer so that it can be used again without retraining.

## Requirements

To run this script, ensure that you have the following dependencies installed:

- torch
- transformers
- sklearn
- nltk
- tqdm

News Commentary - https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz (use DE-EN parallel data)

## Data

The script expects German and English parallel texts located in files 'news-commentary-v9.de-en.de' and 'news-commentary-v9.de-en.en'.

## Usage

You can easily run the main script with the command `python translator.py`. 

## Function Description

The script first loads a pre-trained tokenizer and model from the Helsinki-NLP/opus-mt-de-en model, then it loads the dataset. 

The data preparation function `prepare_data()` tokenizes the inputs and targets. 

DataLoaders are created using the `create_dataloader()` function. 

Model training proceeds in epochs, with the training and evaluation of each epoch defined in the `train_epoch()` and `evaluate_epoch()` functions respectively. The model's loss is outputted for each epoch.

At the end of the training, the model and tokenizer are saved. 

## Output

The script outputs the Cross Entropy loss for the training and validation sets at each epoch. It also computes and outputs the BLEU score after each epoch as an evaluation metric for the translations. The best performing model is then saved to the local disk.
