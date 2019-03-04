# LSTNet
This repository is a Tensorflow / Keras implementation of "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks" paper https://arxiv.org/pdf/1703.07015.pdf

This implementation has been inspired by the following Pytorch implementation https://github.com/laiguokun/LSTNet

## Prerequisite
This repository uses the following modules from https://github.com/fbadine/util:
- Msglog.py
- model_util.py

## Usage
There are 4 different script samples to run the model on the different datasets:
- electricity.sh
- exchange_rate.sh
- solar.sh
- traffic.sh

## Dataset
You can download the dataset from: https://github.com/laiguokun/multivariate-time-series-data

## Environment
- Python 3.6.8
- Tensorflow 1.11.0
- Keras 2.1.6-tf
