# Learning to Plan for Human-Robot Cooperative Carrying
Code for the paper paper *[It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying](https://arxiv.org/abs/2209.12890)* [1].
Link to [video](https://www.youtube.com/watch?v=CqWh-yWOgeA).

The main branch contains code for training a Variational Recurrent Neural Network for the cooperative table-carrying task (link to [repository for human-robot cooperative table-carrying, a custom gym environment](https://github.com/eleyng/table-carrying-ai). To *execute* the trained model in the environment, please see instructions housed in the gym environment repository.

## Installation

We recommend following the instructions for [creating a virtual environment and installation for the custom gym environment](https://github.com/eleyng/table-carrying-ai) first. Activate the environment using `conda activate [environment name]`. Afterwards, to install the remaining packages required for training the model, clone this repo and run:
```
$ cd cooperative-world-models
$ pip install -e .
```

## Download dataset and trained models

Download [Link.](https://drive.google.com/drive/folders/1RqmUrl0xPPURRrGFpoC3pgIm-NmgyKV6?usp=share_link)

## Training

To train the model, run the following:
```
python3 -m scripts.traindynamics --SEQ_LEN [num timesteps for model output] --H [num timesteps of observation before prediction] --RSIZE [hidden size] --data_dir [path to dataset dir from base dir] --experiment_name [custom name for model]
```

