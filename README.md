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

## Download dataset and trained models used in [1]

See [documentation for dataset](https://github.com/eleyng/cooperative-planner/tree/main/datasets) and [documentation for trained models](https://github.com/eleyng/cooperative-planner/tree/main/trained_models).

## Training

To train the model, run the following:
```
python3 -m scripts.run --train
```
See the full list of args in `configs/exp_config.py`.

## Visualize

During training, you can visualize plots of the predictions on the maps in the `results/plots` directory that is created when you begin training.

## TODO

Test the following:
- Train
  - train from scratch (completed)
  - resume training by load model from icra (seems to work, but the model I am providing doesn't seem to load -- need to retrain or find the real model??)
- Eval 
  - load model and run predictions
  - plots
