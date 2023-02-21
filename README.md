# Learning to Plan for Human-Robot Cooperative Carrying
Code for the paper paper *[It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying](https://arxiv.org/abs/2209.12890)* [1].
Link to [video](https://www.youtube.com/watch?v=CqWh-yWOgeA).

<p float="center">
  <img src="/media/test_holdout_pred_8x.gif" width="45%" />
  <img src="/media/unseen_map_pred_8x.gif" width="45%" /> 
</p>

The main branch contains code for training a Variational Recurrent Neural Network for the cooperative table-carrying task (link to [repository for human-robot cooperative table-carrying, a custom gym environment](https://github.com/eleyng/table-carrying-ai). To *execute* the trained model in the environment, please see instructions housed in the gym environment repository.

## Installation

We recommend following the instructions for [creating a virtual environment and installation for the custom gym environment](https://github.com/eleyng/table-carrying-ai) first. Activate the environment using `conda activate [environment name]`. Afterwards, to install the remaining packages required for training the model, clone this repo and run:
```
$ cd cooperative-world-models
$ pip install -e .
```

## Download dataset and trained models used in [1]

Download full dataset for [1][here](https://drive.google.com/drive/folders/1S5HoeQzykBcxXs9lG_e-_Kjdf89kiW_C?usp=share_link). To use this dataset, see [documentation for dataset](https://github.com/eleyng/cooperative-planner/tree/main/datasets) and [documentation for trained models](https://github.com/eleyng/cooperative-planner/tree/main/trained_models).

## Training

To train the model, run the following:
```
python3 -m scripts.run --train
```
See the full list of args in `configs/exp_config.py`.

## Testing

To test the model on the dataset, run the following:
```
python3 -m scripts.run --restore --artifact-path [path to saved model .ckpt file] --test-data [test_holdout | unseen_map]
```
See the full list of args in `configs/exp_config.py`.

## Visualize

During training, you can visualize plots of the predictions *while the script is running* in the `results/plots` directory that is created when you begin training/testing.

## TODO
- upload trained model
