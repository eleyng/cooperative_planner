# Learning to Plan for Human-Robot Cooperative Carrying
Code for the ICRA 2023 paper **[It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying](https://sites.google.com/view/cooperative-carrying)** [1].

<p float="center">
  <img src="/media/test_holdout_pred_8x.gif" width="45%" />
  <img src="/media/unseen_map_pred_8x.gif" width="45%" /> 
</p>

The main branch contains code for training a Variational Recurrent Neural Network for the cooperative table-carrying task (link to [repository for human-robot cooperative table-carrying, a custom gym environment](https://github.com/eleyng/table-carrying-ai). To *execute* the trained model in the environment, please see instructions housed in the gym environment repository.

## Installation

We recommend following the instructions for [creating a virtual environment and installation for the custom gym environment](https://github.com/eleyng/table-carrying-ai) first. Activate the environment using `conda activate [environment name]`. Afterwards, to install the remaining packages required for training the model, clone this repo and run:
```
$ cd cooperative_planner
$ pip install -e .
```

## Download dataset and trained models used in [1]

Download full dataset for [1] [here](https://drive.google.com/drive/folders/1RqmUrl0xPPURRrGFpoC3pgIm-NmgyKV6?usp=share_link). To use this dataset, see [documentation for dataset](https://github.com/eleyng/cooperative-planner/tree/main/datasets) and [documentation for trained models](https://github.com/eleyng/cooperative-planner/tree/main/trained_models).

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

## Cite
If you would like to use our environment, please cite us:
```
@article{ng2022takes,
  title={It Takes Two: Learning to Plan for Human-Robot Cooperative Carrying},
  author={Ng, Eley and Liu, Ziang and Kennedy III, Monroe},
  journal={arXiv preprint arXiv:2209.12890},
  year={2022}
}
```

## Contact  
For issues, comments, suggestions, or anything else, please contact [Eley Ng](https://eleyng.github.io) at eleyng@stanford.edu.
