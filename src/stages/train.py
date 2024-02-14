import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.load_params import load_params
from src.utils.train_utils import train_model


def train_and_save_model(params):
    random_state = params.base.random_state
    train_test_dir_path = Path(params.data_split.train_test_dir_path)
    class_weight = params.train.class_weight
    criterion = params.train.criterion
    max_leaf_nodes = params.train.max_leaf_nodes
    min_samples_split = params.train.min_samples_split
    min_samples_leaf = params.train.min_samples_leaf
    splitter = params.train.splitter
    target_column = params.train.target_column
    model_name = params.train.model_name
    model_path = Path(params.train.model_path).absolute()
    model_path.mkdir(exist_ok=True)
    train_model(train_test_dir_path=train_test_dir_path,
                class_weight=class_weight,
                max_leaf_nodes=max_leaf_nodes,
                criterion=criterion,
                min_samples_split=min_samples_split,
                splitter=splitter,
                min_samples_leaf=min_samples_leaf,
                target_column=target_column,
                model_name=model_name,
                model_path=model_path,
                seed=random_state)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    train_and_save_model(params)
