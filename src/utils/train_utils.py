from src.pylogger import logger
import pandas as pd
from sklearn.tree import  DecisionTreeClassifier
import joblib
import os




def train_model(train_test_dir_path,
                class_weight,
                max_leaf_nodes,
                criterion,
                min_samples_split,
                splitter,
                min_samples_leaf,
                target_column,
                model_name,
                model_path,
                seed):

    train_data = pd.read_csv(train_test_dir_path/ 'train.csv', encoding = "ISO-8859-1")


    train_x = train_data.drop([target_column], axis=1)
    train_y = train_data[[target_column]]



    model = DecisionTreeClassifier(class_weight = class_weight,
            criterion =  criterion,
            max_leaf_nodes = max_leaf_nodes,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            splitter =  splitter, 
            random_state=seed)
    model.fit(train_x, train_y)
    logger.info(f"training model {model_name}")
    joblib.dump(model, os.path.join(model_path, model_name))
    logger.info(f"Downloading model {model_name} into file {model_path/model_name}")