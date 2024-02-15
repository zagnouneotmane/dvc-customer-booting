import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import Live
from src.pylogger import logger







def get_metrics(train_test_dir_path,
                model_file_path,
                target_column):

    test_data = pd.read_csv(train_test_dir_path/"test.csv", encoding = "ISO-8859-1")
    model = joblib.load(model_file_path)
    logger.info(f"training model {model_file_path}")

    test_x = test_data.drop([target_column], axis=1)
    test_y = test_data[[target_column]]

    predicted_booking = model.predict(test_x)
    logger.info(f'Predicted bookings:{target_column}')

    accuracy = accuracy_score(test_y, predicted_booking)
    precision = precision_score(test_y, predicted_booking)
    recall = recall_score(test_y, predicted_booking)
    f1 = f1_score(test_y, predicted_booking)
    scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    logger.info(f"Model Scores: {target_column}")
    return scores
def log_into_dvclive(scores):
    with Live(dir="dvcliveevaluation") as live:
        #live.log_params(params=params)
        live.log_metric("accuracy", scores["accuracy"])
        live.log_metric("precision", scores["precision"])
        live.log_metric("recall", scores["recall"])
        live.log_metric( "f1", scores["f1"])
