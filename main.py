# -*- coding: utf-8 -*-
""" main.py """

# standard
import argparse
import os
import time
import logging

# external
import mlflow

# internal
from src.configs.config import CFG
from src.models.lstm_ae import LSTMAutoencoder
from src.inference.inferrer import Inferrer
from src.utils.logging_utils import connect_to_mlflow, check_gpu_usage

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')


def run_training():
    """Builds model, loads data, trains and evaluates"""
    with_mlflow = CFG["mlflow_config"]["enabled"]

    if with_mlflow:
        mlflow_run = connect_to_mlflow(CFG["mlflow_config"])

    model = LSTMAutoencoder(CFG)
    model.run_train()

    if mlflow.active_run():
        mlflow.end_run()


def run_inference():
    """Loads a trained model and data for prediction
        :return Dataframe with all the features plus a prediction column: 0=normal, 1=anomaly
    """
    infer = Inferrer(CFG)
    infer.load_model()
    infer.load_data()
    infer.predict()
    infer.eval()


if __name__ == '__main__':

    # This is different now, the data are defined in the src/configs/config.py
    # python teaching/learning_modules/anomaly_detection/main.py -e 'train'
    # python teaching/learning_modules/anomaly_detection/main.py -e 'infer' -m "20210708-114002_lstmae" -d "data/unsw-nb15/attack_short.csv"

    parser = argparse.ArgumentParser(description='Define mode of LM execution and parameters')

    parser.add_argument('-e', '--exec', help="mode of execution: 'train' or 'infer'", required=False, default='infer')
    # parser.add_argument('-m', '--model', help="the name of the trained model, if --exec='infer'", required=False,
    #                     default="20210714-123413_lstmae")
    # parser.add_argument('-d', '--datapath', help="the path to the dataframe, if --exec='infer'", required=False,
    #                     default="data/unsw-nb15/attack_short.csv")

    args = parser.parse_args()

    exec_mode = args.exec

    CFG['mem'] = check_gpu_usage()

    if exec_mode == 'train':
        run_training()
    elif exec_mode == 'infer':
        run_inference()
