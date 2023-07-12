# -*- coding: utf-8 -*-
"""LSTM-AE model"""

# standard
import joblib
import os
import time
from urllib.parse import urlparse

# internal
from src.models.base_model import BaseModel
from src.dataloader.dataloader import DataLoader
from src.utils.preprocessing_utils import create_sequences, transform_df, main_transformer, \
    create_dataframe_of_predicted_labels, DictToObject, split_dataset
from src.utils.eval_utils import *
from src.utils.logging_utils import *
from src.utils.tuning_utils import model_spaces

# external
import pandas as pd
import tensorflow
import mlflow
from mlflow.models.signature import infer_signature
from tensorflow import keras
from keras.callbacks import EarlyStopping, TensorBoard
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin, space_eval
from numpy.random import seed

seed(1)
tensorflow.random.set_seed(2)

# Run from terminal: tensorboard --logdir=./logs
tensorboard_callback = TensorBoard(log_dir='./logs')


class LSTMAutoencoder(BaseModel):
    """LSTM-AE model"""

    def __init__(self, config):
        super().__init__(config)

        self.data_types = self.config.data.data_types
        self.n_rows = self.config.data.n_rows
        self.ground_truth_cols = self.config.data.ground_truth_cols

        self.normal_x = None
        self.anomaly_x = None
        self.verification_x = None
        self.normal_y = None
        self.anomaly_y = None
        self.verification_y = None

        self.normal_y_seq = None
        self.normal_x_seq = None
        self.anomaly_y_seq = None
        self.anomaly_x_seq = None
        self.verification_y_seq = None
        self.verification_x_seq = None

        self.hyperparams = self.config.train.hyperparams
        self.train_setup = self.config.train.train_setup

        self.seq_time_steps = self.train_setup.seq_time_steps

        self.transformer = None
        self.trials = None

        self.model_storage = self.config.model.storage_path
        self.model_history = None
        self.model = None

        self.threshold = None
        self.mahalanobis_params = None
        # self.cov = None
        # self.mean = None

        self.with_mlflow = self.config.mlflow_config.enabled

    def _load_data(self):
        """Loads and Preprocess data """
        features = self.config.data.features
        columns = features + self.ground_truth_cols
        dict_data_types = (dict(zip(features, self.data_types)) if self.data_types else None)

        # Verification data will be used for testing unseen data
        normal, anomaly, verification = DataLoader().load_data(columns, self.config.data,
                                                               dict_data_types, self.n_rows,
                                                               mode='train')

        # If there are ground truth columns, remove and store them
        if self.ground_truth_cols is not None:
            # y labels
            self.normal_y = normal.loc[:, self.ground_truth_cols]  # only label as ground truth, skip attack_cat
            self.anomaly_y = anomaly.loc[:, self.ground_truth_cols]
            self.verification_y = verification.loc[:, self.ground_truth_cols]
            # X features
            self.normal_x = normal.loc[:, features]
            self.anomaly_x = anomaly.loc[:, features]
            self.verification_x = verification.loc[:, features]

    def _preprocess_data(self):
        """ Splits into training and obs and set training parameters"""

        # Fit a transformer to the concatenated data that will be used for training, tuning and validation
        total = pd.concat([self.normal_x, self.anomaly_x])
        self.transformer = main_transformer(total)

        # Prefixes to add _x or _y in the subsequent loop
        data = ["normal", "anomaly", "verification"]

        sequences = {}

        for key in data:
            # Transform feature sets
            x_trans = transform_df(getattr(self, key + "_x"), self.transformer)

            # Sequencing x features
            x_seq = create_sequences(x_trans, self.seq_time_steps)
            sequences[key + "_x_seq"] = x_seq
            print(f"{key.capitalize()} data input shape (#seqs, seq len, feats): {x_seq.shape}")

            # Sequencing y targets
            y_seq = create_sequences(getattr(self, key + "_y"), self.seq_time_steps)
            sequences[key + "_y_seq"] = y_seq
            print(f"{key.capitalize()} target shape (#seqs, seq len): {y_seq.shape}")

        # Assign each obtained sequence to the corresponding attribute
        for key, value in sequences.items():
            setattr(self, key, value)

        self._split_subsets()

    def _split_subsets(self):
        """ Split according to Malhotra et al.
            normal train (Malhotra: s_N)
            normal validation-1 (Malhotra: v_N1), validation early stopping, mean and cov (mahalanobis params)
            normal validation-2 (Malhotra: v_N2), threshold
            normal test(Malhotra: tN)
            anomalous validation (Malhotra: vA), threshold
            anomalous test (Malhotra: tA), testing
        """
        self.norm_train_x_seq, self.norm_val1_x_seq, self.norm_val2_x_seq, self.norm_test_x_seq = split_dataset(
            self.normal_x_seq, self.config.data.normal_splits_pc)
        self.norm_train_y_seq, self.norm_val1_y_seq, self.norm_val2_y_seq, self.norm_test_y_seq = split_dataset(
            self.normal_y_seq, self.config.data.normal_splits_pc)
        self.anom_val_x_seq, self.anom_test_x_seq = split_dataset(self.anomaly_x_seq,
                                                                  self.config.data.anomaly_splits_pc)
        self.anom_val_y_seq, self.anom_test_y_seq = split_dataset(self.anomaly_y_seq,
                                                                  self.config.data.anomaly_splits_pc)

    def _tuning(self, space):
        """Sets training parameters"""

        def objective(hyperparams):
            hyperparams = DictToObject(hyperparams)
            self._build(hyperparams)
            self._train(hyperparams)

            validation_loss = np.amin(self.model_history.history['val_loss'])
            print('Best validation loss of epoch:', validation_loss)

            return {'loss': validation_loss,
                    'status': STATUS_OK,
                    'model': self.model,
                    'params': hyperparams,
                    'model_history': self.model_history}

        self.trials = Trials()

        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=self.trials,
            max_evals=self.train_setup.tuning_max_evals)

        best_model = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['model']
        best_params = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['params']
        best_model_history = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['model_history']
        print(f"Optimized hyperparams: {best_params}")
        return best_params, best_model, best_model_history

    def _build(self, hyperparams):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.norm_train_x_seq.shape[1], self.norm_train_x_seq.shape[2])))

        # ENCODER
        for i in range(hyperparams.encoder_layers):
            if i == hyperparams.encoder_layers - 1:
                model.add(keras.layers.LSTM(units=hyperparams.encoder_units[i], activation=hyperparams.activation,
                                            return_sequences=False))
            else:
                model.add(keras.layers.LSTM(units=hyperparams.encoder_units[i], activation=hyperparams.activation,
                                            return_sequences=True))
            model.add(keras.layers.Dropout(rate=hyperparams.dropout_rate))

        # DECODER
        # Repeat the single vector from the encoder to match the length of the input sequences
        model.add(keras.layers.RepeatVector(self.norm_train_x_seq.shape[1]))

        for i in range(hyperparams.decoder_layers):
            model.add(keras.layers.LSTM(units=hyperparams.decoder_units[i], activation=hyperparams.activation,
                                        return_sequences=True))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(self.norm_train_x_seq.shape[2])))

        if hyperparams.regularization == 'l1':
            model.add(keras.layers.ActivityRegularization(l1=0.01))
        elif hyperparams.regularization == 'l2':
            model.add(keras.layers.ActivityRegularization(l2=0.01))

        self.model = model
        self.model.summary()

    def _train(self, hyperparams):
        optimizer = keras.optimizers.Adam(learning_rate=hyperparams.learning_rate)
        self.model.compile(loss='mse', optimizer=optimizer)
        es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
        self.model_history = self.model.fit(self.norm_train_x_seq, self.norm_train_x_seq,
                                            batch_size=hyperparams.batch_size,
                                            epochs=hyperparams.epochs, verbose=1,
                                            validation_data=(self.norm_val1_x_seq, self.norm_val1_x_seq),
                                            callbacks=[es, tensorboard_callback])

    def _eval(self):
        """Predicts results for the test dataset"""

        # Get the mahalanobis params (mean and cov) from the reconstruction error of normal val-1 (Malhotra: v_N1)
        rec_error_norm_val1_x = get_reconstruction_error(self.model, self.norm_val1_x_seq)
        self.mahalanobis_params = compute_mahalanobis_params(rec_error_norm_val1_x, self.norm_val1_x_seq)

        # Get the threshold from the anomaly scores of normal val-2 (Malhotra: v_N2) and anomalous val (Malhotra: vA)
        rec_error_norm_val2_x = get_reconstruction_error(self.model, self.norm_val2_x_seq)
        normal_scores = anomaly_scoring(rec_error_norm_val2_x, self.seq_time_steps, self.mahalanobis_params)

        # Get the threshold from the anomaly scores of anomalous val (Malhotra: vA)
        rec_error_anom_val_x = get_reconstruction_error(self.model, self.anom_val_x_seq)
        anomaly_scores = anomaly_scoring(rec_error_anom_val_x, self.seq_time_steps, self.mahalanobis_params)

        # Use normal and anomaly scores to compute the threshold
        self.threshold = compute_threshold(normal_scores, anomaly_scores)

        # TESTING METRICS

        # Evaluate on normal test (Malhotra: tN)
        rec_error_norm_test_x = get_reconstruction_error(self.model, self.norm_test_x_seq)
        n_accuracy, n_precision, n_recall, n_f1 = get_eval_metrics(rec_error_norm_test_x,
                                                                   self.norm_test_x_seq,
                                                                   self.norm_test_y_seq,
                                                                   self.threshold,
                                                                   self.seq_time_steps,
                                                                   self.mahalanobis_params)

        # Log metrics to mlflow
        log_mlflow_metrics(n_accuracy, n_precision, n_recall, n_f1, 'train')

        # Evaluate on anomalous test (Malhotra: tA)
        rec_error_anom_test_x = get_reconstruction_error(self.model, self.anom_test_x_seq)
        a_accuracy, a_precision, a_recall, a_f1 = get_eval_metrics(rec_error_anom_test_x,
                                                                   self.anom_test_x_seq,
                                                                   self.anom_test_y_seq,
                                                                   self.threshold,
                                                                   self.seq_time_steps,
                                                                   self.mahalanobis_params)

        # Log metrics to mlflow
        log_mlflow_metrics(a_accuracy, a_precision, a_recall, a_f1, 'val')

        # Evaluate on verification set (mixed)
        rec_error_verification_x_seq = get_reconstruction_error(self.model, self.verification_x_seq)
        accuracy, precision, recall, f1 = get_eval_metrics(rec_error_verification_x_seq,
                                                           self.verification_x_seq,
                                                           self.verification_y_seq,
                                                           self.threshold,
                                                           self.seq_time_steps,
                                                           self.mahalanobis_params)

        # Log metrics to mlflow
        log_mlflow_metrics(accuracy, precision, recall, f1, 'ver')

    def _save_transformer(self):
        # Save time steps for creating sequences in transformer
        self.transformer.seq_time_steps = self.seq_time_steps
        # Save Mahalanobis params from train
        self.transformer.mahalanobis_params = {"mean": self.mean, "cov": self.cov}
        self.transformer.threshold = self.threshold
        # Save the transformer
        self.transformer_name = "transformer.sav"
        self.transformer_path = os.path.join(self.model_storage, self.transformer_name)
        joblib.dump(self.transformer, self.transformer_path)

    def _save_model(self):
        """ Save the model and its artifacts. """
        try:
            if self.with_mlflow:
                print("Logging the model to MLflow ...")
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    signature = infer_signature(self.norm_train_x_seq, self.model.predict(self.norm_train_x_seq))
                    mlflow.tensorflow.log_model(self.model, artifact_path="model", signature=signature)
                else:
                    mlflow.tensorflow.log_model(self.model, artifact_path="model")
                print("Model saved successfully.")
            else:
                # Fallback mechanism if MLflow is not available
                self.model.save(os.path.join(self.model_storage, self.config.model.model_name))
                print("Model saved successfully.")
        except Exception as e:
            print("An error occurred while saving the model:", str(e))

    def _write_logs(self):
        """Log training results"""

        # ToDo: same log for tune or without tune. the hyperparams are modified directly in the respective class attributes
        # ToDo: log tn, fp, fn, tp as metrics
        # Log parameters: training parameters
        mlflow.log_params({
            "train_shape": self.norm_train_x_seq.shape,

            "seq_time_steps": self.seq_time_steps,
            "early_stopping_rounds": self.config.train.train_setup.early_stopping_rounds,
            "tuning": self.train_setup.tuning,

            # config.train.hyperparams
            "encoder_layers": self.hyperparams.encoder_layers,
            "decoder_layers": self.hyperparams.decoder_layers,
            "encoder_units": self.hyperparams.encoder_units,
            "decoder_units": self.hyperparams.decoder_units,
            "learning_rate": self.hyperparams.learning_rate,
            "dropout_rate": self.hyperparams.dropout_rate,
            "batch_size": self.hyperparams.batch_size,
            "epochs": self.hyperparams.epochs,
            "activation": self.hyperparams.activation,
            "regularization": self.hyperparams.regularization
        })

        # Log artifacts
        mlflow.log_artifact("src/configs/config.py")  # config file
        mlflow.log_artifact(self.transformer_path)  # data fitted transformer

        # Log tags: model name
        mlflow.set_tag("model_name", self.config.model.model_name)

        # Log metrics:
        mlflow.log_metric("threshold", self.threshold)  # learn anomaly threshold tau
        mlflow.log_metric("train_rec_loss", self.model_history.history["loss"][-1])  # reconstruction norm_train_x_seq
        mlflow.log_metric("val_rec_loss", self.model_history.history["val_loss"][-1])  # reconstruction norm_val_x_seq

    def run_train(self):
        """Training orchestration."""

        self._load_data()
        self._preprocess_data()

        if self.train_setup.tuning:
            # Uses hyperopt to tune and return best model
            self.hyperparams, self.model, self.model_history = self._tuning(model_spaces[self.train_setup.hp_space])
        else:
            self._build(self.hyperparams)
            self._train(self.hyperparams)

        self._eval()

        self._save_transformer()
        self._save_model()

        if self.with_mlflow:
            self._write_logs()
