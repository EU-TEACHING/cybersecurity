# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {

    "data": {
        "path_normal": "data/unsw-nb15/short_attack_normal/normal_short.csv",
        "path_anomaly": "data/unsw-nb15/short_attack_normal/attack_short.csv",
        "verification_set": "data/verification/UNSW-NB15_1.csv",
        "ground_truth_cols": ['label'],  # a list with names of columns or None
        "features": ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl",
                     "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb",
                     "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
                     "ct_srv_src",
                     "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
                     "is_ftp_login",
                     "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"],
        "data_types": ["Float32", "category", "category", "category", "Int64", "Int64", "Int64", "Int64",
                       "Int64", "Int64", "Float32", "Float32", "Int64", "Int64", "Float32", "Float32",
                       "Float32", "Float32", "Int64", "Int64", "Int64", "Int64", "Float32", "Float32", "Float32",
                       "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64",
                       "category", "Int64", "Int64", "Int64", "Int64", "category"],
        "n_rows": None,
        "normal_splits_pc": [0.7, 0.1, 0.1, 0.1],
        "anomaly_splits_pc": [0.5, 0.5]
    },
    "train": {
        "hyperparams": {
            "encoder_layers": 2,
            "decoder_layers": 2,
            "encoder_units": (100, 50),
            "decoder_units": (50, 100),
            "learning_rate": 0.0032265,
            "dropout_rate": 0.2,
            "batch_size": 25,
            "epochs": 150,
            "activation": "tanh",
            "regularization": None
        },
        "train_setup": {
            "seq_time_steps": 4,
            "early_stopping_rounds": 5,
            "tuning": True,
            "tuning_max_evals": 5,
            "hp_space": "lstm-ae-extra"
        }
    },
    "model": {
        "model_name": "LSTM-AE",
        "storage_path": "local_model_storage"
    },
    "mlflow_config": {
        "enabled": True,
        "experiment_name": "teaching-ADLM-v5",
    },
    "inference": {
        "data_path": "data/verification/UNSW-NB15_1.csv",
        "ground_truth_cols": ['label'],  # or None for new data
        "model_path": "local_model_storage/lstmae",
        "transformer_path": "local_model_storage/transformer.sav"
    }
}
