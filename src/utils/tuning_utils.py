from hyperopt import hp
import numpy as np
from hyperopt.pyll.base import scope

# MS1
model_spaces = {
    "lstm-ae": {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.005)),
        'units': scope.int(hp.quniform('units', 80, 100, 6)),
        'batch_size': scope.int(hp.quniform('batch_size', 32, 64, 25)),
        'epochs': scope.int(hp.quniform('epochs', 50, 150, 50))
    }
}

# MS2
# model_spaces = {
#     "lstm-ae": {
#         'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
#         'units': scope.int(hp.quniform('units', 50, 150, 10)),
#         'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
#         'epochs': scope.int(hp.quniform('epochs', 100, 300, 50))
#     }
# }

# MS3
"""
    num_layers: The number of LSTM layers in the model. It is sampled from a uniform distribution between 1 and 3, 
    with steps of 1. This allows for exploring different depths of the LSTM architecture.

    dropout_rate: The dropout rate between LSTM layers. It is sampled from a uniform distribution between 0.0 and 0.5. 
    Dropout can help regularize the model and prevent overfitting.

    activation: The activation function used in the LSTM layers. It is chosen from a list of options, including 'relu' 
    and 'tanh'. Experimenting with different activation functions can affect the model's performance.

    regularization: The type of regularization applied to the LSTM layers. It is chosen from a list of options, 
    including None, 'l1', and 'l2'. Regularization techniques such as L1 or L2 regularization can help control model 
    complexity and prevent overfitting.

"""
# model_spaces = {
#     "lstm-ae": {
#         'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
#         'units': scope.int(hp.quniform('units', 50, 150, 10)),
#         'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
#         'epochs': scope.int(hp.quniform('epochs', 100, 300, 50)),
#         'num_layers': scope.int(hp.quniform('num_layers', 1, 3, 1)),
#         'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
#         'activation': hp.choice('activation', ['relu', 'tanh']),
#         'regularization': hp.choice('regularization', [None, 'l1', 'l2'])
#     }
# }


