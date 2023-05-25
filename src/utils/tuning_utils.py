from hyperopt import hp
import numpy as np
from hyperopt.pyll.base import scope

model_spaces = {
    "lstm-ae": {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.005)),
        'units': scope.int(hp.quniform('units', 80, 100, 6)),
        'batch_size': scope.int(hp.quniform('batch_size', 32, 64, 25)),
        'epochs': scope.int(hp.quniform('epochs', 50, 150, 50))
    },
    "lstm-ae-auto": {
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
        'units': scope.int(hp.quniform('units', 50, 150, 10)),
        'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
        'epochs': scope.int(hp.quniform('epochs', 100, 300, 50))
    },
    # "lstm-ae-extra": {
    #     'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
    #     'units': scope.int(hp.quniform('units', 50, 150, 10)),
    #     'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
    #     'epochs': scope.int(hp.quniform('epochs', 100, 300, 50)),
    #     'num_layers': scope.int(hp.quniform('num_layers', 1, 3, 1)),
    #     'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
    #     'activation': hp.choice('activation', ['relu', 'tanh']),
    #     'regularization': hp.choice('regularization', [None, 'l1', 'l2'])
    # },

    "lstm-ae-extra": {
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),
        'encoder_layers': scope.int(hp.quniform('encoder_layers', 1, 2, 1)),
        'decoder_layers': scope.int(hp.quniform('decoder_layers', 1, 2, 1)),
        'encoder_units': [scope.int(hp.quniform(f'encoder_units_{i}', 50, 150, 10)) for i in range(1, 3)],
        'decoder_units': [scope.int(hp.quniform(f'decoder_units_{i}', 50, 150, 10)) for i in range(1, 3)],
        'batch_size': scope.int(hp.quniform('batch_size', 16, 128, 16)),
        'epochs': scope.int(hp.quniform('epochs', 1, 6, 1)),
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
        'activation': hp.choice('activation', ['relu', 'tanh']),
        'regularization': hp.choice('regularization', [None, 'l1', 'l2'])
    }

}

"""
    "lstm-ae-extra"
    
        learning_rate: Uniformly distributed between 0.0001 and 0.01. This controls the step size during training.

    units: Quantized uniform distribution between 50 and 150 with a step size of 10. This determines the number of units
     (neurons) in each LSTM layer.

    batch_size: Quantized uniform distribution between 16 and 128 with a step size of 16. This defines the number of 
    samples per gradient update during training.

    epochs: Quantized uniform distribution between 100 and 300 with a step size of 50. This specifies the number of 
    times the entire training dataset is passed through the model during training.

    num_layers: Quantized uniform distribution between 1 and 3 with a step size of 1. This sets the number of LSTM 
    layers in your model.

    dropout_rate: Uniformly distributed between 0.0 and 0.5. This controls the dropout rate, which helps prevent
     overfitting.

    activation: Choice between 'relu' and 'tanh'. This determines the activation function used in the LSTM layers.

    regularization: Choice between None, 'l1', and 'l2'. This specifies the type of regularization to apply, if any.

"""
