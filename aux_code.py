"""
Hackathon - INCAP - IconPro GmbH
Timeseries Classification with Transformers
"""

from dataclasses import dataclass
import os
import numpy as np
from sklearn.model_selection import train_test_split
from lstm import lstm 
# Import packages as you need
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras_nlp.layers import SinePositionEncoding
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import pickle

from keras.callbacks import TensorBoard
from tensorflow.keras import layers
from datetime import datetime
from packaging import version
import time

NAME = "Model_TF-{}".format(int(time.time()))
tb_callback = TensorBoard(log_dir='logs/{}'.format(NAME))
# %load_ext tensorboard

def load_data(data_path):
    """
    Loading of the dataset provided
    Edit the code below
    """
    data=pd.read_pickle(data_path)
    return data


def preprocess_data(data):
    """
    A standard nan removal to be added.
    Add more preprocessing steps if needed.
    """
    data["labels"] = data["labels"].astype(int)
    data.dropna(inplace = True)
    return data.sample(frac=1)

def split_train_test(data):
    """
    Splitting the data into train, test, validation 
    """
    train, test = train_test_split(data,test_size=0.4, random_state=42,stratify=data['labels'])
    test, val = train_test_split(test,test_size=0.5, random_state=42, stratify=test['labels'])

    return train, test, val

def timeseries_transform(data):
    """
    Implement the timeseries transformer here
    """
    Y = np.array(data["labels"].astype(int))
    X = np.array(data["dim_0"].to_list())
    Y[Y==-1]=0
    N,D = X.shape
    X = X.reshape((N,D,1))
    
    return X, Y

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=4, activation="relu",padding='same')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x_ = SinePositionEncoding()(x)
    x= x + x_
    
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

def model_training(train):
    """
    Train the data with the compatible model
    """
    X_train, Y_train = train
    input_shape = X_train.shape[1:]
    model=build_model(
                        input_shape,
                        head_size=256,
                        num_heads=4,
                        ff_dim=4,
                        num_transformer_blocks=4,
                        mlp_units=[128],
                        mlp_dropout=0.4,
                        dropout=0.25,
                     )
    model.compile(
                    loss=keras.losses.BinaryCrossentropy(),
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    metrics=keras.metrics.BinaryAccuracy(),
    )
    # callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    # %tensorboard --logdir logs/fit
    model.fit(
    X_train,
    Y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[tb_callback,keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )
    
    
    return model



def metric(validation_data, model):
    """
    Standard metrics and plotting should be same
    Metrics should be computed on validation data(unseen data)
    1. Balanced accuracy score
    2. Confusion matrix
    3. Per-class accuracy
    """
    X_val, Y_val = validation_data
    Y_pred = model.predict(X_val)>0.5
    ba=balanced_accuracy_score(Y_val, Y_pred)
    cm=confusion_matrix(Y_val, Y_pred)
    # cm_display = ConfusionMatrixDisplay(cm).plot()
    metrics=[cm,ba]
    return metrics

def validation(metrics, metrics_validation):
    """
    Comparing the results with provided Series Embedder
    Plot confusion matrices of self analysis and LSTM with balanced_accuracy
    
    """
    cm_model = ConfusionMatrixDisplay(metrics[0]).plot()
    cm_lstm = ConfusionMatrixDisplay(metrics_validation[0]).plot()

if __name__=="__main__":
    path="./FordA/data.pkl"
    data=load_data(path)
    preprocessed_data=preprocess_data(data)
    
    train,test,val=split_train_test(preprocessed_data)
    train = train[0:1000]
    X_train,Y_train = timeseries_transform(train)
    X_test,Y_test = timeseries_transform(train)
    X_val,Y_val = timeseries_transform(val)
    
    
    
    # model_self=model_training((X_train,Y_train))
    # model_self.save("Pos_enc1")
    metrics=metric(val,model_self)
    
    lstm_cm,lstm_balanced_accuracy=lstm(preprocessed_data,target='labels')
    metrics_validation = [lstm_cm, lstm_balanced_accuracy]
    validation(metrics,metrics_validation)
    
    