import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import pandas as pd
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt

def build_simple_model():

    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), 
                metrics=['mean_squared_error'])

    return model


def train_simple_model(model, X_frame, y_frame, split, data_epochs, batch_size, y_scaler):

    length = X_frame.shape[0]
    X_train = X_frame[:int(length*split),:]
    y_train = y_frame[:int(length*split)]

    X_test = X_frame[int(length*split):,:]
    y_test = y_frame[int(length*split):]

    model.fit(X_train, y_train, verbose=0, epochs=data_epochs,
                    batch_size=batch_size, validation_split=0.2)
    preds = model.predict(X_test, verbose=0)
    preds = y_scaler.inverse_transform(preds)
    y_test = y_scaler.inverse_transform(y_test)

    return mse(y_test, preds, squared=True)

