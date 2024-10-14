#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
 
sys.path.append('../')
from loglizer import dataloader, preprocessing
 
struct_log = 'data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = 'data/HDFS/anomaly_label.csv'  # The anomaly label file
threshold = 0.1  # Threshold for anomaly detection based on reconstruction error
 
 
class AutoencoderModel:
 
    def __init__(self, input_dim):
        """
        Initialize the autoencoder model.
        Args:
            input_dim: The number of input features.
        """
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
 
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
 
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
 
    def fit(self, X_train, epochs=50, batch_size=32):
        """
        Train the autoencoder model on normal data.
        """
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1)
 
    def predict(self, X):
        """
        Get the reconstruction errors of the data.
        """
        reconstructed = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        return mse
 
    def evaluate(self, X, y_true, threshold):
        """
        Evaluate the model and print precision, recall, and F1 score.
        Args:
            X: Test data.
            y_true: True labels.
            threshold: Threshold for anomaly detection.
        """
        reconstruction_errors = self.predict(X)
        y_pred = (reconstruction_errors > threshold).astype(int)
 
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
 
        print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}')
        return precision, recall, f1
 
 
def autoEncoder(x_train, y_train, x_test):
    x_train_normal = x_train[y_train==0]
    model = AutoencoderModel(input_dim=x_train.shape[1])
    model.fit(x_train_normal, epochs=50, batch_size=64)
    res = model.predict(x_test)
    nom_res = np.where(res> 1, 1, 0)

    return nom_res


 
    