import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from dynamo_preprocess2 import generate_anomalous_sequences, generate_non_anomalous_sequences
import numpy as np
num_sequences = 100000
non_anomalous_sequences = generate_non_anomalous_sequences(num_sequences)
anomalous_sequences = generate_anomalous_sequences(100) + generate_non_anomalous_sequences(10)

max_length = 12

def pad_sequence(seq, max_length):
    return seq + [0] * (max_length - len(seq))

non_anomalous_sequences_padded = np.array([pad_sequence(seq, max_length) for seq in non_anomalous_sequences])


all_sequences = non_anomalous_sequences_padded / np.max(non_anomalous_sequences_padded)

X_train = all_sequences[:num_sequences]


input_layer = Input(shape=(max_length,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(max_length, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=8, shuffle=True)

autoencoder.save('model.h5')