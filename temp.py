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
anomalous_sequences_padded = np.array([pad_sequence(seq, max_length) for seq in anomalous_sequences])

all_sequences = np.concatenate((non_anomalous_sequences_padded, anomalous_sequences_padded), axis=0)

all_sequences = all_sequences / np.max(all_sequences)

X_train = all_sequences[:num_sequences]
X_test = all_sequences[num_sequences:]


input_layer = Input(shape=(max_length,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(max_length, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=8, shuffle=True)

reconstructed = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.abs(reconstructed - X_test), axis=1)

# threshold = np.percentile(reconstruction_error, 50)  
# predicted_anomalies = reconstruction_error > threshold

print(reconstruction_error)

count=0
for i in range(len(reconstruction_error)):
    if reconstruction_error[i]> 1.68505197e-03:
        count+=1
        print(anomalous_sequences[i])

# count=0
# for i, is_anomaly in enumerate(predicted_anomalies):
#     if is_anomaly:
#         print(f"Sequence {i} is anomalous: {anomalous_sequences[i]}")
#         count+=1
#     else:
#         print(f"Sequence {i} is non-anomalous: {anomalous_sequences[i]}")

print("Count: ", count)