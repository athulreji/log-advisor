import random
import numpy as np
import tensorflow as tf
from dynamo_preprocess2 import generate_anomalous_sequences, generate_non_anomalous_sequences
import numpy as np

# anomalous_sequences = generate_anomalous_sequences(100) + generate_non_anomalous_sequences(50000)
def pad_sequence(seq, max_length):
    return seq + [0] * (max_length - len(seq))

def predict(sequences):
    max_length = 12

    sequences_padded = np.array([pad_sequence(seq, max_length) for seq in sequences])

    all_sequences = sequences_padded / np.max(sequences_padded)

    X_test = all_sequences

    autoencoder = tf.keras.models.load_model('model.h5')

    reconstructed = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.abs(reconstructed - X_test), axis=1)

    # print(reconstruction_error)
    res = []
    # count=0
    for i in range(len(reconstruction_error)):
        if reconstruction_error[i]> 0.01:
            # count+=1
            # print(sequences[i])
            res.append(sequences[i])
    return res

# print(count)