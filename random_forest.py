import numpy as np
import joblib
 
def pad_sequences(sequences, max_length):
    """
    Pads each sequence with 0s to make them the same length.
    """
    padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in sequences]
    return np.array(padded_sequences)
 
def predict2(sequences):
    """
    Loads the model and predicts the labels for the provided sequences.
    """
    # Load the trained model
    model = joblib.load('random_forest_modelnew.pkl')
   
    # Find the maximum length of sequences for padding
    max_length = 15

    # Pad the sequences to the maximum length
    padded_sequences = pad_sequences(sequences, max_length)
   
    # Make predictions using the model
    predictions = model.predict(padded_sequences)
   
    anomalous_sequences = []
    non_anomalous_sequences = []
    
    # Populate the lists based on predictions
    for seq, prediction in zip(sequences, predictions):
        if prediction == 1:
            anomalous_sequences.append(seq)
        else:
            non_anomalous_sequences.append(seq)
    
    return anomalous_sequences,non_anomalous_sequences
