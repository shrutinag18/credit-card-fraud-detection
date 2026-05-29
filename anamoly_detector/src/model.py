from keras import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

def build_autoencoder(seq_len, n_features):
    model = Sequential([
        # Encoder
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=False),
        
        # Bottleneck
        RepeatVector(seq_len),
        
        # Decoder
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])

    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model