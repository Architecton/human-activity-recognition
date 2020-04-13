from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM
from tensorflow.keras import optimizers

from sklearn.ensemble import RandomForestClassifier


def get_cnn_model(n_rows, n_cols, num_classes, num_filters=128, kernel_size=2, 
        pool_window_size=2, dropout_ratio=0.2, num_neurons_fcl1=128, num_neurons_fcl2=128, init='uniform'):
    """Get CNN Keras model used for evaluations.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        n_rows (int): number of rows in a data segment.
        n_cols (int): number of columns in a data segment.
        num_classes (int): number of different classes in the dataset.
        num_filters (int): number of filters.
        kernel_size (int): kernel size.
        pool_window_size (int): pooling window size.
        dropout_ratio (float): dropout ratio.
        num_neurons_fcl1 (int): number of neurons in the first dense layer.
        num_neurons_fcl2 (int): number of neurons in the second dense layer.
        init (str): kernel initialization method.

    Returns:
        (object): initialized and compiled model.
    """

    # Set model.
    model = Sequential()
    
    # Set model topology.
    model.add(Conv2D(num_filters, (kernel_size, kernel_size), input_shape=(n_rows, n_cols, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool_window_size, pool_window_size), padding='valid'))
    model.add(Dropout(dropout_ratio))
    model.add(Flatten())
    model.add(Dense(num_neurons_fcl1, activation='relu'))
    model.add(Dense(num_neurons_fcl2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.Adam(lr = 0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def get_lstm_model(n_rows, n_cols, num_classes, dropout_ratio=0.2, num_neurons_lstm1=128, 
        num_neurons_lstm2=128):
    """Get LSTM Keras model used for evaluations.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        n_rows (int): number of rows in a data segment.
        n_cols (int): number of columns in a data segment.
        num_classes (int): number of different classes in the dataset.
        num_filters (int): number of filters.
        kernel_size (int): kernel size.
        pool_window_size (int): pooling window size.
        dropout_ratio (float): dropout ratio.
        num_neurons_lstm1 (int): number of neurons in the first LSTM layer.
        num_neurons_lstm2 (int): number of neurons in the second LSTM layer.

    Returns:
        (object): initialized and compiled model.
    """

    # Set model.
    model = Sequential()

    # Set model topology.
    model.add(CuDNNLSTM(num_neurons_lstm1, input_shape = (n_rows, n_cols), return_sequences = True))
    # model.add(LSTM(num_neurons_lstm1, input_shape = (n_rows, n_cols), return_sequences = True))
    model.add(Dropout(dropout_ratio))
    # model.add(LSTM(num_neurons_lstm2)) 
    model.add(CuDNNLSTM(num_neurons_lstm2)) 


    model.add(Dense(num_classes, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model


def get_rf_model(n_estimators=100, n_jobs=1):
    """Get Random Forest Scikit-Learn model used as the baseline in the evaluation.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        n_estimators (int): number of estimators to use.
        n_jobs (int): number of jobs to run in parallel.

    Returns:
        (object): initialized model.
    """

    # Initialize model
    model = RandomForestClassifier(n_estimators=n_estimators)
    return model

