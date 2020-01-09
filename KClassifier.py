
class KClassifier():
    """Wrapper for Keras model implementing the methods necessary
    for sklearn pipeline use.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Attributes:
        model (object): keras model to wrap.
        epoch (int): number of training epochs.
        batch_size (int): training batch size.
        shuffle (bool): if true, shuffle the training data.
    """

    def __init__(self, model, epochs, batch_size, shuffle=True):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle


    def fit(self, data, target):
        """Fit model to training data.

        Author: Jernej Vivod (vivod.jernej@gmail.com)

        Args:
            data (numpy.ndarray): data samples
            target (numpy.ndarray): data labels (target variable values)

        Returns:
            (object): reference to self.

        """
        self.model.fit(data, target, epochs=self.epochs, batch_size=self.batch_size, shuffle=self.shuffle, verbose=2)
        return self


    def predict(self, data):
        """Predict classes of unseen data.

        Author: Jernej Vivod (vivod.jernej@gmail.com)

        Args:
            data (numpy.ndarray): data samples

        Returns:
            (numpy.ndarray): class predictions for data samples. 

        """

        return self.model.predict_classes(data)


