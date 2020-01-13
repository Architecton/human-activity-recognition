import numpy as np
import imblearn

def get_resampler(resampling_method):
    """Get class instance for class wrapping imblearn library resamplers
    to handle 2D samples.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        resampling_method (str): string specifying resampling method to use.

    Returns:
        object: class instance wrapping imblearn library resampler to handle 2D samples.

    """

    # Construct and return specified resampler.
    if resampling_method == 'random_undersampling':

        class RandomUnderSampler():
            def __init__(self):
                self.rus = imblearn.under_sampling.RandomUnderSampler()

            def fit(self, data, target):
                self.rus.fit(np.array([el.flatten() for el in data]), target)
                return self

            def fit_sample(self, data, target):
                shape_pre_flatten = data[0].shape
                data_rsmp, target_rsmp = self.rus.fit_sample(np.array([el.flatten() for el in data]), target)
                return np.array([np.reshape(el, shape_pre_flatten) for el in data_rsmp]), target_rsmp

            def fit_resample(self, data, target):
                return self.fit_sample(data, target)

        return RandomUnderSampler()
                
    elif resampling_method == 'random_oversampling':

        class RandomOversampler():
            def __init__(self):
                self.ros = imblearn.over_sampling.RandomOverSampler()

            def fit(self, data, target):
                self.ros.fit(np.array([el.flatten() for el in data]), target)
                return self

            def fit_sample(self, data, target):
                shape_pre_flatten = data[0].shape
                data_rsmp, target_rsmp = self.ros.fit_sample(np.array([el.flatten() for el in data]), target)
                return np.array([np.reshape(el, shape_pre_flatten) for el in data_rsmp]), target_rsmp

            def fit_resample(self, data, target):
                return self.fit_sample(data, target)

        return RandomOversampler()

    elif resampling_method == 'smote':

        class Smote():
            def __init__(self):
                self.smote = imblearn.over_sampling.SMOTE()

            def fit(self, data, target):
                self.smote.fit(np.array([el.flatten() for el in data]), target)
                return self

            def fit_sample(self, data, target):
                shape_pre_flatten = data[0].shape
                data_rsmp, target_rsmp = self.smote.fit_sample(np.array([el.flatten() for el in data]), target)
                return np.array([np.reshape(el, shape_pre_flatten) for el in data_rsmp]), target_rsmp

            def fit_resample(self, data, target):
                return self.fit_sample(data, target)

        return Smote()

    elif resampling_method == 'svmsmote':

        class SVMSmote():
            def __init__(self):
                self.svmsmote = imblearn.over_sampling.SVMSMOTE()

            def fit(self, data, target):
                self.svmsmote.fit(np.array([el.flatten() for el in data]), target)
                return self

            def fit_sample(self, data, target):
                shape_pre_flatten = data[0].shape
                data_rsmp, target_rsmp = self.svmsmote.fit_sample(np.array([el.flatten() for el in data]), target)
                return np.array([np.reshape(el, shape_pre_flatten) for el in data_rsmp]), target_rsmp

            def fit_resample(self, data, target):
                return self.fit_sample(data, target)

        return SVMSmote()


    elif resampling_method == 'smote_tomek':

        class SmoteTomek():
            def __init__(self):
                self.smotetmk = imblearn.combine.SMOTETomek()

            def fit(self, data, target):
                self.smotetmk.fit(np.array([el.flatten() for el in data]), target)
                return self

            def fit_sample(self, data, target):
                shape_pre_flatten = data[0].shape
                data_rsmp, target_rsmp = self.smotetmk.fit_sample(np.array([el.flatten() for el in data]), target)
                return np.array([np.reshape(el, shape_pre_flatten) for el in data_rsmp]), target_rsmp

            def fit_resample(self, data, target):
                return self.fit_sample(data, target)

        return SmoteTomek()

