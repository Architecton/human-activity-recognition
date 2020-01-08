import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime

import data_preprocessing
import models
import model_params
import resampling
from KClassifier import KClassifier

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report

from imblearn.pipeline import Pipeline


def get_confusion_matrix(clf, data, target, clf_name, cm_save_path):
    """Plot and save confuction matrix for specified classifier.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        clf (object): classifier for which to plot the confuction matrix.
        data (numpy.ndarray): data samples
        target (numpy.ndarray): data labels (target variable values)
        clf_name (str): name of the classifier (used for plot labelling)
        cm_save_path (str): path for saving the confusion matrix plot

    """

    # Split data into training and test sets.
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=0)

    # Fit model.
    clf.fit(data_train, target_train)
    np.set_printoptions(precision=2)

    # Plot confusion matrix and save plot.
    disp = plot_confusion_matrix(clf, data_test, target_test if clf._kind == 'rf' else np.argmax(target_test, axis=1),
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')
    disp.ax_.set_title("Normalized Confusion Matrix - " + clf_name)
    disp.figure_.set_size_inches(9.0, 9.0, forward=True)
    plt.tight_layout()
    plt.savefig(cm_save_path)
    plt.clf()
    plt.close()


# Set resampling method ('none' means no resampling).
RESAMPLING_METHOD = 'random_oversampling'

# Get data.
data =  data_preprocessing.get_preprocessed_dataset()

segments = data['segments']
seg_target = data['seg_target']
seg_target_encoded = data['seg_target_encoded']
class_names = data['class_names']

data_fe = sio.loadmat('./data/data_fe/data1.mat')['data']
data_fe[np.isnan(data_fe)] = 0.0
target_fe = np.ravel(sio.loadmat('./data/data_fe/target1.mat')['target'])


#### GET MODEL CONFUSION MATRICES ####

# NN training parameters
EPOCHS_CNN = 10
BATCH_SIZE_CNN = 10

EPOCHS_LSTM = 10
BATCH_SIZE_LSTM = 10

### Initialize models. ###

# CNN
model_cnn = models.get_cnn_model(**model_params.get_params('cnn', n_rows=segments[0].shape[0], n_cols=segments[0].shape[1], num_classes=np.unique(seg_target).size))
clf_cnn = KClassifier(model_cnn, EPOCHS_CNN, BATCH_SIZE_CNN)
clf_cnn._estimator_type = 'classifier'

# LSTM
model_lstm = models.get_lstm_model(**model_params.get_params('lstm', n_rows=segments[0].shape[0], n_cols=segments[0].shape[1], num_classes=np.unique(seg_target).size))
clf_lstm = KClassifier(model_lstm, EPOCHS_LSTM, BATCH_SIZE_LSTM)
clf_lstm._estimator_type = 'classifier'

# RF
clf_rf = models.get_rf_model(**model_params.get_params('rf'))

# If resampling method specified, integrate into pipeline.
if RESAMPLING_METHOD != 'none':
    clf_cnn = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_cnn)])
    clf_lstm = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_lstm)])
    clf_rf = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_rf)])


# Set classifier identification properties.
clf_cnn._kind = 'cnn'
clf_lstm._kind = 'lstm'
clf_rf._kind = 'rf'

### Plot and save confusion matrices ###

# get_confusion_matrix(clf_cnn, segments[:, :, :, np.newaxis], seg_target_encoded, 'CNN',  './plots/conf_mat_cnn.svg')
# get_confusion_matrix(clf_lstm, segments, seg_target_encoded, 'LSTM', './plots/conf_mat_lstm.svg')
# get_confusion_matrix(clf_rf, np.array([el.flatten() for el in segments]), seg_target, 'Random Forest', './plots/conf_mat_rf.svg')
get_confusion_matrix(clf_rf, data_fe, target_fe, 'Random Forest - Engineered Features', './plots/conf_mat_fe.svg')

######################################

