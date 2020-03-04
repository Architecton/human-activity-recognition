import numpy as np
import scipy.io as sio
import os
import datetime
import argparse

import data_preprocessing
import models
import model_params
import resampling
from KClassifier import KClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from imblearn.pipeline import Pipeline

# Parse evaluation parameters (dataset id and methods to evaluate).
parser = argparse.ArgumentParser(description='Human activity recognition method evaluation')
parser.add_argument('--method', type=str, metavar='METHOD', nargs='+', default=['rf', 'cnn', 'lstm', 'fe'], 
        choices=['rf', 'cnn', 'lstm', 'fe'], help='method to evaluate (rf, cnn, lstm or fe)')
parser.add_argument('--eval-method', type=str, metavar='EVAL_METHOD', default='cv', 
        choices=['tts', 'cv'], help='evaluation method (tts or cv)')
parser.add_argument('--dataset', type=int, metavar='DATASET', default=1, 
        choices=[1, 2, 3], help='dataset id (1, 2 or 3)')

# Parse arguments.
args = parser.parse_args()

# Set CV parameters.
N_SPLITS = 10
N_REPEATS = 1

# Set resampling method ('none' means no resampling).
RESAMPLING_METHOD = 'none'

# List of models to evaluate.
EVALUATE = args.method

#### (1) DATA PARSING AND PREPROCESSING ############

# Specify dataset id.
DATASET_ID = args.dataset

# Specify indices of features to deselect.
DESELECT = []

# Parse sampling frequency for selected dataset.
with open('./datasets/data' + str(DATASET_ID) + '/fs.txt', 'r') as f:
    sampling_frequency = int(f.readline().strip())

# Set number of seconds in each window and calculate window size.
NUM_SECONDS_WINDOW = 3
WINDOW_SIZE = sampling_frequency*NUM_SECONDS_WINDOW
SHUFFLE = True

# Get preprocessed data.
data = data_preprocessing.get_preprocessed_dataset(dataset_id=DATASET_ID, window_size=WINDOW_SIZE, overlap=0.3, deselect=DESELECT, shuffle=SHUFFLE)
segments = data['segments']
seg_target = data['seg_target']
seg_target_encoded = data['seg_target_encoded'] 
deselect_len = len(data['deselect'])
class_names = data['class_names']

# Save segments and target values (for sharing)
np.save('./data-share/segments_' + str(DATASET_ID) + '.npy', segments)
np.save('./data-share/class_' + str(DATASET_ID) + '.npy', seg_target)

####################################################

# Set path for classification reports file.
CLF_REPORTS_PATH = './results/clf_reports.txt'

def format_clf_report(clf_report, clf_name, class_names, dataset_id, save_path): 
    """Print classification statistics to file.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        clf_report (numpy.ndarray): matrix containing the classification statistics to print.
        clf_name (str): name of the classifier.
        class_names (list): list of class names.
        save_path (str): path of the file in which to print the statistics.
    """

    with open(save_path, 'a') as f:
        if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            f.write('\n')

        f.write('Date: {0}\n'.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        f.write('Model: {0}\n'.format(clf_name))
        f.write('Window size: {0}\n'.format(data['window_size']))
        f.write('Overlap: {0}\n'.format(data['overlap']))
        f.write('Deselected: {0}\n'.format(data['deselect']))
        f.write('Dataset: {0}\n'.format(dataset_id))
        f.write('\n')
        for idx in np.arange(len(class_names)-1):
            f.write('{0} '.format(class_names[idx]))
        f.write('{0}\n'.format(class_names[-1]))
        
        f.write('Precision  ')
        for idx in np.arange(clf_report.shape[1]-1):
            f.write('{0:.4f} '.format(clf_report[0, idx]))
        f.write('{0:.2f}\n'.format(clf_report[0, -1]))

        f.write('Recall  ')
        for idx in np.arange(clf_report.shape[1]-1):
            f.write('{0:.4f} '.format(clf_report[1, idx]))
        f.write('{0:.2f}\n'.format(clf_report[1, -1]))

        f.write('F-Score  ')
        for idx in np.arange(clf_report.shape[1]-1):
            f.write('{0:.4f} '.format(clf_report[2, idx]))
        f.write('{0:.4f}\n'.format(clf_report[2, -1]))

        f.write('Support  ')
        for idx in np.arange(clf_report.shape[1]-1):
            f.write('{0:.4f} '.format(clf_report[3, idx]))
        f.write('{0:.4f}\n'.format(clf_report[3, -1]))


#### (2) EVALUATE BASELINE RANDOM FOREST ##########


# If evaluating RF model.
if 'rf' in EVALUATE:

    # Initialize random forest model with specified parameters.
    clf_rf = models.get_rf_model(**model_params.get_params('rf'))

    # If resampling method specified, integrate into pipeline.
    if RESAMPLING_METHOD != 'none':
        clf_rf = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_rf)])
  
    # If evaluating using cross-validation.
    if args.eval_method == 'cv':

        # Initialize accumulator for fold results.
        score_acc_rf = 0

        # Initilize array for accumulating classification scoring reports.
        cr_rf = np.zeros((4, len(class_names)))

        # Perform CV.
        idx_it = 1
        for train_idx, test_idx in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS).split(segments, seg_target):
            segments_train = np.array([el.flatten() for el in segments[train_idx, :, :]])
            segments_test = np.array([el.flatten() for el in segments[test_idx, :, :]])
            seg_target_train = seg_target[train_idx]
            seg_target_test = seg_target[test_idx]
            clf_rf.fit(segments_train, seg_target_train)
            pred_test = clf_rf.predict(segments_test)
            score_acc_rf += accuracy_score(seg_target[test_idx], pred_test)
            cr_rf = cr_rf + np.array(precision_recall_fscore_support(seg_target_test, pred_test, labels=list(np.arange(1,len(class_names)+1))))
            print("RF - finished {0}/{1}".format(idx_it, N_SPLITS*N_REPEATS))
            idx_it += 1

        # Get mean fold score for RF model.
        cv_score_rf = score_acc_rf / (N_SPLITS*N_REPEATS)

        # Get mean classification report for RF model.
        cv_cr_rf = cr_rf / (N_SPLITS*N_REPEATS)

        # Write classification scoring report.
        format_clf_report(cv_cr_rf, "Random Forest", class_names, DATASET_ID, CLF_REPORTS_PATH)

    else:

        # Split data into training and test sets.
        data_train, data_test, target_train, target_test = train_test_split(np.array([el.flatten() for el in segments]), seg_target, random_state=0, 
                stratify=seg_target)

        # Train model and evaluate on test set.
        score = clf_rf.fit(data_train, target_train).score(data_test, target_test)
        print("Finised evaluating RF model using a train-test split with score={0:.4f}.".format(score))


####################################################


#### (3) EVALUATE CNN ##############################

# If evaluating CNN model.
if 'cnn' in EVALUATE:

    # Set CNN model training parameters.
    BATCH_SIZE_CNN = 32
    EPOCHS_CNN = 10

    # Initialize CNN model with specified parameters.
    model_cnn = models.get_cnn_model(**model_params.get_params('cnn', n_rows=segments[0].shape[0], n_cols=segments[0].shape[1], num_classes=np.unique(seg_target).size))
    clf_cnn = KClassifier(model_cnn, EPOCHS_CNN, BATCH_SIZE_CNN)

    # If resampling method specified, integrate into pipeline.
    if RESAMPLING_METHOD != 'none':
        clf_cnn = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_cnn)])

    # If evaluating using cross-validation.
    if args.eval_method == 'cv':

        # Initialize accumulator for fold results.
        scores_acc_cnn = 0

        # Initilize array for accumulating classification scoring reports.
        cr_cnn = np.zeros((4, len(class_names)))

        # Perform CV.
        idx_it = 1
        for train_idx, test_idx in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS).split(segments, seg_target):
            segments_train = segments[train_idx, :, :, np.newaxis] 
            segments_test = segments[test_idx, :, :, np.newaxis] 
            seg_target_encoded_train = seg_target_encoded[train_idx, :]
            seg_target_encoded_test = seg_target_encoded[test_idx, :]
            clf_cnn.fit(segments_train, seg_target_encoded_train)
            pred_test = clf_cnn.predict(segments_test)
            scores_acc_cnn += accuracy_score(seg_target[test_idx] - deselect_len, pred_test+1)
            cr_cnn = cr_cnn + np.array(precision_recall_fscore_support(np.argmax(seg_target_encoded_test, axis=1), pred_test, labels=list(np.arange(len(class_names)))))
            print("CNN - finished {0}/{1}".format(idx_it, N_SPLITS*N_REPEATS))
            idx_it += 1

        # Get mean fold score for CNN model.
        cv_score_cnn = scores_acc_cnn / (N_SPLITS*N_REPEATS)

        # Get mean classification report for CNN model.
        cv_cr_cnn = cr_cnn / (N_SPLITS*N_REPEATS)

        # Write classification scoring report.
        format_clf_report(cv_cr_cnn, "CNN", class_names, DATASET_ID, CLF_REPORTS_PATH)

    else:

        # Split data into training and test sets.
        data_train, data_test, target_train, target_test = train_test_split(segments, seg_target_encoded, random_state=0, 
                stratify=np.argmax(seg_target_encoded, axis=1))

        # Train model and evaluate on test set.
        score = clf_cnn.fit(data_train[:, :, :, np.newaxis], target_train).score(data_test[:, :, :, np.newaxis], np.argmax(target_test, axis=1))
        print("Finised evaluating CNN model using a train-test split with score={0:.4f}.".format(score))

####################################################


#### (4) EVALUATE LSTM NEURAL NETWORK ##############

# If evaluating LSTM model.
if 'lstm' in EVALUATE:
        
    # Set LSTM model training parameters.
    BATCH_SIZE_LSTM = 32
    EPOCHS_LSTM = 50

    # Initialize LSTM model with specified parameters.
    model_lstm = models.get_lstm_model(**model_params.get_params('lstm', n_rows=segments[0].shape[0], n_cols=segments[0].shape[1], num_classes=np.unique(seg_target).size))
    clf_lstm = KClassifier(model_lstm, EPOCHS_LSTM, BATCH_SIZE_LSTM, False)

    # If resampling method specified, integrate into pipeline.
    if RESAMPLING_METHOD != 'none':
        clf_lstm = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_lstm)])

    # If evaluating using cross-validation.
    if args.eval_method == 'cv':

        # Initialize accumulator for fold results.
        scores_acc_lstm = 0

        # Initilize array for accumulating classification scoring reports.
        cr_lstm = np.zeros((4, len(class_names)))

        # Perform CV.
        idx_it = 1
        for train_idx, test_idx in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS).split(segments, seg_target):
            segments_train = segments[train_idx, :, :] 
            segments_test = segments[test_idx, :, :] 
            seg_target_encoded_train = seg_target_encoded[train_idx, :]
            seg_target_encoded_test = seg_target_encoded[test_idx, :]
            clf_lstm.fit(segments_train, seg_target_encoded_train)
            pred_test = clf_lstm.predict(segments_test)
            scores_acc_lstm += accuracy_score(seg_target[test_idx] - deselect_len, pred_test+1)
            cr_lstm = cr_lstm + np.array(precision_recall_fscore_support(np.argmax(seg_target_encoded_test, axis=1), pred_test, labels=list(np.arange(len(class_names)))))
            print("LSTM - finished {0}/{1}".format(idx_it, N_SPLITS*N_REPEATS))
            idx_it += 1
        
        # Get mean fold score for LSTM model.
        cv_score_lstm = scores_acc_lstm / (N_SPLITS*N_REPEATS)

        # Get mean classification report for LSTM model.
        cv_cr_lstm = cr_lstm / (N_SPLITS*N_REPEATS)

        # Write classification scoring report.
        format_clf_report(cv_cr_lstm, "LSTM", class_names, DATASET_ID, CLF_REPORTS_PATH)

    else:

        # Split data into training and test sets.
        data_train, data_test, target_train, target_test = train_test_split(segments, seg_target_encoded, random_state=0, 
                stratify=np.argmax(seg_target_encoded, axis=1))
        
        # Train model and evaluate on test set.
        score = clf_lstm.fit(data_train, target_train).score(data_test, np.argmax(target_test, axis=1))
        print("Finised evaluating LSTM model using a train-test split with score={0:.4f}.".format(score))


####################################################


#### (5) EVALUATE CLASSIFIERS ON ENG. FEATUERES ####

# If evaluating feature engineering method:
if 'fe' in EVALUATE:

    # Parse data.
    data_fe = sio.loadmat('./datasets/data_fe/data' + str(DATASET_ID) + '.mat')['data']
    target_fe = np.ravel(sio.loadmat('./datasets/data_fe/target' + str(DATASET_ID) + '.mat')['target'])
    data_fe[np.isnan(data_fe)] = 0.0
     
    # Initialize random forest model with specified parameters.
    clf_rf = models.get_rf_model(**model_params.get_params('rf'))

    # If resampling method specified, integrate into pipeline.
    if RESAMPLING_METHOD != 'none':
        clf_rf = Pipeline([('resampler', resampling.get_resampler(RESAMPLING_METHOD)), ('clf', clf_rf)])

    # If evaluating using cross-validation.
    if args.eval_method == 'cv':
        
        # Initialize accumulator for fold results.
        score_acc_fe = 0

        # Initilize array for accumulating classification scoring reports.
        cr_fe = np.zeros((4, len(class_names)))

        # Perform CV.
        idx_it = 1
        for train_idx, test_idx in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS).split(data_fe, target_fe):
            data_train = data_fe[train_idx, :]
            data_test = data_fe[test_idx, :]
            target_train = target_fe[train_idx]
            target_test = target_fe[test_idx]
            clf_rf.fit(data_train, target_train)
            pred_test = clf_rf.predict(data_test)
            score_acc_fe += accuracy_score(target_test, pred_test)
            cr_fe = cr_fe + np.array(precision_recall_fscore_support(target_test, pred_test, labels=list(np.arange(1,len(class_names)+1))))
            print("FE - finished {0}/{1}".format(idx_it, N_SPLITS*N_REPEATS))
            idx_it += 1

        # Get mean fold score for RF model.
        cv_score_fe = score_acc_fe / (N_SPLITS*N_REPEATS)

        # Get mean classification report for feature engineering method.
        cv_cr_fe = cr_fe / (N_SPLITS*N_REPEATS)

        # Write classification scoring report.
        format_clf_report(cv_cr_fe, "Feature Engineering", class_names, DATASET_ID, CLF_REPORTS_PATH)

    else:

        # Split data into training and test sets.
        data_train, data_test, target_train, target_test = train_test_split(data_fe, target_fe, random_state=0, 
                stratify=target_fe)
        
        # Train model and evaluate on test set.
        score = clf_rf.fit(data_train, target_train).score(data_test, target_test)
        print("Finised evaluating FE+RF model using a train-test split with score={0:.4f}.".format(score))


####################################################


#### SAVE RESULTS TO FILE ##########################

if args.eval_method == 'cv':

    # Set path to results file.
    RESULTS_PATH = './results/results.txt'

    # Open results file and append results.
    with open(RESULTS_PATH, 'a') as f:
        if os.path.isfile(RESULTS_PATH) and os.path.getsize(RESULTS_PATH) > 0:
            f.write('\n')

        f.write('Date: {0}\n'.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        f.write('Window size: {0}\n'.format(data['window_size']))
        f.write('Overlap: {0}\n'.format(data['overlap']))
        f.write('Deselected: {0}\n'.format(data['deselect']))
        f.write('Dataset: {0}\n'.format(DATASET_ID))
        f.write('\n')
        f.write('Model | CV Score\n')
        f.write('----------------\n')
        if 'rf' in EVALUATE:
            f.write('RF    | {0:.4f}\n'.format(cv_score_rf))
        if 'cnn' in EVALUATE:
            f.write('CNN   | {0:.4f}\n'.format(cv_score_cnn))
        if 'lstm' in EVALUATE:
            f.write('LSTM  | {0:.4f}\n'.format(cv_score_lstm))
        if 'fe' in EVALUATE:
            f.write('FE  | {0:.4f}\n'.format(cv_score_fe))

####################################################

