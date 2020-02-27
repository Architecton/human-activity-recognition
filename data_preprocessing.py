import numpy as np
import os
from scipy import stats
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import sklearn.utils

import data_parsing

def segment_data(data, target, window_size, overlap):
    """Segment data into overlaping windows.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        data (numpy.ndarray): data samples
        target (numpy.ndarray): data labels (target variable values)
        window_size (int): size of window used in segmenting.
        overlap (int): overlap of sequential windows used in segmenting.

    Returns:
        tuple: segmented data and data labels (target variable).

    """

    # Compute number of segments and preallocate arrays.
    num_segments = np.int(np.ceil((data.shape[0]-window_size+1)/max(np.int(np.round(window_size - window_size*overlap)), 1)))
    segments = np.empty((num_segments, window_size, data.shape[1]), dtype=float)
    labels = np.empty((num_segments), dtype=int)

    # Segment data and labels into overlapping windows.
    segment_idx = 0
    for idx in np.arange(0, data.shape[0]-window_size+1, max(np.int(np.round(window_size - window_size*overlap)), 1)):
        segments[segment_idx] = data[np.newaxis, idx:idx+window_size, :]
        labels[segment_idx] = stats.mode(target[idx:idx+window_size])[0][0]
        segment_idx += 1

    return segments, labels


def normalize_data(data):
    """Normalize data to mean 0 and standard deviation of 1.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        data (numpy.ndarray): data samples

    Returns:
        numpy.ndarray: normalized data samples.

    """

    # Normalize data.
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def encode_target(target):
    """Apply one-hot-encoding to encode target values.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        target (numpy.ndarray): target variable values.

    Returns:
        numpy.ndarray: one-hot-encoded target variable values.

    """

    # Encode data using one-hot-encoding.
    enc = OneHotEncoder(sparse=False, categories='auto')
    return enc.fit_transform(target.reshape(target.size, 1))


def get_preprocessed_dataset(dataset_id=1, window_size=-1.0, overlap=-1.0, deselect=[], shuffle=True):
    """Get dictionary containing data formatted for model evaluation,

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        dataset_id (int): id of dataset to use.
        window_size (int): size of the window used to segment the data.
        overlap (float): share of segment overlap.
        deselect (list): indices of classes to exclude from the dataset.

    Returns:
        dict: dictionary containing data formatted for model evaluation.

    Raises:
        ValueError

    """

    # Initialize flag that specifies whether to parse data from cache in folder or
    # from .xlsx files in data folder.
    parse = False
    cached_data_path = './cached-data/proc_seg_data' + str(dataset_id) + '.mat'
    if os.path.exists(cached_data_path):

        # Load cached data.
        loaded_data = sio.loadmat(cached_data_path)

        # Check if requested window size and overlap match that of cached data.
        if (loaded_data['window_size'][0] == window_size and 
                    loaded_data['overlap'][0] == overlap and 
                    loaded_data['shuffle'][0] == shuffle and
                    np.all(loaded_data['deselect'] == np.array(deselect)) or
                    window_size == -1.0 and 
                    overlap == -1.0 and 
                    deselect == []):
            
            # Parse contents of loaded cached data.
            segments = loaded_data['segments']
            seg_target_encoded = loaded_data['seg_target_encoded']
            seg_target = np.ravel(loaded_data['seg_target'])
            target = np.ravel(loaded_data['target'])
            class_names = np.ravel(loaded_data['class_names'])
            overlap = loaded_data['overlap'][0][0]
            window_size = loaded_data['window_size'][0][0]
            shuffle = loaded_data['shuffle'][0]
        else:
            parse = True
    else:
        parse = True
    if parse:
        
        # Parse class names for selected dataset.
        with open('./datasets/data' + str(dataset_id) + '/labels.txt', 'r') as f:
            class_names = [el.strip() for el in f.readlines()]

        # Else, parse data, preprocess, segment and persist.
        if dataset_id == 1:
            data, target = data_parsing.get_data1('./datasets/data' + str(dataset_id) + '/')
        elif dataset_id == 2:
            data, target = data_parsing.get_data2('./datasets/data' + str(dataset_id) + '/')
            data = data[target != 0, :]
            target = target[target != 0]
        elif dataset_id == 3:
            data, target = data_parsing.get_data3('./datasets/data' + str(dataset_id) + '/')
        else:
            raise ValueError('Unknown value of dataset index {0}'.format(dataset_id))

        # If deselecting any of the classes.
        if len(deselect) > 0:
            msk_deselection = np.in1d(target, deselect)
            target = target[~msk_deselection]
            data = data[~msk_deselection, :]
            class_names = [class_names[idx-1] for idx in np.arange(len(class_names)) if idx not in deselect]
        
        # Normalize data.
        data_normalized = normalize_data(data)

        # Segment data.
        WINDOW_SIZE = window_size
        OVERLAP = overlap
        segments, seg_target = segment_data(data_normalized, target, WINDOW_SIZE, OVERLAP)

        # If shuffling segments.
        if shuffle:
            segments, seg_target = sklearn.utils.shuffle(segments, seg_target)

        # encode target values using one-hot-encoding.
        seg_target_encoded = encode_target(seg_target)

        # Persist processed data.
        sio.savemat('./cached-data/proc_seg_data' + str(dataset_id) + '.mat', {'segments' : segments, 
            'seg_target_encoded' : seg_target_encoded, 
            'seg_target' : seg_target, 
            'target' : target,
            'window_size' : window_size,
            'overlap' : overlap,
            'deselect' : deselect,
            'class_names' : class_names,
            'shuffle' : shuffle
            })

    # Return dictionary of preprocessed data.
    return {'segments' : segments, 
            'seg_target_encoded' : seg_target_encoded, 
            'seg_target' : seg_target, 
            'target' : target, 
            'window_size' : window_size,
            'overlap' : overlap,
            'deselect' : deselect,
            'class_names' : class_names, 
            'shuffle' : shuffle
            }

