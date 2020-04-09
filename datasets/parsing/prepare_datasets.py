import numpy as np
from scipy import stats
import sklearn
import glob
import xlrd
import csv
import argparse


def parse_dataset(dataset_idx, window_size, shuffle, overlap):

    # Check if correct dataset specified.
    if dataset_idx not in [1, 2, 3]:
        raise ValueError("Unknown dataset selected")
    
    # Join, process and segment dataset. Return segments and labels.
    data_joined, target_joined = join_data(dataset_idx)
    data_processed, target_processed = process_data(data_joined, target_joined, dataset_idx)
    segments, labels = segment_data(data_processed, target_processed, window_size, overlap)
    if shuffle:
        return sklearn.utils.shuffle(segments, labels, random_state=0)
    else:
        return segments, labels


def join_data(dataset_idx):

    # Parse and join appropriate dataset.
    if dataset_idx == 1:

        # Set data folder path and initialize list for parts of dataset.
        data_folder_path = './data1/'
        data_aggr = []
        
        # Go over files constituting dataset and parse.
        for f in glob.glob(data_folder_path + '*.xlsx'):
            worksheet = xlrd.open_workbook(f).sheet_by_index(0)
            data = np.empty((worksheet.nrows-1, worksheet.ncols-1), dtype=object)
            for row in np.arange(1, worksheet.nrows):
                for col in np.arange(1, worksheet.ncols):
                    data[row-1, col-1] = worksheet.cell_value(row, col)
            
            # Add parsed dataset file contents to aggregate list.
            data_aggr.append(data[data[:, -1] != '', :])
        
        # Join data in aggregated list and return.
        data_joined = np.vstack(data_aggr)
        return data_joined[:, :-1], data_joined[:, -1]
    
    elif dataset_idx == 2:
        
        # Set data folder path and initialize list for parts of dataset.
        data_folder_path = './data2/'
        data_aggr = []

        # Go over files constituting dataset and parse.
        for f in glob.glob(data_folder_path + '*.csv'):
            reader = csv.reader(open(f, "r"), delimiter=",")
            raw_data = list(reader)
            raw_data = [[x for x in el if x != ''] for el in raw_data]
            raw_data = [el for el in raw_data if len(el) == 4]
            data_aggr.append(np.array(raw_data).astype(np.float))
        
        # Join data in aggregated list and return.
        data_joined = np.vstack(data_aggr)
        return data_joined[:, :-1], data_joined[:, -1]

    elif dataset_idx == 3:
        
        # Set data folder path.
        data_folder_path = './data3/'
        f = data_folder_path + 'actitracker.csv'

        # Parse dataset.
        reader = csv.reader(open(f, "r"), delimiter=",")
        raw_data = list(reader)
        raw_data = [[x for x in el if x != ''] for el in raw_data]
        raw_data = [el for el in raw_data if len(el) == 4]
        data = np.array(raw_data).astype(np.float)

        # Return data and labels.
        return data[:, :-1], data[:, -1]



def process_data(data_joined, target_joined, dataset_idx):

    # Process appropriate dataset.
    if dataset_idx == 1:
        return data_joined.astype(float), target_joined.astype(int)
    if dataset_idx == 2:
        return data_joined.astype(float), target_joined.astype(int)
    if dataset_idx == 3:
        return data_joined.astype(float), target_joined.astype(int)


def segment_data(data, target, window_size, overlap):

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
    
    # Return segments and their labels.
    return segments, labels


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--window-len-sec', type=float)
    group.add_argument('--window-len-samp', type=int)
    parser.add_argument("--shuffle", action='store_true', default=True)
    parser.add_argument("--overlap", type=float, required=True)
    args = parser.parse_args()
    #######################

    for dataset_idx in [1, 2, 3]:
    
        # Parse sampling frequency of specified dataset.
        with open('data' + str(dataset_idx) + '/fs.txt', 'r') as f:
            sampling_frequency = int(f.readline().strip())

        # Set number of seconds in each window and calculate window size.
        window_size = int(np.round(sampling_frequency*args.window_len_sec)) if args.window_len_sec else args.window_len_samp
        
        # Get parsed, processed and segmented data and labels.
        segments, labels = parse_dataset(dataset_idx, window_size=window_size, shuffle=args.shuffle, overlap=args.overlap)
        
        # Save segments and labels.
        np.save('segments' + str(dataset_idx) + '.npy', segments)
        np.save('labels' + str(dataset_idx) + '.npy', labels)

