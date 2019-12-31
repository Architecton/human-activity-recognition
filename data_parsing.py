import glob
import numpy as np
import xlrd
import csv


def parse_xlsx_file(file_path):
    """Parse contents of xlsx file into numpy array.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        file_path (str): path for the xlsx file.

    Returns:
        numpy.ndarray: parsed contents of the xlsx file.

    """

    # Open worksheet.
    worksheet = xlrd.open_workbook(file_path).sheet_by_index(0)

    # Parse data from worksheet.
    data = np.empty((worksheet.nrows-1, worksheet.ncols), dtype=object)
    for row in np.arange(1, worksheet.nrows):
        for col in np.arange(worksheet.ncols):
            data[row-1, col] = worksheet.cell_value(row, col)
    return data

def parse_csv_file(file_path):
    """Parse contents of csv file into numpy array.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        file_path (str): path for the csv file.

    Returns:
        numpy.ndarray: parsed contents of the csv file.

    """
    
    # Parse contents into array and return it.
    reader = csv.reader(open(file_path, "r"), delimiter=",")
    raw_data = list(reader)
    data = np.array(raw_data).astype(np.float)

    return data


def parse_acc_data(data_folder_path, dataset_num):
    """Parse contents of data files into one numpy array.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        data_folder_path (str): path of folder containing the dataset.

    Returns:
        numpy.ndarray: parsed contents of next raw data file.

    """

    if dataset_num == 1:
        # Go over xlsx files in data folder.
        for f in glob.glob(data_folder_path + '*.xlsx'):
            data_raw = parse_xlsx_file(f)
            data = data_raw[:, 2:-1]
            target = data_raw[:, -1]
            unlabeled_msk = target == ''
            target_final = target[~unlabeled_msk].astype(np.int)
            data_final = data[~unlabeled_msk, :].astype(np.float)
            yield data_final, target_final
    elif dataset_num == 2:
        # Go over .csv files in data folder.
        for f in glob.glob(data_folder_path + '*.csv'):
            data_raw = parse_csv_file(f)
            data = data_raw[:, 1:-1]
            target = data_raw[:, -1]
            yield data.astype(np.int), target.astype(np.int)



def concatenate_data(data_folder_path, dataset_num):
    """Concatenate data from different files of same dataset into
    one numpy array.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        data_folder_path (str): path of folder containing the dataset.
        dataset_num (int): index of dataset to parse.

    Returns:
        numpy.ndarray: parsed contents of dataset files.

    """

    data_master = None
    target_master = None
    first = True

    # Parse accelerometer data and concatenate into data matrix.
    for data, target in parse_acc_data(data_folder_path, dataset_num):
        if first:
            data_master = data
            target_master = target
            first = False
        else:
            data_master = np.vstack((data_master, data))
            target_master = np.hstack((target_master, target))

    return data_master, target_master


def get_data(data_folder_path):
    """Get numpy array representing the contents of specified datset.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        file_path (str): path for the xlsx file.

    Returns:
        numpy.ndarray: parsed contents of the xlsx file.

    Raises:
        ValueError

    """
    dataset_num = 1
    return concatenate_data(data_folder_path, dataset_num)


def get_data2(data_folder_path):
    """Parse contents of xlsx file into numpy array.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        file_path (str): path for the xlsx file.

    Returns:
        numpy.ndarray: parsed contents of the xlsx file.

    Raises:
        ValueError

    """
    dataset_num = 2
    return concatenate_data(data_folder_path, dataset_num)

