import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt

import data_preprocessing


def plot_class_counts(target, class_names, title, save_path):
    """Plot counts of different target variable values.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        target (numpy.ndarray): data labels (target variable values).
        class_names (list): list of class names.
        title (str): bar plot tile.
        save_path (str): path for saving the class counts plot.
    """

    # Get class counts in target array.
    _, class_counts = np.unique(target, return_counts=True)

    # Normalize class counts.
    class_counts_normalized = class_counts / sum(class_counts)
    
    # Plot histogram of relative frequencies of class values.
    plt.bar(np.arange(len(class_names)), class_counts_normalized, align='center', alpha=0.5)
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.xticks(rotation=35)
    plt.ylabel('frequency')
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()


# Get preprocessed data.
data = data_preprocessing.get_preprocessed_dataset(dataset_id=2, window_size=154, overlap=0.5, deselect=[])


# Get non-segmented and segmented data class values.
target = data['target']
seg_target = data['seg_target']

# Get class names.
class_names = data['class_names']


# Plot class frequencies in non-segmented and segmented datasets.
plot_class_counts(target, class_names, "Dataset Class Frequencies", './plots/class_frequencies_unsegmented.eps')
plot_class_counts(seg_target, class_names, "Segmented Dataset Class Frequencies", './plots/class_frequencies_segmented.eps')

