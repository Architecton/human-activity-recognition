# human-activity-recognition
Implementations and evaluations of various approaches to human activity recognition from accelerometer data.

## Files contained in this repository
- **data\_parsing.py** contains functionality related to parsing data from the dataset folders,

- **data\_preprocessing.py** contains functionality related to preprocessing and segmenting the data
parsed from the dataset,

- **data\_stats.py**
contains functionality used to plot the class distributions for datasets,

- **evaluate.py**
is the main script used to implement the method evaluations,

- **get\_confustion\_matrices.py**
contains the functionality used to plot confusion matrices for different methods and datasets,

- **KClassifier.py**
contains a Wrapper for Keras classifier used for compatibility with imblearn resampling pipeline,

- **model\_params.py**
contains the functionality related to obtaining and setting the parameters of evaluated models,

- **models.py**
contains the functionality related to obtaining the models to be evaluated,

- **resampling.py**
contains the functionality related to resampling methods (uses imblearn),

- **engineer\_features.m**
is the MATLAB script used to perform feature engineering on the segmented dataset.
The data is loaded from the cached *mat* filed contained in the *cached-data* folder.


## How to run the evaluations

The evaluations are performed by the script evaluate.py. The script accepts two arguments - method
and dataset id. Valid arguments for the --method argument are *cnn* for the convolutional neural network
model, *lstm* for the long short-term memory neural network model, *fe* for the feature engineering
method and *rf* for the baseline random forest model.

Other evaluation parameters can be set by editing the script. The variables representing the
parameters are written in capitals using snake case and can be easily located at the start of the
script and in the sections corresponding to evaluations of individual methods.

An evaluation of a convolutional neural network model on the first dataset can be run as follows:

`Python3 evaluate.py --method cnn --dataset 1`

After the evaluation using cross-validation is finished, the mean cross-validation score for the specified dataset is
written to the *results/results.txt* file. The classification report is written to the *results/clf\_reports.txt* file.

