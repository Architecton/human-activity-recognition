
def get_params(model_type, n_rows=0, n_cols=0, num_classes=0):
    """Get specified model parameters.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        model_type (str): the model for which to return the parameters.
        n_rows (int): number of rows of an input segment.
        n_cols (int): number of columns of an input segment.

    Returns:
        dict: dictionary containing the specified model parameters.
    
    Raises:
        ValueError

    """

    if model_type == 'cnn':
        # PARAMETERS FOR CNN MODEL
        return {
                'n_rows' : n_rows, 
                'n_cols' : n_cols,
                'num_classes' : num_classes,
                'num_filters' : 128,
                'kernel_size' : 1,
                'pool_window_size' : 1,
                'dropout_ratio' : 0.2,
                'num_neurons_fcl1' : 128,
                'num_neurons_fcl2' : 128,
                }
    elif model_type == 'rf':
        # PARAMETERS FOR RANDOM FOREST MODEL
        return {
                'n_estimators' : 100,
                'n_jobs' : -1,
                }
    elif model_type == 'lstm':
        # PARAMETERS FOR LSTM MODEL
        return {
                'n_rows' : n_rows,
                'n_cols' : n_cols,
                'num_classes' : num_classes,
                'dropout_ratio' : 0.2,
                'num_neurons_lstm1' : 128,
                'num_neurons_lstm2' : 128,
                }
    else:
        raise ValueError('Unknown model type specification {0}'.format(model_type))

