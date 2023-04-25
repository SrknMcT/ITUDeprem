from math import ceil
import numpy as np
from pywt import wavedec
from pecnet.normalize import WindowNormalizer

def get_rolling_windows(data, window_size, fill=False, step=1, include_first=True):
    """
    Get rolling windows for the given data
     Parameters
    ----------
    data: List or ndarray
    window_size: int
    fill:
        If true, adds zero padding to the start of the data.First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]            
    """
    #TODO:  fill == False and include_first==False should not be in the same time.

    margin = 0
    if include_first == False:
        margin = 1
        data = data[:-1]

    if fill: # Fill with zeros to get exact same size
        data = np.insert(data, np.zeros((window_size - 1 + margin), dtype=int), 0.0)

    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)[::step, :].copy()
    return windows

def rolling_op(data, window_size, op, fill=True, step=1, include_first=True):
    """
    Get rolling operation for the given data
     Parameters
    ----------
    data: List or ndarray
    window_size: int
    op: Numpy function which takes axis parameter.
    fill:
        If true, adds zero padding to the start of the data. Output will be the same size as input.
        First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    """
    windows = get_rolling_windows(data, window_size, fill, step, include_first)
    results = op(windows, axis=-1)
    return results

def rolling_mean(data, window_size, fill=True, step=1, include_first=True): # == yy5
    """
    Get rolling mean for the given data
     Parameters
    ----------
    data: List or ndarray
    window_size: int
    fill:
        If true, adds zero padding to the start of the data. Output will be the same size as input.
        First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    """
    windows = get_rolling_windows(data, window_size, fill, step, include_first)
    means = np.nanmean(windows,axis=-1)
    return means

def rolling_max(data, window_size, fill=True, step=1):
    """
    Get rolling max for the given data
     Parameters
    ----------
    data: List or ndarray
    window_size: int
    fill:
        If true, adds zero padding to the start of the data. Output will be the same size as input.
        First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    """
    windows = get_rolling_windows(data, window_size, fill, step)
    maxs = np.nanmax(windows,axis=-1)
    return maxs

def wavelet_window(window, wavelet = 'haar', level=2):
    coeffs = wavedec(window, wavelet, mode='zero', level=level)
    return np.concatenate(coeffs)
    
def apply_wavelet(data, func=wavelet_window, wavelet = 'haar', level=2): #four_wavelets
    return np.apply_along_axis(func, 1, data, wavelet=wavelet, level=level)

def get_xy(arr, window_size, step=1, fill=False, include_first=True):
    """
    Get X and y windowized.
     Parameters
    ----------
    arr: List or ndarray
    window_size: int
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    fill:
        If true, adds zero padding to the start of the data.First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]

    """
    X = get_rolling_windows(arr, window_size, fill=fill, step=step, include_first=include_first)

    if fill:
        y = arr[1::step]
    else:
        y = arr[window_size::step]
    if len(X) != len(y):
        y = np.append(y, [np.nan])
    return X, y

def train_test(X,y, split_index):
    
    if split_index < 0 or split_index > len(X):
        raise ValueError(f"split_index={split_index} should be in range (0, len(X) )")
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

def label_train_test(y, split_index):
    if split_index < 0 or split_index > len(y):
        raise ValueError(f"split_index={split_index} should be in range (0, len(y) )")
    y_train = y[:split_index]
    y_test = y[split_index:]

    return y_train, y_test

def get_window_split_index(split_index, window_size, step, fill, start):
    """
    Get the window split index from split index. Use same values used for prepare_data
    Parameters
    ----------
    Return
    ----------
    Returns the window split index from deserved split index. 
    """
    return int(ceil((split_index - 1) / step)) - start if fill else int(ceil((split_index - window_size) / step) - start)


def prepare_data(input_data, window_size, mean_window_size, step=1, fill=False, normalize=True, start=0, include_first=True):
    """
    Applies all of the preprocessing steps. 
    In Wavelet cA2 is discarded.
    This function is just a wrapper for the process of data preparation. If the process is different you need to write your own prepare_data.
     Parameters
    ----------
    input_data: List or ndarray
    window_size: int
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    fill: boolean
        If true, adds zero padding to the start of the data.First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    normalize: boolean
    start: int
        Starting index of the windows. For input_data, start will be same as start*step.
    Return
    ----------
    Returns X,y. If normalize, also returns mean values.
    """
    # X,y = get_xy(input_data, window_size, step=step, fill=fill, include_first=include_first)
    # scaler = None
    # if normalize:
    #     scaler = WindowNormalizer()
    #     norm_X = scaler.fit_transform(X)
    #     print("mean: ",scaler.mean)
    #     norm_y = scaler.transform(y,False)
    #     X = norm_X
    #     y = norm_y

    
    X,y = get_xy(input_data, window_size, step=step, fill=fill, include_first=include_first)
    X2,y2 = get_xy(input_data, mean_window_size, step=step, fill=fill, include_first=include_first)
    if normalize:
        length2 = X2.shape[0]
        scaler2 = WindowNormalizer()
        scaler2.fit(X2)
        norm_X2 = scaler2.transform(X[:length2])
        norm_y2 = scaler2.transform(y[:length2],False)

        scaler = WindowNormalizer()
        scaler.fit(X)
        norm_X = scaler.transform(X)
        norm_y = scaler.transform(y,False)

        norm_X[:length2] = norm_X2
        norm_y[:length2] = norm_y2
        X = norm_X
        y = norm_y
    
    wavelet = apply_wavelet(X)
    X = wavelet[:,1:]
    if scaler:
        return X[start:], y[start:], scaler.mean[start:]

    return X[start:], y[start:]

def prepare_error_data(input_data, window_size, mean_window_size, step=1, fill=False, normalize=True, start=0, include_first=True):
    """
    Almost same as prepare_data. Only difference is last value is clipped. So there is no nan value in the end of the output.
    In Wavelet cA2 is discarded.
    This function is just a wrapper for the process of data preparation. If the process is different you need to write your own prepare_data.
     Parameters
    ----------
    input_data: List or ndarray
    window_size: int
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    fill: boolean
        If true, adds zero padding to the start of the data.First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    normalize: boolean
    start: int
        Starting index of the windows. For input_data, start will be same as start*step.
    Return
    ----------
    Returns X,y. If normalize, also returns mean values.
    """
    # X,y = get_xy(input_data, window_size, step=step, fill=fill)
    # scaler = None
    # if normalize:
    #     scaler = WindowNormalizer()
    #     norm_X = scaler.fit_transform(X)
    #     norm_y = scaler.transform(y,False)
    #     X = norm_X
    #     y = norm_y
    
    X,y = get_xy(input_data, window_size, step=step, fill=fill, include_first=include_first)
    X2,y2 = get_xy(input_data, mean_window_size, step=step, fill=fill, include_first=include_first)
    if normalize:
        length2 = X2.shape[0]
        scaler2 = WindowNormalizer()
        scaler2.fit(X2)
        norm_X2 = scaler2.transform(X[:length2])
        norm_y2 = scaler2.transform(y[:length2],False)

        scaler = WindowNormalizer()
        scaler.fit(X)
        norm_X = scaler.transform(X)
        norm_y = scaler.transform(y,False)

        norm_X[:length2] = norm_X2
        norm_y[:length2] = norm_y2
        X = norm_X
        y = norm_y
    
    wavelet = apply_wavelet(X)
    X = wavelet[:,1:]
    if scaler:
        return X[start:-1], y[start:-1], scaler.mean[start:-1]

    return X[start:-1], y[start:-1]

def prepare_and_split_data(input_data, window_size, mean_window_size, split_index=None, step=1, fill=False, normalize=True, start=0, window_split_index=None, include_first=True):
    """
    Applies all of the preprocessing steps. 
    In Wavelet cA2 is discarded.
    This function is just a wrapper for the process of data preparation. If the process is different you need to write your own prepare_data.
     Parameters
    ----------
    input_data: List or ndarray
    window_size: int
    split_index: int
        Index for dividing the data to train and test.
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    fill: boolean
        If true, adds zero padding to the start of the data.First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    normalize: boolean
    start: int
        Starting index of the windows. For input_data, start will be same as start*step.
    window_split_index: int
        Use it to override default window split index calculation
    Return
    ----------
    Returns X,y train and test pairs. If normalize, also returns mean values.
    """
    if not window_split_index:
        if not split_index:
            raise Exception("Arguments 'split_index' and 'window_split_index' are empty. At least one of them must be given to calculate the train-test split.")
        window_split_index = get_window_split_index(split_index, window_size, step, fill, start)

    if normalize:
        X, y, mean = prepare_data(input_data, window_size, mean_window_size, step, fill, normalize, start, include_first)
        return *train_test(X, y, window_split_index), mean

    X, y = prepare_data(input_data, window_size, mean_window_size, step, fill, normalize, start, include_first)
    return train_test(X, y, window_split_index)


def prepare_and_split_error_data(input_data, window_size, mean_window_size, split_index=None, step=1, fill=False, normalize=True, start=0, window_split_index=None):
    """
    Applies all of the preprocessing steps. 
    In Wavelet cA2 is discarded.
    This function is just a wrapper for the process of data preparation. If the process is different you need to write your own prepare_data.
     Parameters
    ----------
    input_data: List or ndarray
    window_size: int
    split_index: int
        Index for dividing the data to train and test.
    step:
        Used in the windowization process. Distance between consequent windows. 
        Ex: window_size = 4, step = 1 [1,2,3,4,5,6] -> [[1,2,3,4],[2,3,4,5], ...]
        step = 2 [[1,2,3,4],[3,4,5,6], ...]
    fill: boolean
        If true, adds zero padding to the start of the data.First window will be all zeros except the last value will be the first value of the input_data.
        Ex: window_size = 4, step = 1, [1,2,3,4,5,6] -> [[0,0,0,1], [0,0,1,2], ...]
    normalize: boolean
    start: int
        Starting index of the windows. For input_data, start will be same as start*step.
    window_split_index: int
        Use it to override default window split index calculation
    Return
    ----------
    Returns X,y train and test pairs. If normalize, also returns mean values.
    """
    if not window_split_index:
        if not split_index:
            raise Exception("Arguments 'split_index' and 'window_split_index' are empty. At least one of them must be given to calculate the train-test split.")
        window_split_index = get_window_split_index(split_index, window_size, step, fill, start)

    if normalize:
        X, y, mean = prepare_error_data(input_data, window_size, mean_window_size, step, fill, normalize, start)
        return *train_test(X, y, window_split_index), mean

    X, y = prepare_error_data(input_data, window_size, mean_window_size, step, fill, normalize, start)
    return train_test(X, y, window_split_index)