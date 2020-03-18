import pandas as pd
import math
from scipy import ndimage as ndi
import numpy as np


def extract(signal, channels, threshold = 8):
    """ Extract the IES from the EEG

    Parameters
    ----------
    signal : Recording 
        The merged signal information in the form of a Recording
    channles : array
		The channles to consider while calculating IES 
    threshold: int, optional
		The threshold in μ Volts
    

    Returns
    ----------
    ies : Recording
        The extracted durations of Iso-Electric suppressions in the  EEG
        defined as |(Fp1+Fp2)/2| < 8 μV
        The data has an additional IES column indicating whether the pulse is
        IES or not
    """
    mean_signal = np.absolute(signal.data.loc[:, channels].values.mean(axis=1))
    mean_signal = mean_signal < threshold 
    possible_ies = ndi.morphology.binary_opening(mean_signal, structure =
                                                 np.ones(math.ceil(signal.frequency)))
    signal.data['IES'] = possible_ies
    return signal