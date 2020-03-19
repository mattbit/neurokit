"""This module calculates the IES Suppression by taking average
    of the frontal electrodes """

import math
from scipy import ndimage as ndi
import numpy as np
import copy


def extract(recording, channels, threshold=8):
    """Extract the IES from the EEG

    Parameters
    ----------
    recording : neurokit.io.Recording
        The merged recording information in the form of a Recording
    channels : numpy.ndarray
        The channels to consider while calculating IES
    threshold: float, optional
        The threshold in μV

    Returns
    ----------
    ies : neurokit.io.Recording
        The extracted durations of Iso-Electric suppression in the  EEG
        defined as |(Fp1+Fp2)/2| < 8 μV
        The data has an additional IES column indicating whether the pulse is
        IES or not
    """
    mean_recording = np.absolute(recording.data.loc[:, channels].values.mean(axis=1))
    possible_ies = mean_recording < threshold
    ies_mask = ndi.morphology.binary_opening(possible_ies,
                                             structure=np.ones(math.ceil(recording.frequency)))
    ies = copy.deepcopy(recording)
    ies.data['IES'] = ies_mask
    return ies
