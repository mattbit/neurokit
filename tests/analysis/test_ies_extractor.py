from unittest import TestCase

from neurokit.io import edf
from neurokit.analysis import ies_extractor
import pandas as pd
import numpy as np

class TestIESExtractor(TestCase):
    def test_ies(self):
        test_signal = np.zeros(128)
        test_signal[64:] = 10
        test_bool = test_signal < 10
        df = pd.DataFrame(data = {'EEG L1(Fp1)': test_signal, 'EEG R1(Fp2)':
                                  test_signal})
        channels = ('EEG L1(Fp1)','EEG R1(Fp2)')
        test_obj = Recording(df,channels,64)
        s = ies_extractor.extract(test_obj, channels, 8)
        final_bool = s.data['IES']
        print(np.array_equal(final_bool,test_bool))
        self.assertTrue(np.array_equal(final_bool, test_bool))



class Recording :
    """ Stub Class """
    def __init__(self, data,channels,frequency):
        self.data = data
        self.channels = channels
        self.frequency = frequency

