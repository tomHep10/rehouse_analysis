import os
import numpy as np
import unittest
import bidict
import preprocessor as preprocessor
import shutil
from lfp.tests import utils

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745

OUTPUT_DIR = os.path.join("lfp", "tests", "output")


class test_lfp_recording_preprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    def test_map_to_region(self):
        traces = utils.load_test_traces()
        self.assertIs(type(traces), np.ndarray)

        brain_regions, sorted_channels = preprocessor.map_to_region(SUBJECT_DICT)
        # brain regions is a bidict: {brain_region: brainregion_index}
        self.assertIs(type(brain_regions), bidict.bidict)
        self.assertCountEqual(brain_regions.keys(), ["mPFC", "vHPC", "BLA", "NAc", "MD"])
        # traces is numpy array [ brainregion_index, timebins ]
        self.assertIs(type(traces), np.ndarray)

        # All of the indexes in brain_regions exist in traces
        for each in brain_regions.items():
            self.assertEqual(traces[:, each[1]].shape, (2500,))
        self.assertEqual(brain_regions["mPFC"], 0)
        self.assertEqual(brain_regions["vHPC"], 4)

    def test_zscore(self):
        traces = utils.load_test_traces()
        brain_regions, sorted_channels = preprocessor.map_to_region(SUBJECT_DICT)
        # use scipy median_abs_deviation , put in 5, X array, get an array of 5 by 1
        mad_list = preprocessor.median_abs_dev(traces)
        self.assertEqual(mad_list.shape[0], traces.shape[1])
        zscore_traces = preprocessor.zscore(traces)
        self.assertEqual(traces.shape, zscore_traces.shape)

    def test_rms(self):
        traces = utils.load_test_traces()
        brain_regions, sorted_channels = preprocessor.map_to_region(SUBJECT_DICT)
        rms_traces = preprocessor.root_mean_square(traces)
        self.assertEqual(traces.shape, rms_traces.shape)

    def test_plot_zscore(self):
        self.assertTrue(os.path.isdir(OUTPUT_DIR))
        OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "test_zscore.png")

        traces = np.loadtxt("lfp/tests/test_data/11_cups_p4_merged.rec_500_100000.csv", delimiter=",").T
        SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
        brain_regions, sorted_channels = preprocessor.map_to_region(SUBJECT_DICT)
        zscore_traces = preprocessor.zscore(traces)
        scaled_traces = preprocessor.scale_voltage(traces, 0.195)
        zscore_threshold = preprocessor.zscore_filter(zscore_traces, scaled_traces, 4)
        preprocessor.plot_zscore(traces, zscore_traces, zscore_threshold, OUTPUT_FILE_PATH)
        self.assertTrue(os.path.exists(OUTPUT_DIR))

    def test_zscore_filter(self):
        zscores = np.array([[-5.0, -1.0, 2.0, 3.0, 4.0, 5.0], [-5.0, -1.0, 2.0, 3.0, 4.0, 5.0]])
        voltage_scaled = np.array(
            [[-0.975, -0.195, 0.39, 0.585, 0.78, 0.975], [-0.975, -0.195, 0.39, 0.585, 0.78, 0.975]]
        )
        threshold = 3
        zscore_filtered_zscores = preprocessor.zscore_filter(zscores, voltage_scaled, threshold)
        self.assertEqual(zscore_filtered_zscores.shape, zscores.shape)
        self.assertTrue(np.isnan(zscore_filtered_zscores[0][0]))
        self.assertEqual(zscore_filtered_zscores[0][1], -0.195)
