"""File for loading in the results of tiling
"""
import pickle
import cPickle

pkl_file = "results/fw_tiling_172607_1634.pkl"

pkl_file = open(pkl_file, "rb")
(test_tiling, remaining_targets, run_settings) = cPickle.load(pkl_file)