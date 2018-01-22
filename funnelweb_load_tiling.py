"""File for loading in the results of tiling
"""
import pickle
import cPickle

pkl_file_name = "results/170609_0101_fw_tiling.pkl"

pkl_file = open(pkl_file_name, "rb")
(tiling, remaining_targets, run_settings) = cPickle.load(pkl_file)