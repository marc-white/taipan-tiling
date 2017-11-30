"""File for loading in the results of tiling
"""
import pickle
import cPickle

pkl_file_name = "results/171008_1145_fw_tiling.pkl"

pkl_file = open(pkl_file_name, "rb")
(tiling, remaining_targets, run_settings) = cPickle.load(pkl_file)