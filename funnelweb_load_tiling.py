"""File for loading in the results of tiling
"""
import pickle
import cPickle

pkl_file = "results/170508_1615_fw_tiling.pkl"

pkl_file = open(pkl_file, "rb")
(tiling, remaining_targets, run_settings) = cPickle.load(pkl_file)