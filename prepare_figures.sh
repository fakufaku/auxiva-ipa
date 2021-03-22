#!/bin/bash

# make the figures about LQPQM
python ./make_figure_1_2_LQPQM.py

# figure with box-plots of SDR/SIR
python ./make_figure3_journal_ipa_separation_hist.py \
  ./data/20201222-185917_speed_contest_journal_ipa_2bc728a163/ --pca

# figure with wall-clock vs SDR/SIR
python ./make_figure4_journal_ip_speed_contest.py \
  ./data/20201222-185917_speed_contest_journal_ipa_2bc728a163/ --pca --pickle

# figure for the HEAD experiment
python ./experiment_head_plot.py data/20201211-113855_experiment_head_results.npz

# figure for the IVA experiment
python ./experiment_iva_plot.py data/merge_20210305-190428_experiment_iva_results_20210320-174319_experiment_iva_results.npz

# table for the runtime table
python ./make_table_runtime.py data/20201222-185917_speed_contest_journal_ipa_2bc728a163/ --pickle
