#!/bin/bash

# Figure 1 and 2: LQPQM
python ./make_figure_1_2_LQPQM.py

# Figure 3 and Table II: Random SeDJoCo Problems
python ./experiment_head_plot.py data/20201211-113855_experiment_head_results.npz

# Figure 4 and Table III: Separation of Synthetic Mixtures
python ./experiment_iva_plot.py data/merge_20210305-190428_experiment_iva_results_20210320-174319_experiment_iva_results.npz

# Figure 5: Box-plots of SDR/SIR
python ./make_figure3_journal_ipa_separation_hist.py \
  ./data/20201222-185917_speed_contest_journal_ipa_2bc728a163/ --pca

# Figure 6: Wall-clock vs SDR/SIR
python ./make_figure4_journal_ip_speed_contest.py \
  ./data/20201222-185917_speed_contest_journal_ipa_2bc728a163/ --pca --pickle

# Table IV: Runtimes
python ./make_table_runtime.py data/20201222-185917_speed_contest_journal_ipa_2bc728a163/ --pickle
