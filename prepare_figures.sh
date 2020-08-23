#!/bin/bash

# make the figure with the RT60 distribution
python ./make_figure_rt60_hist.py ./data/experiment1_rooms.json

# make the figures about LQPQM
python ./make_figure_1_2_LQPQM.py

# figure with box-plots of SDR/SIR
python ./make_figure3_journal_ipa_separation_hist.py \
  ./data/20200706-225526_speed_contest_journal_ipa_720398193e

# figure with wall-clock vs SDR/SIR
python ./make_figure4_journal_ip_speed_contest.py \
  ./data/20200706-225526_speed_contest_journal_ipa_720398193e
