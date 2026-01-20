#!/bin/bash
# Master script to run all tuning experiments
# Experiment: baseline
# Generated: 2026-01-19T19:11:06.351344

echo "Running Mutagenicity with GIN on fold 0"
bash run_Mutagenicity_GIN_fold0.sh

echo "Running Mutagenicity with GIN on fold 1"
bash run_Mutagenicity_GIN_fold1.sh

