#!/bin/bash
# Run Mutagenicity with GIN on fold 1


# Config: config_0000
# Params: {'pred_loss_coef': 1.0, 'info_loss_coef': 1.0, 'init_r': 0.9, 'final_r': 0.1, 'decay_r': 0.1, 'decay_interval': 10, 'tuning_id': 'config_0000', 'motif_loss_coef': 0.0}
# Seed 0
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config /nfs/hpc/share/kokatea/ChemIntuit/MotifSAT/configs/tuning/baseline/config_0000.yaml

# Seed 1
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config /nfs/hpc/share/kokatea/ChemIntuit/MotifSAT/configs/tuning/baseline/config_0000.yaml


# Config: config_0001
# Params: {'pred_loss_coef': 1.0, 'info_loss_coef': 1.0, 'init_r': 0.9, 'final_r': 0.1, 'decay_r': 0.1, 'decay_interval': 10, 'tuning_id': 'config_0001', 'motif_loss_coef': 0.5}
# Seed 0
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0001.yaml

# Seed 1
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0001.yaml


# Config: config_0002
# Params: {'pred_loss_coef': 1.0, 'info_loss_coef': 1.0, 'init_r': 0.9, 'final_r': 0.1, 'decay_r': 0.1, 'decay_interval': 10, 'tuning_id': 'config_0002', 'motif_loss_coef': 1.0}
# Seed 0
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0002.yaml

# Seed 1
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0002.yaml


# Config: config_0003
# Params: {'pred_loss_coef': 1.0, 'info_loss_coef': 1.0, 'init_r': 0.9, 'final_r': 0.1, 'decay_r': 0.1, 'decay_interval': 10, 'tuning_id': 'config_0003', 'motif_loss_coef': 2.0}
# Seed 0
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0003.yaml

# Seed 1
python run_gsat.py \
  --dataset Mutagenicity \
  --backbone GIN \
  --fold 1 \
  --cuda 0 \
  --config configs/tuning/baseline/config_0003.yaml

