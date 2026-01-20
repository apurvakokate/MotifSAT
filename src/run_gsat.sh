#!/bin/bash

#SBATCH --time=1-00:00:00 --mem=50G
#SBATCH --gres=gpu:1 

# Initialize Conda
source ~/hpc-share/anaconda3/etc/profile.d/conda.sh

# Activate the desired environment
conda activate l2xgnn


for FOLD in 0 1 2 3 4
do
    for LAYERTYPE in GAT SAGE GIN GCN
    do
                
        for DATASETNAME in Lipophilicity esol
        do
        
            python run_gsat.py --dataset $DATASETNAME --fold $FOLD --task regression --backbone ${LAYERTYPE}  --cuda 0

        done
        
        for DATASETNAME in BBBP hERG Benzene Alkane_Carbonyl Fluoride_Carbonyl Mutagenicity
        do
        
            python run_gsat.py --dataset $DATASETNAME --fold $FOLD --task classification --backbone ${LAYERTYPE}  --cuda 0

        done
        
    done
done
