# mfgpc
Multi-fidelity Gaussian Process Classification

Current repository contains artifitial datasets for binary classification problem. 

Each dataset contains 5000 entries, they are stored in csv files with the following fields:

`feature_1`, ..., `feature_D`, `target_gold`, `target_noisy_0.0`, ..., `target_noisy_0.5`

Datasets have different dimensions (D).
D takes values from 2, 5, 10, and 20. 
There are 10 datasets for each value of D.
All features are real-valued in range [0, 1]. 
Feature-vectors are uniformly scattered across 0-1 D-dimensional hyper-cube.
`target_gold` represents ground truth labels,
`target_noisy_<c>` represent corrupted labels with noise approximately equal to value <c>.


Model for Multi-fidelity Gaussian Process Classifier is implemented based on Scikit-learn Gaussian Process Classifier in mfgpc_opt.py module.

To allow reproducibility of experiments jupyter notebooks have also been published.