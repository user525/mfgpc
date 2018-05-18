# mfgpc
Multi-fidelity Gaussian Process Classification

Current repository contains artifitial datasets for binary classification problem. 

Each dataset is stored in a csv file with the following fields:

\texttt{feature\_1}, ..., \texttt{feature\_D}, \texttt{target\_gold}, \texttt{target\_noisy\_0.0}, ... \texttt{target\_noisy\_0.5}

All features are real-valued in range [0, 1]. 
Datasets have different dimensions (D). D takes values from 2, 5, 10, and 20. There are 10 datasets for each value of D.
\texttt{target\_gold} represents ground truth labels,
target\_noisy\_<C> represent corrupted labels with noise approximately equal to value C.

The source code is coming soon. 