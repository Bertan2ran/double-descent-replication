# Deep Double Descent: Reproducibility Study

This repository contains the codebase and replication materials for a reproducibility study of *Deep Double Descent: Where Bigger Models and More Data Hurt* (Nakkiran et al., 2021). This project was completed as part of Group Assignment 2 for the Advanced Topics in Deep Learning (ATDL) course.

The project replicates the main experimental findings of the paper across three core phenomena:
* **Epoch-Wise Double Descent:** Replicating the training dynamics over time using a unified UCloud batch processing pipeline and deterministic data loaders.
* **Sample-Wise Non-Monotonicity:** Executing a 10-point sample size sweep ($n \in [6250, 50000]$) under 20% label noise to demonstrate critical regime data scaling.
* **Model-Wise Double Descent:** Scaling CNN architecture width parameters to capture the interpolation peak under both noisy (20% label noise on UCloud) and noiseless (Google Cloud) configurations.


## Reproducing the Epoch-Wise Experiment

The instructions below detail how to run the epoch-wise double descent experiment. All code for this specific experiment is contained within the `Epoch_Wise` directory.

### Repository Structure
* `Epoch_Wise/models/cnn.py`: Contains the 5-layer Convolutional Neural Network architecture with scalable width parameter $k$.
* `Epoch_Wise/train.py`: The main training loop, including deterministic data loading, 20% symmetric label noise injection, and the SGD inverse-square-root learning rate scheduler.
* `Epoch_Wise/init.sh` & `Epoch_Wise/run.sh`: Bash scripts formatted for automated batch deployment on UCloud GPU instances.
* `Epoch_Wise/plot.py`: Generates the final ISBI-formatted `.pdf` vector graphic from the exported CSV logs.

### Dependencies
To run the code locally or on a cloud instance, ensure you have Python 3.8+ and the following packages installed:
```bash
pip install torch torchvision numpy pandas matplotlib
