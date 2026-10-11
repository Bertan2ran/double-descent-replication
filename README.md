# double-descent-replication
## Reproducing Epoch-Wise Double Descent (CNN on CIFAR-10)

This repository contains the codebase to reproduce the **Epoch-Wise Double Descent** experiment (Figure 10c) from *Deep Double Descent: Where Bigger Models and More Data Hurt* (Nakkiran et al., 2021). This replication is part of Group Assignment 2 for the Advanced Topics in Deep Learning (ATDL) course.

### Repository Structure
* `models/cnn.py`: Contains the 5-layer Convolutional Neural Network architecture with scalable width parameter $k$.
* `train.py`: The main training loop, including deterministic data loading, 20% symmetric label noise injection, and the SGD inverse-square-root learning rate scheduler.
* `init.sh` & `run.sh`: Bash scripts formatted for automated batch deployment on UCloud GPU instances.
* `plot.py`: Generates the final ISBI-formatted `.pdf` vector graphic from the exported CSV logs.

### Dependencies
To run the code locally or on a cloud instance, ensure you have Python 3.8+ and the following packages installed:
```bash
pip install torch torchvision numpy pandas matplotlib
```
