# double-descent-replication

## Reproducing Model-Wise Double Descent (CNN on CIFAR-10)

This experiment reproduces the **Model-Wise Double Descent** phenomenon (Figure 5) from
*Deep Double Descent: Where Bigger Models and More Data Hurt*
(Nakkiran et al., 2021).

We train a 5-layer convolutional neural network on CIFAR-10 while varying the
network width parameter \(k\). The training set is corrupted with 20% symmetric
label noise in order to make the interpolation regime more visible.

### Experimental Setup

- Dataset: CIFAR-10
- Architecture: 5-layer CNN
- Label noise: 20% symmetric label noise
- Batch size: 128
- Optimizer: SGD
- Initial learning rate: 0.1
- Learning-rate decay: factor 0.1 after epochs 100 and 150
- Training duration: 250 epochs per model
- Random seed: 41
- Model widths:
  `2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64`

The CNN uses four convolutional layers with channel widths
`[k, 2k, 4k, 8k]`, followed by a fully connected classifier.

### Repository Structure

- `model_wise_test.py`: Implements the complete model-wise double descent experiment,
  including CIFAR-10 loading, label-noise injection, CNN training, evaluation,
  the model-width sweep, and generation of the final plot.
- `model_wise_results.log`: Full training output for the width sweep and the final training and test results for all
  evaluated model widths.

### Running the Experiment
Make sure you installed the following packages
```bash
pip install torch torchvision numpy matplotlib
python model_wise_test.py
```


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
