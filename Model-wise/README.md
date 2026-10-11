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
