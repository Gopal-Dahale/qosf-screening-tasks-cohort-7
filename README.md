# QOSF screening tasks cohort 7
This repository contains the solution for QOSF screening task 3: QSVM

## Problem Statement
Generate a Quantum Support Vector Machine (QSVM) using the iris dataset and try to propose a kernel from a parametric quantum circuit to classify the three classes(setosa, versicolor, virginica) using the one-vs-all format, the kernel only works as a binary classification. Identify the proposal with the lowest number of qubits and depth to obtain higher accuracy. You can use the UU† format or using the Swap-Test.

## Methodoloy

### Data preprocessing
Each data point contains 4 features. We pad to it with two zeros to make it a multiple of 3. This is necessary for the ansatz that we will use. We then perform a min-max scaling of the dataset (0 to 1). 

The dataset contains 150 samples. We split the dataset into train (90 samples) and test (60 samples).

### Parametric quantum circuit

We created a data re-uploading ansatz inspired by the paper [Data re-uploading for a universal quantum classifier](https://quantum-journal.org/papers/q-2020-02-06-226/). In brief, every feature of the input vector is multiplied with weight and added with a bias i.e. $z_i = w_ix_i + b$. $z_i$ will be passed into a rotation gate ($R_x$, $R_y$ or $R_z$). We use the Rot gate which is $R_zR_yR_z$ repeatedly and for this, the input feature vector has to be a multiple of 3. This block can then be repeated on the single qubit or on more qubits (in this case we can have CZ entanglement).

The figure shows a data re-uploading ansatz for an input vector of size 3. It has 4 qubits and 2 layers.

```
0: ──H──RZ(26.54)──RY(6.94)──RZ(27.18)─╭●──RZ(34.98)──RY(7.90)──RZ(10.63)────╭Z─┤ ╭Probs
1: ──H──RZ(20.11)──RY(6.35)──RZ(18.14)─╰Z──RZ(33.72)──RY(7.05)──RZ(6.89)──╭●─│──┤ ├Probs
2: ──H──RZ(17.14)──RY(7.68)──RZ(22.64)─╭●──RZ(11.87)──RY(9.20)──RZ(5.86)──╰Z─│──┤ ├Probs
3: ──H──RZ(25.66)──RY(8.97)──RZ(4.99)──╰Z──RZ(33.86)──RY(3.82)──RZ(20.04)────╰●─┤ ╰Probs
```

This will be the $U(x)$ where $x$ is the input feature vector. The actual ansatz will be $U(x)$ $U(x)$† followed by measuring the probs.

### One vs All strategy

One-vs-All (OVA) is a strategy to extend binary classifiers for multi-class classification problems. In this strategy, we train one binary classifier per class, where each classifier is trained to separate the samples of that class from the samples of all other classes.

To use QSVC for OVA in multi-class classification, we train one binary classifier for each class. During training, we set the samples of the current class as positive (+1) and all other classes as negative (-1). Once we have trained a binary classifier for each class, we can use them to predict the class of a new sample by applying each classifier to the sample and selecting the class associated with the classifier that produces the highest score.

## Results

For evaluation, we vary the number of qubits and layers and obtain the [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

We first evaluate the kernel with random parameters. The below images show the train and test f1 scores along with the runtimes.

<p align="center">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/assets/49199003/f58d0b06-3308-4fa6-88a8-c622802a15ae">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/assets/49199003/e849d374-f69c-4089-b103-04991cc5e2c1">
</p>

It is evident that as the number of qubits increase (so is the number of trainable parameters), the train/test score increases. For a fixed layer, the scores seem to saturate with 3 and 4 qubits (although, we need to perform more rigorous testing). It's difficult to comment at this time on which choice of ansatz is the best as the parameters are random. We train them and then evaluate them.

Regarding the runtimes, unsurprisingly, they increase as we increase the number of qubits and layers. For a fixed layer, we are expected to see an exponential increase in the runtime with every addition of a qubit. With 4 qubits and 4 layers, it takes nearly 2000s i.e. ~33 minutes to simulate with `lightning.qubit`. There exists an accuracy and time trade-off.

To train the kernel, we use the kernel-target alignment method. The kernel-target alignment evaluates the similarity between the labels in the training data and those predicted by the quantum kernel. It is based on kernel alignment, which compares two kernels with known kernel matrices $K_1$ and $K_2$ to determine how similar they are.

We were not able to train with more than 2 layers as it was not time effective.

<p align="center">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/assets/49199003/84eb3572-0218-473b-bd1a-fd87ae1c2e1d">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/assets/49199003/772b674e-298f-4a91-8774-fc24bb677a82">
</p>

After training the kernel, the f1 scores have improved and are within 0.95 for layer 2. This comes at the cost of runtime. With 4 qubits and 2 layers, the runtime being the highest 16k seconds i.e. ~ 4.5 hrs. Although training improves the score, the runtime is not satisfiable with 90 training data points.
The choice of ansatz should be determined by a balance between the f1 score and runtime

## Bonus: Reducing runtime with JAX

The implementation of [qml.kernels.square_kernel_matrix](https://docs.pennylane.ai/en/stable/_modules/pennylane/kernels/utils.html#square_kernel_matrix) uses nested for loops for computing the kernel matrix. It computes $\frac{1}{2}(N^2−N)$ kernel values for $N$ datapoints. We modify the function to use [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) transform to compute matrix elements in parallel. 

We also use [jax.jit](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) transform with which JAX can compile ts computation to XLA. The compiler performs a number of optimization passes while compiling an XLA program to improve computation performance. The first time you call the function will typically be slow due to this compilation cost, but all subsequent calls will be much, much faster.

We create three functions `square_kernel_matrix_jax`, `kernel_matrix_jax` and `target_alignment_jax` which will be used by QSVC with JAX.

We now compare the runtime of JAX implementation with the default one for 2 layers and 4 qubits ansatz. The y-axis is log scaled.

<p align="center">
  <img width="400" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/assets/49199003/6788112f-fbc5-4d0d-b52a-dc31dc90e66d">
</p>

We found that there is a `99.60 %` and `97.84 %` reduction in runtime with random and trained params respectively without compromising on the f1 scores. These results suggest that the proposed approach can significantly improve the efficiency of the classification model without sacrificing its performance, indicating its potential for large datasets.

## Structure of repository

1. `qsvc-ova-random.ipynb` implements QSVCs using the One vs All strategy with random parameters.
2. `qsvc-ova-trained.ipynb` implements QSVCs using the One vs All strategy with trained parameters.
3. `qsvc-ova-jax.ipynb` utilizes JAX to implement QSVCs using the One vs All strategy with random and trained parameters.
4. `results` directory consists of the results obtained during the execution of the notebook cells.

We create some Python files that are used in the notebooks 

1. `quantum_kernel.py` implements the `QuantumKernel` class for handling the creation and execution of quantum kernels.
2. `quantum_kernel_trainer.py` implements the `QuantumKernelTrainer` class is used to train quantum kernels using kernel-target alignment.
3. `qsvc.py` contains the `QSVC` class which is an extension of sklearn's SVC.
4. `utils.py` contains functions to handle the One vs All strategy.
5. `jax_utils.py` contains helper functions written in JAX.
6. `loss_functions.py` contain a single function for kernel-target alignment.

## References

1. [Training and evaluating quantum kernels](https://pennylane.ai/qml/demos/tutorial_kernels_module.html)
2. [Kernel-based training of quantum models with scikit-learn](https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html)
3. [Using JAX with PennyLane](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html)
4. [Data re-uploading for a universal quantum classifier](https://quantum-journal.org/papers/q-2020-02-06-226/)
