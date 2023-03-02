# QOSF screening tasks cohort 7
This repository contains the solution for QOSF screening task 3: QSVM

## Problem Statement
Generate a Quantum Support Vector Machine (QSVM) using the iris dataset and try to propose a kernel from a parametric quantum circuit to classify the three classes(setosa, versicolor, virginica) using the one-vs-all format, the kernel only works as binary class;pification. Identify the proposal with the lowest number of qubits and depth to obtain higher accuracy. You can use the UU† format or using the Swap-Test.

## Methodoloy

### Data preprocessing
Each data point contain 4 features. We pad to it with two zeros to make it a multiple of 3. This is necessary for the ansatz that we will use. We then perform a min-max scaling of the dataset (0 to 1). 

The dataset contains 150 samples. We split the dataset into train (90 samples) and test (60 samples).

### Parametric quantum circuit

We create a data re-uploading ansatz inspired from the paper [Data re-uploading for a universal quantum classifier](https://quantum-journal.org/papers/q-2020-02-06-226/). In brief, every feature of the input vector is multiplied with a weight and added with a bias i.e. $z_i = w_ix_i + b$. $z_i$ will be passed into a rotation gate ($R_x$, $R_y$ or $R_z$). We use th Rot gate which is $R_zR_yR_z$ repeatedly and for this the input feature vector has to be a multiple of 3. This block can then be repeated on the single qubit or on more qubits (in this case we can have CZ entanglement).

The figure shows a data re-uploading ansatz for a input vector of size 3. It has 4 qubits and 2 layers.

```
0: ──H──RZ(26.54)──RY(6.94)──RZ(27.18)─╭●──RZ(34.98)──RY(7.90)──RZ(10.63)────╭Z─┤ ╭Probs
1: ──H──RZ(20.11)──RY(6.35)──RZ(18.14)─╰Z──RZ(33.72)──RY(7.05)──RZ(6.89)──╭●─│──┤ ├Probs
2: ──H──RZ(17.14)──RY(7.68)──RZ(22.64)─╭●──RZ(11.87)──RY(9.20)──RZ(5.86)──╰Z─│──┤ ├Probs
3: ──H──RZ(25.66)──RY(8.97)──RZ(4.99)──╰Z──RZ(33.86)──RY(3.82)──RZ(20.04)────╰●─┤ ╰Probs
```

This will be the $U(x)$ where $x$ is the input feature vector. The actual ansatz will be $U(x)$ $U(x)$† followed by measuring the probs.

### One vs All strategy

One-vs-All (OVA) is a strategy to extend binary classifiers for multi-class classification problems. In this strategy, we train one binary classifier per class, where each classifier is trained to separate the samples of that class from the samples of all other classes.

To use QSVC for OVA in multi-class classification, we train one binary classifier for each class. During training, we set the samples of the current class as positive (+1) and samples of all other classes as negative (-1). Once we have trained a binary classifier for each class, we can use them to predict the class of a new sample by applying each classifier to the sample and selecting the class associated with the classifier that produces the highest score.

## Results

For evaluation, we vary the number of qubits and layers and obtain the [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

We first evaluate the kernel with random parameters. Below images shows the train and test f1 scores along with the runtimes.

<p align="center">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/blob/main/results/qsvc-ova-random-scores.png">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/blob/main/results/qsvc-ova-random-runtimes.png">
</p>

It is evident that as the number of qubits increase (so is the number of trainable parameters), the train/test score increases. For a fixed layer, the scores seems to saturate with 3 and 4 qubits (although, we need to perform more rigorous testing). Its difficult to comment at this time which choice of ansatz is the best as the parameters are random. We train them and then evaluate them.

Regarding the runtimes, unsuprisingly, they increase as we increase the number of qubits and layers. For a fixed layer, we are expected to see an exponential increase in the runtime with every addition of a qubit. With 4 qubits and 4 layers, it takes nearly 2000s i.e. ~33 minutes to simulate with `lightning.qubit`. There exists a accuracy and time trade off.

To train the kernel, we use kernel-target alignment method. The kernel-target alignment evaluates the similarity between the labels in the training data and those predicted by the quantum kernel. It is based on kernel alignment, which compares two kernels with known kernel matrices $K_1$ and $K_2$ to determine how similar they are.

We were not able to train with more than 2 layers as it was not time effective.

<p align="center">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/blob/main/results/qsvc-ova-trained-scores.png">
  <img width="500" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/blob/main/results/qsvc-ova-trained-runtimes.png">
</p>

After training the kernel, the f1 scores have improved and are within 0.95 for layer 2. This comes at the cost of runtime. With 4 qubits and 2 layers the runtime being the highest 16k seconds i.e ~ 4.5 hrs. Although, training improves score, the runtime is not satisfiable with 90 training data points.
The choice of ansatz should be determined by a balance between the f1 score and runtime

## Bonus: Reducing runtime with JAX

The implementation of [qml.kernels.square_kernel_matrix](https://docs.pennylane.ai/en/stable/_modules/pennylane/kernels/utils.html#square_kernel_matrix) uses nested for loops for computing the kernel matrix. It computes $\frac{1}{2}(N^2−N)$ kernel values for $N$ datapoints. We modify the function to use [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) transform to compute matrix elements in parallel. 

We also use [jax.jit](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) transform with which JAX can compile ts computation to XLA. The compiler performs a number of optimization passes while compiling an XLA program to improve computation performance. The first time you call the function will typically be slow due to this compilation cost, but all subsequent calls will be much, much faster.

We create three functions `square_kernel_matrix_jax`, `kernel_matrix_jax` and `target_alignment_jax` which will be used by QSVC with JAX.

We now compare the runtime of JAX implementation with the default one for 2 layers and 4 qubits ansatz. The y-axis is log scaled.

<p align="center">
  <img width="400" height="auto" src="https://github.com/Gopal-Dahale/qosf-screening-tasks-cohort-7/blob/main/results/default_vs_jax.png">
</p>

We found that there is a `99.60 %` and `97.84 %` reduction in runtime with random and trained params respectively without compromisng on the f1 scores. These results suggest that the proposed approach can significantly improve the efficiency of the classification model without sacrificing its performance, indicating its potential for large datasets.

## Structure of repository
