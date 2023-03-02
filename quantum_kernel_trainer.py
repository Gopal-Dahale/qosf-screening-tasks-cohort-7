from abc import ABC
import pennylane.numpy as np
from loss_functions import target_alignment

class QuantumKernelTrainer(ABC):
    def __init__(self, quantum_kernel, optimizer, batch_size, init_params = None, iters = 10):
        self.quantum_kernel = quantum_kernel
        self.opt = optimizer
        self.iters = iters
        self.batch_size = batch_size
        self.loss = target_alignment

        if init_params is not None:
            self.quantum_kernel.assign_params(init_params)
    
    def fit(self, X, y):
        for i in range(self.iters):
            # print('step', i)
            # Choose subset of datapoints to compute the KTA on.
            subset = np.random.choice(list(range(len(X))), self.batch_size)
            
            # Define the cost function for optimization
            cost = lambda _params: -self.loss(
                X[subset],
                y[subset],
                _params,
                self.quantum_kernel,
                assume_normalized_kernel=True,
            )
            # Optimization step
            _params = self.opt.step(cost, self.quantum_kernel.params)
            self.quantum_kernel.assign_params(_params)
                
            # Report the alignment on the full dataset every every iter.
            if (i + 1) % 50 == 0:
                current_alignment = self.loss(
                    X,
                    y,
                    self.quantum_kernel.params,
                    self.quantum_kernel,
                    assume_normalized_kernel=True,
                )
                print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

        result = {}
        result['optimal_params'] = self.quantum_kernel.params
        result['quantum_kernel'] = self.quantum_kernel
        return result


