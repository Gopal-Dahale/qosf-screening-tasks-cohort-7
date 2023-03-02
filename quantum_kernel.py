from abc import ABC
import pennylane.numpy as np
import pennylane as qml

class QuantumKernel(ABC):
    """Base class for quantum kernels in PennyLane.
    Args:
        feature_map (pennylane.templates.Embedding): Quantum circuit that encodes data into a quantum
            state
        device (pennylane.Device): A PennyLane device to use for quantum circuit execution
    """
    def __init__(self, feature_map, num_layers, num_wires, params, device):
        self.feature_map = feature_map
        self.adjoint_feature_map = qml.adjoint(feature_map)
        self.dev = qml.device(device, wires=num_wires)
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.assign_params(params)

    def assign_params(self, params):
        """Assigns new parameters to the quantum kernel.
        Args:
            params (array[float]): New parameters.
        """
        self.params = np.array(params, requires_grad=True)

    def kernel(self, x1, x2):
        """Evaluate the kernel function.
        Args:
            x1 (array[float]): First datapoint.
            x2 (array[float]): Second datapoint.
        Returns:
            array[float]: Kernel value.
        """
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2, params):
            self.feature_map(x1, params, self.num_layers, n_wires=self.num_wires)
            self.adjoint_feature_map(x2, params, self.num_layers, n_wires=self.num_wires)
            return qml.probs(wires=range(self.num_wires))
        
        res = kernel_circuit(x1, x2, self.params)
        # print(x1, x2)
        # print(res)
        return res[0]

    def evaluate(self, X1, X2):
        """Evaluate the kernel function on two matrices.
        Args:
            X1 (array[float]): A matrix of n_samples1 of dimension n_features.
            X2 (array[float]): A matrix of n_samples2 of dimension n_features.
        Returns:
            array[float]: A matrix of shape (n_samples_X1, n_samples_X2).
        """
        
        return qml.kernels.kernel_matrix(X1, X2, self.kernel)