import pennylane as qml
import pennylane.numpy as np

def target_alignment(
        features,
        labels,
        params,
        quantum_kernel,
        assume_normalized_kernel=False,
        rescale_class_labels=True,
    ):
        """Kernel-target alignment between kernel and labels."""

        quantum_kernel.assign_params(params)
        
        K = qml.kernels.square_kernel_matrix(
            features,
            quantum_kernel.kernel,
            assume_normalized_kernel=assume_normalized_kernel,
        )

        # print(K)

        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(labels) == 1)
            nminus = len(labels) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in labels])
        else:
            _Y = np.array(labels)
        # print('labels', labels)
        # print('_Y', _Y)
        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        inner_product = inner_product / norm

        # print('inner_product', inner_product)

        return inner_product
    