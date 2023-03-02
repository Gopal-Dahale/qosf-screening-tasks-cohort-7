from sklearn.svm import SVC
import warnings

class QSVC(SVC):
    def __init__(self, *, quantum_kernel, **kwargs):

        if "kernel" in kwargs:
            msg = (
                "'kernel' argument is not supported and will be discarded, "
                "please use 'quantum_kernel' instead."
            )
            warnings.warn(msg, stacklevel=2)
            del kwargs["kernel"]

        self.quantum_kernel = quantum_kernel
        super().__init__(kernel= self.quantum_kernel.evaluate, **kwargs)

    