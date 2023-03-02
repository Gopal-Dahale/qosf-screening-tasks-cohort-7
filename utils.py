import pennylane.numpy as np
import pennylane as qml
from quantum_kernel import QuantumKernel
from qsvc import QSVC
from quantum_kernel_trainer import QuantumKernelTrainer
from sklearn.metrics import f1_score

def softmax(x):
    bs, n_c = x.shape
    return np.exp(x)/np.repeat(np.exp(x).sum(axis = -1), n_c).reshape(bs, 3)

def predict(svms, features):
    probs = np.array([svm.predict_proba(features)[:,1]  for svm in svms]).T
    predictions = softmax(probs).argmax(axis = -1)
    return predictions

def ova_random(features_train, labels_train, features_test, labels_test, feature_map, n_layers, n_wires, 
               params_shape):
    svms = []
    classes = np.unique(labels_train)
    for i in classes:
        print(f"class {i}")
        y_onevsall_train = np.where(labels_train == i, 1, -1)
        y_onevsall_test = np.where(labels_test == i, 1, -1)
        
        params = np.random.uniform(0, 2 * np.pi, params_shape, requires_grad=True)
        qkernel = QuantumKernel(feature_map, n_layers, n_wires, params, 'lightning.qubit')
        svms.append(QSVC(quantum_kernel=qkernel, probability=True).fit(features_train, y_onevsall_train))

        accuracy_train = svms[i].score(features_train, y_onevsall_train)
        print(f"The accuracy of the kernel for class {i} with random parameters is {accuracy_train:.3f}")
        
        accuracy_test = svms[i].score(features_test, y_onevsall_test)
        print(f"The accuracy of the kernel for class {i} with random parameters is {accuracy_test:.3f}")
        
    return svms

def ova_trained(features_train, labels_train, features_test, labels_test, feature_map, n_layers, n_wires, 
                params_shape, iters = 10, batch_size = 4, lr = 0.1):
    svms_trained = []
    classes = np.unique(labels_train)
    for i in classes:
        print(f"class {i}")
        y_onevsall_train = np.where(labels_train == i, 1, -1)
        y_onevsall_test = np.where(labels_test == i, 1, -1)

        params = np.random.uniform(0, 2 * np.pi, params_shape, requires_grad=True)
        qkernel = QuantumKernel(feature_map, n_layers, n_wires, params, 'lightning.qubit')

        qkt = QuantumKernelTrainer(qkernel, qml.AdamOptimizer(lr), batch_size, iters = iters)
        qka_results = qkt.fit(features_train, y_onevsall_train)
        optimized_kernel = qka_results['quantum_kernel']

        svms_trained.append(QSVC(quantum_kernel=optimized_kernel, probability=True).fit(features_train, 
                                                                                        y_onevsall_train))
        
        accuracy_train = svms_trained[i].score(features_train, y_onevsall_train)
        print(f"The accuracy of the kernel for class {i} with trained parameters is {accuracy_train:.3f}")
        
        accuracy_test = svms_trained[i].score(features_test, y_onevsall_test)
        print(f"The accuracy of the kernel for class {i} with trained parameters is {accuracy_test:.3f}")
    return svms_trained

def get_scores(svms, features_train, labels_train, features_test, labels_test):
    preds = predict(svms, features_train)
    score_train = f1_score(labels_train, preds, average='micro')
    print(f"F1 score on train set: {score_train:.3f}")

    preds = predict(svms, features_test)
    score_test = f1_score(labels_test, preds, average='micro')
    print(f"F1 score on test set: {score_test:.3f}")
    
    return score_train, score_test