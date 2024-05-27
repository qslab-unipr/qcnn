This repository contains the source code of the paper "Multi-Class Quantum Convolutional Neural Networks" https://arxiv.org/abs/2404.12741
submitted to the QUASAR Workshop (https://sites.google.com/view/quasar24/home?authuser=0) in conjunction with the 33rd ACM International
Symposium on High-Performance Parallel and Distributed Computing.

If you want to execute the code step by step run the qcnn.ipynb file. In the first block of the notebook, you can set the different variables.

If you want to execute all the code together, you can run the python_files/main.py file. 
In this case, when you execute the code, you can set different argument to modify the value of the hyperparameters, e.g.: <br />
`python ./main.py --epochs 2 --opt 'Adam' --lr 0.01 --batch_size 32 --num_classes 10 `<br />
This will execute the QCNN with 2 epochs, with Adam as optimizer with 0.01 as learning rate. The batch size is 32 and it will take all the classes of the MNIST dataset.

The default parameters are: <br />
`epochs 2` <br />
`batch_size 64` <br />
`lr 0.01` <br />
`opt 'Adam'` <br />
`num_classes 10` <br />
`encoding 'amplitude'` <br />
`num_layer 1` <br />
`all_samples False` <br />
`seed None` <br />

To get the same results of the paper, run the code with the default value, setting all_samples = True and seed = 43 and changing the number of classes.
In this way, you will obtain the result when all the samples of the dataset are used.
Otherwise, you have to change the number of epochs to 20, 40, 60,... and the seed to 43.
In this way, you will obtain the results when only 250 samples of the dataset are used.


**file:** <br />
`python_files/main.py` contains the main file. To execute the qcnn, you need to execute this file. You can set different parameters such as number of epochs,
learning rate, optimizer, etc. <br />
`python_files/data.py` contains the loading and processing of the MNIST dataset <br />
`python_files/QCNN_circuit.py` contains the structure of QCNN  <br />
`python_files/unitary.py` implements the single circuits which implement the convolutional and the pooling layer <br />
`python_files/Training.py` implements the training process of the network. <br />
`qcnn.ipynb` contains all the code as a notebook

## **How to run** <br />
Set, in the first cell of the notebook, the parameters:
```python
n_qubit = 8 #number of qubit in the circuit
encoding = 'amplitude' #choose the quantum encoding: 'amplitude' or 'angle'
num_classes = 10 # choose how many classes: 4, 6, 8, 10
all_samples = True #True if you want all the samples, False, if you want only 250 samples for each class
seed = 43 #set to None to generate the seed randomly
U_params = 15 #number of parameters of F_2 circuit
num_layer = 1 #number of convolutional layer repetitions
load_params = False #if True load parameters from a file
opt = 'Adam' #choose the optimizer: Adam, QNGO, or GDO
lr = 0.01 #learning rate
epochs = 2 #number of epochs
batch_size = 64 #size of batch
```
or, if the python code is used, you can set them as arguments: <br />
`python ./main.py --epochs 2 --opt 'Adam' --lr 0.01 --batch_size 32 --num_classes 10`

**MNIST Dataset**
```python
""" 
It loads the MNIST dataset and then it processes the dataset based on the encoding method, number of classes and if we want all the samples. 
param encoding: indicate the quantum encoding used: 'amplitude' or 'angle'
param num_classes: number of classes to be predicted, which samples take from the dataset
param all_samples: True if we want all the samples, False to take only 250 samples for each class
param seed: random_state seed
return X_train, X_test, Y_train, Y_test: the dataset divided in training and test set
"""
def data_load_and_process(encoding, num_classes, all_samples, seed):
	if seed != None:
		tf.random.set_seed(seed)
		np.random.seed(seed)

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


	x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0	 # normalize the data
```
If `all_samples = False`, then only 250 samples per class from the training set are considered for training:
```python
  if not all_samples:
    num_examples_per_class = 250
    selected_indices = []

    # Iterate through each class to select 250 examples
    for class_label in range(10):
      indices = np.where(y_train == class_label)[0][:num_examples_per_class]
      selected_indices.extend(indices)

    # Filter the training data to contain only the selected examples
    x_train_subset = x_train[selected_indices]
    y_train_subset = y_train[selected_indices]
```
Then, depending on the value of `num_classes`, only that range of classes are taken into account:
```python
  mask_train = np.isin(y_train, range(0, num_classes))
  mask_test = np.isin(y_test, range(0, num_classes))

  X_train = x_train[mask_train]
  X_test = x_test[mask_test]		
  Y_train = y_train[mask_train]
  Y_test = y_test[mask_test]
```
Then if the amplitude encoding is used, only 256 features are taken in consideration, because with 8 qubits we can represent at most \$2^8 = 256\$ features. Instead, if we use angle encoding, only the 8 most significant features are taken. The selected features are taken using PCA: 
```python
  #check which encoding is used
  #if amplitude encoding is used, then the 256 most important features are taken using PCA
  if encoding == 'amplitude':
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    pca = PCA(n_components = 256)
    X_train = pca.fit_transform(X_train_flat)
    X_test = pca.transform(X_test_flat)
    return X_train, X_test, Y_train, Y_test
  #if amplitude encoding is used, then the 8 most important features are taken using PCA
  elif encoding == 'angle':
    X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
    X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
    X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)

    pca = PCA(8)
		
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Rescale for angle embedding
		
    X_train, X_test = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min())),\
                      (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))
    return X_train, X_test, Y_train, Y_test
```
**QCNN** <br />
The QCNN is implemented. First of all, depending on the selected encoding, one of the two functions implemented in PennyLane (`AmplitudeEmbedding()` or `AngleEmbedding()`) is utilized. Then, the VQC is built with the function `QCNN_structure()` and, finally the measurement is performed. Depending on how many classes we want to classify, the number of qubits measured changes: <br />
4 classes: we need 4 states, from |00> to |11>, so we need only 2 qubits (|00> is associated with class 0, |01> with 1 and so on) <br/>
6 classes: 6 states, from |000> to |101> &rarr; 3 qubits <br/>
8 classes: 8 states, from |000> to |111> &rarr; 3 qubits <br/>
10 classes: 10 states, from |0000> to |1001> &rarr; 4 qubits <br/>
```python
"""
define the simulator and the QCNN: encoding + VQC + measurement
param X: sample in input
param params: theta angle of the rotations. parameters to be trained
param U_params: number of parameters which implement a single block of the F_2 circuit
param embedding_type: which encoding is chosen
param num_classes: how many classes the QNN has to predict
param num_layer: number of repetition of the convolutional layer
return result: the probabilities of the states, which are associated to the MNIST classes
"""
dev = qml.device('default.qubit', wires = n_qubit)
@qml.qnode(dev)
def QCNN(X, params, U_params, embedding_type='amplitude', num_classes=10, num_layer = 1):
	# Data Embedding
	if embedding_type == 'amplitude':
		AmplitudeEmbedding(X, wires=range(8), normalize=True)
	elif embedding_type == 'angle':
		AngleEmbedding(X, wires=range(8), rotation='Y')
	
	# Create the VQC
	QCNN_structure(CC14, params, U_params, num_classes, num_layer)
	
	#Measures the necessary qubits
	if num_classes == 4:
		result = qml.probs(wires=[0, 4])
	elif num_classes == 6:
		result = qml.probs(wires=[0, 2, 4])
	elif num_classes == 8:
		result = qml.probs(wires=[0, 2, 4])
	else:
		result = qml.probs(wires=[0, 2, 4, 6])			
					
	return result
```
The following function implement the general structure of the QCNN:
It divides the parameters across the various layers. Then it calls the two functions used to add a convolutional layer or a pooling layer. For each pooling layer, only two qubits are traced out, so another pooling layer is added if only 4/6 or 8 classes need to be classified.
```python
"""
It implements the structure of the QCNN
param U: unitary F_1
param params: theta angle of the rotations. parameters to be trained
param U_params: number of parameters which implement a single block of the F_2 circuit
param num_classes: how many classes the QNN has to predict
param num_layer: number of repetition of the convolutional layer
"""
def QCNN_structure(U, params, U_params, num_classes, num_layer):
	#divide the number of parameters for each layer: conv layer1, pooling layer 1, conv layer 2, ...
	#U_params indicates the number of parameters of the F_2 circuit (the circuit applied to couple of adjacent qubit)
	#n_qubit * 2: is the number of parameters for the circuit F_1
	param1CL = params[0: (U_params + n_qubit * 2) * num_layer]
	param1PL = params[(U_params + n_qubit * 2) * num_layer: ((U_params + n_qubit * 2) * num_layer) + 2]
		
	param2CL = params[((U_params + n_qubit * 2) * num_layer) + 2: ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer)]
	param2PL = params[((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer): 
					  ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2]

	param3CL = params[((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2: 
					  ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer)]

	#apply the circuits
	conv_layer(U, param1CL, U_params, num_layer, range(n_qubit))
	pooling_layer1(Pooling_ansatz, param1PL)
	
	conv_layer(U, param2CL, U_params, num_layer, [0, 2, 3, 4, 5, 6])
	pooling_layer2(Pooling_ansatz, param2PL)
	
	conv_layer(U, param3CL, U_params, num_layer, [0, 2, 4, 6])

	#if we have only 4, 6 or 8 classes, then we need another pooling layer and we need to trace out:
	#another qubit if we have 6 or 8 classes, because we need only 3 qubits to represent 6 or 8 classes
	#2 qubits if we have 4 classes, because we need only 2 qubits to represent 4 classes/states
	#if we have 10 classes, then we don't apply another pooling layer, because we need 4 qubits
	if num_classes == 4 or num_classes == 6 or num_classes == 8:
		param3PL = params[((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer):
						 ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer) + 2]	  

		pooling_layer3(Pooling_ansatz, param3PL, num_classes)
```
The following functions implement the structure of the convolutional and the pooling layer. 
```python
"""
Quantum Circuits for Convolutional layers
param U: unitary that implements the convolution
param params: theta angle of the rotations. parameters to be trained
param U_params: number of parameters which implement a single block of the F_2 circuit
param num_layer: number of repetition of the convolutional layer
param qubits: array that indicate to which qubit apply the convolutional layer
"""
def conv_layer(U, params, U_params, num_layer, qubits):
		param0 = 0
		param1 = len(qubits) * 2
		
		#add f_1 circuit
		for l in range(num_layer):
			if len(qubits) == 8: #if it is the first layer, the F_1 circuit is "divided"
				for i in range(0, len(qubits), len(qubits)//2):
					U(params[param0: param1], wires = qubits[i: i + len(qubits)//2])
			else:
				param1 += len(qubits) * 2
				U(params[param0: param1], wires = qubits[0: len(qubits)])

			#now add the two-qubit circuit (F_2)
			param0 = param1
			param1 += U_params
			for i in range(0, len(qubits), 2):
				U_SU4(params[param0: param1], wires = [qubits[i % len(qubits)], qubits[(i + 1) % len(qubits)]])
			
			for i in range(1, len(qubits), 2):
				U_SU4(params[param0: param1], wires = [qubits[i % len(qubits)], qubits[(i + 1) % len(qubits)]])

			param0 = param1
			param1 += len(qubits) * 2
"""
Quantum Circuits for Pooling layers
param V: unitary which implements the pooling operation
param params: theta angle of the rotations. parameters to be trained
"""
def pooling_layer1(V, params):
	V(params, wires=[7, 6]) 
	V(params, wires=[1, 0]) 

def pooling_layer2(V, params):
	V(params, wires=[3, 2]) 
	V(params, wires=[5, 4]) 

def pooling_layer3(V, params, num_classes):
	if num_classes == 4: #if we need only 4 classes. we trace out another qubit
		V(params, wires=[2,0])				   
	V(params, wires=[6,4])
```
The following functions implement, respectively, the F_2 circuit (the convolutional filter, applied to more than just 2 qubit, to enahnce expressibility and entangling capability), the F_1 circuit (which is the circuit applied to couple of adjacent qubits), and the pooling circuit of the paper:
```python
"""
F_1 circuit of the paper
param params: theta angle of the rotations. parameters to be trained
param wires: qubits to apply the gates
"""
def CC14(params, wires):
	#U_CC14 r = 1
	for i in range(0, len(wires)):
		qml.RY(params[i], wires=wires[i])
	for i in range(0, len(wires)):
		qml.CRX(params[i + len(wires)], wires=[wires[(i - 1) % len(wires)], wires[i]])
		
	
	
	#U_CC14 r = -1 or 3
	for i in range(0, len(wires)):
		qml.RY(params[i + 2 * len(wires)], wires=wires[i])
		
	if len(wires) % 3 == 0 or len(wires) == 2:
		for i in range(len(wires) - 1, -1, -1):
			qml.CRX(params[i + 3 * len(wires)], wires=[wires[i], wires[(i-1) % len(wires)]])
			
	else:
		control = len(wires) - 1
		target = (control + 3) % len(wires)
		for i in range(len(wires) - 1, -1, -1):
			qml.CRX(params[i + 3 * len(wires)], wires=[wires[control], wires[target]])
			
			control = target
			target = (control + 3) % len(wires)

"""
F_2 circuit of the paper
param params: theta angle of the rotations. parameters to be trained
param wires: qubits to apply the gates
"""
def U_SU4(params, wires): # 15 params
	qml.U3(params[0], params[1], params[2], wires=wires[0])
	qml.U3(params[3], params[4], params[5], wires=wires[1])
	qml.CNOT(wires=[wires[0], wires[1]])
	qml.RY(params[6], wires=wires[0])
	qml.RZ(params[7], wires=wires[1])
	qml.CNOT(wires=[wires[1], wires[0]])
	qml.RY(params[8], wires=wires[0])
	qml.CNOT(wires=[wires[0], wires[1]])
	qml.U3(params[9], params[10], params[11], wires=wires[0])
	qml.U3(params[12], params[13], params[14], wires=wires[1])

"""
It implements the pooling circuit
param params: theta angle of the rotations. parameters to be trained
param wires: qubits to apply the gates
"""
def Pooling_ansatz(params, wires): #2 params
	qml.CRZ(params[0], wires=[wires[0], wires[1]])
	qml.PauliX(wires=wires[0])
	qml.CRX(params[1], wires=[wires[0], wires[1]])
```

**Training**<br/>
The function `circuit_training` run the training of the QCNN.<br/>
At the beginning, the number of required parameters is calculated based on the number of classes. If there are 4, 6, or 8 classes, then an additional pooling layer is needed, thus two more parameters are added.
```python
"""
It executes the training of the QNN
param X_train: X training set
param Y_train: Y training set
param U_params: number of parameters which implement a single block of the F_2 circuit
param embedding_type: the encoding method used
param num_classes: number of classes
param num_layer: number of repetitions of conv layer
param loadParams: if True the parameters are loaded from a file (used to continue a stopped training)
param optimizer: the optimizer used
param learning_rate: learning rate of the optimizer
param epochs: number of epochs
param all_samples: it all the samples are used
param batch_size: size of the batches
param seed: if None a random seed is used, otherwise the value in the variable
return params: the trained parameters
"""
def circuit_training(X_train, Y_train, U_params, embedding_type, num_classes, num_layer, loadParams, optimizer, learning_rate, epochs, all_samples, batch_size, seed):
	if seed != None:
		np.random.seed(seed)
		anp.random.seed(seed)
	
	#calculate the number of parameters
	if num_classes == 10:
		total_params =	((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer)
	else: #we have to add another pooling layer at the end, so we need two parameters
		total_params =	((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer) + 2

```
Then, if we want to restart the training and we saved them, we can use the `loadParams` variable to load the previous parameters, otherwise the parameters are generated randomly:
```python
  #load the parameters
  if not loadParams:
    params = np.random.randn(total_params, requires_grad=True)
  else:
    fileParams = open('params' + 'L' + str(num_layer) + 'LR' + str(learning_rate) + optimizer + 'C' + str(num_classes) + str(all_samples) + '.obj', 'rb')

    params = pickle.load(fileParams)
    fileParams.close()
    print(params)
```
The optimizer is chosen:
```python
  #choose the optimizer
  if optimizer == 'Adam':
    opt = qml.AdamOptimizer(stepsize=learning_rate)
  elif optimizer == 'GDO':
    opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
  else:
    opt = qml.QNGOptimizer(stepsize=learning_rate)
```
The training starts. It will run for a number of epochs equal to `epochs`. For each epoch, it will take the images in batches of size `batch_size` and perform training. At the end of each epochs the new parameters are saved in a file. Ath the end of the training the trained parameters are returned.
```python
  for e in range(0, epochs):
    print("EPOCH: ", e)
    for b in range(0, len(X_train), batch_size):
      if (b + batch_size) <= len(X_train):
        X_batch = [X_train[i] for i in range(b, b + batch_size)]
        Y_batch = [Y_train[i] for i in range(b, b + batch_size)]
      else:
        X_batch = [X_train[i] for i in range(b, len(X_train))]
        Y_batch = [Y_train[i] for i in range(b, len(X_train))]

      if optimizer == 'QNGO': 
        metric_fn = lambda p: qml.metric_tensor(QCNN, approx="block-diag")(X_batch, p, U_params, embedding_type, num_classes, num_layer)
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U_params, embedding_type, num_classes, num_layer),
                                                    params, metric_tensor_fn=metric_fn)
      else:
      params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U_params, embedding_type, num_classes, num_layer),
                                                    params)
			
			
      if b % (batch_size * 100) == 0:
        print("iteration: ", b, " cost: ", cost_new)
				
			
				
    #save the novel parameters at the end of each epoch	
    fileParams = open('params' + 'L' + str(num_layer) + 'LR' + str(learning_rate) + optimizer + 'C' + str(num_classes) + str(all_samples) + '.obj', 'wb')


    pickle.dump(params, fileParams)
    fileParams.close()
  return params
```

**Test**
The testing of the QCNN is performed. The `accuracy_multi` function is used to calculate the accuracy of the network on the test set, while the `accuracy_test_multiclass` function is used to calculate the Confusion Matrix and then calculate Precision, Recall and F1 Score.
```python
"""
It computes the accuracy on the test set
param predictions: classes predicted
param labels: true classes
param num_classes: number of classes
return accuracy: accuracy 
"""
def accuracy_multi(predictions, labels, num_classes):
	correct_predictions = 0

	
	for l, p in zip(labels, predictions):
		p2 = []
		for i in range(0, num_classes):
			p2.append(p[i])
		predicted_class = np.argmax(p2)	# Find the index of the predicted class with highest probability
		if predicted_class == l:
			correct_predictions += 1

	accuracy = correct_predictions / len(labels)
	return accuracy

"""
It computes the precision, recall, F1 score and Confusion Matrix on the test set
param predictions: classes predicted
param labels: true classes
param num_classes: number of classes
return accuracy: accuracy 
"""
def accuracy_test_multiclass(predictions, label, num_classes):
	#confusion matrix
	
	preds_np = np.array(predictions)
	preds = np.argmax(preds_np[:, :num_classes], axis = 1)
	
	conf_mat = multilabel_confusion_matrix(label, preds, labels = list(range(num_classes)))
	print(conf_mat)
	precision = []
	recall = []
	f1 = []
	i = 0
	for c in conf_mat:
		precision.append(c[1][1] / (c[1][1] + c[0][1]))
		recall.append(c[1][1] / (c[1][1] + c[1][0]))
		f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + np.finfo(float).eps))
		
		print("precision " + str(i) + ": " + str(precision[i])) 
		print("recall " + str(i) + ": " + str(recall[i])) 
		print("f1 " + str(i) + ": " + str(f1[i])) 
		i += 1

predictions = []
				
for x in X_test:	
	predictions.append(QCNN(x, trained_params, U_params, encoding, num_classes, num_layer))
							
accuracy = accuracy_multi(predictions, Y_test, num_classes)
print("Accuracy: " + str(accuracy))
accuracy_test_multiclass(predictions, Y_test, num_classes)
```
