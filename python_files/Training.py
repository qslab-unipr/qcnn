# Implementation of Quantum circuit training procedure
import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
import pickle

n_qubit = 8

def cross_entropy(labels, predictions, num_classes):
	epsilon = 1e-15
	num_samples = len(labels)
	
	
	
	
	num_classes = len(predictions[0])
	Y_true_one_hot = anp.eye(num_classes)[labels]

	loss = 0.0
	for i in range(num_samples):
		predictions[i] = anp.clip(predictions[i], epsilon, 1 - epsilon)
		loss -= anp.sum(Y_true_one_hot[i] * anp.log(predictions[i]))		
	
	
	return loss / num_samples
	
def cost(params, X, Y, U_params, embedding_type, cost_fn, num_classes, circ_layer):
	predictions = [QCNN_circuit.QCNN(x, params, U_params, embedding_type, cost_fn=cost_fn, num_classes=num_classes, num_layer = circ_layer) for x in X]
	
	
	loss = cross_entropy(Y, predictions, num_classes)
	
	return loss

def circuit_training(X_train, Y_train, U_params, embedding_type, cost_fn, num_classes, num_layer, loadParams, optimizer, learning_rate, epochs, all_samples, batch_size, seed):
	if seed != None:
		np.random.seed(seed)
		anp.random.seed(seed)
	
	if num_classes == 10:
		total_params =	((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer)
	else:
		total_params =	((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer) + 2
	
	
	if not loadParams:
		params = np.random.randn(total_params, requires_grad=True)
	else:
		fileParams = open('params' + 'L' + str(num_layer) + 'LR' + str(learning_rate) + optimizer + 'C' + str(num_classes) + str(all_samples) + '.obj', 'rb')

		params = pickle.load(fileParams)
		fileParams.close()
		print(params)
	
	if optimizer == 'Adam':
		opt = qml.AdamOptimizer(stepsize=learning_rate)
	elif optimizer == 'GDO':
		opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
	else:
		opt = qml.QNGOptimizer(stepsize=learning_rate)
	loss_history = []
	grad_vals = []
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
				metric_fn = lambda p: qml.metric_tensor(QCNN_circuit.QCNN, approx="block-diag")(X_batch, p, U_params, embedding_type, cost_fn, num_classes, num_layer)
				params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U_params, embedding_type, cost_fn, num_classes, num_layer),
														 	params, metric_tensor_fn=metric_fn)
			else:
				params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U_params, embedding_type, cost_fn, num_classes, num_layer),
														 	params)
			
			
			if b % (batch_size * 10) == 0:
				print("iteration: ", b, " cost: ", cost_new)
				"""
				loss_history.append(cost_new)
				gradient_fn = qml.grad(cost)
				gradients = gradient_fn(params, X_batch, Y_batch, U, U_params, embedding_type, cost_fn, num_classes, num_layer)
				grad_vals.append(gradients[-1])
				print(gradients)
				print("var ", np.var(grad_vals))
				print("mean grad: ", np.mean(grad_vals))
				"""
			
				
				
		fileParams = open('params' + 'L' + str(num_layer) + 'LR' + str(learning_rate) + optimizer + 'C' + str(num_classes) + str(all_samples) + '.obj', 'wb')


		pickle.dump(params, fileParams)
		fileParams.close()
	return loss_history, params


