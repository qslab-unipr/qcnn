import pennylane as qml
import numpy as np
import unitary
import inspect
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
	
n_qubit = 8

# Quantum Circuits for Convolutional layers
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
				unitary.U_SU4(params[param0: param1], wires = [qubits[i % len(qubits)], qubits[(i + 1) % len(qubits)]])
			
			for i in range(1, len(qubits), 2):
				unitary.U_SU4(params[param0: param1], wires = [qubits[i % len(qubits)], qubits[(i + 1) % len(qubits)]])

			param0 = param1
			param1 += len(qubits) * 2
			

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
	V(params, wires=[7, 6]) 
	V(params, wires=[1, 0]) 

def pooling_layer2(V, params):
	V(params, wires=[3, 2]) 
	V(params, wires=[5, 4]) 

def pooling_layer3(V, params, num_classes):
	if num_classes == 4:
		V(params, wires=[2,0])				   
	V(params, wires=[6,4])


def QCNN_structure(U, params, U_params, num_classes, num_layer):
	
	param1CL = params[0: (U_params + n_qubit * 2) * num_layer]
	param1PL = params[(U_params + n_qubit * 2) * num_layer: ((U_params + n_qubit * 2) * num_layer) + 2]
		
	param2CL = params[((U_params + n_qubit * 2) * num_layer) + 2: ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer)]
	param2PL = params[((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer): 
					  ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2]

	param3CL = params[((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2: 
					  ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer)]

	conv_layer(U, param1CL, U_params, num_layer, range(n_qubit))
	pooling_layer1(unitary.Pooling_ansatz1, param1PL)
	
	conv_layer(U, param2CL, U_params, num_layer, [0, 2, 3, 4, 5, 6])
	pooling_layer2(unitary.Pooling_ansatz1, param2PL)
	
	conv_layer(U, param3CL, U_params, num_layer, [0, 2, 4, 6])

	if num_classes == 4 or num_classes == 6 or num_classes == 8:
		
		param3PL = params[((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer):
						  ((U_params + n_qubit * 2) * num_layer) + 2 + ((U_params + (n_qubit - 2) * 4) * num_layer) + 2 + ((U_params + n_qubit * 2) * num_layer) + 2]	  

		pooling_layer3(unitary.Pooling_ansatz1, param3PL, num_classes)
	

dev = qml.device('default.qubit', wires = n_qubit)
@qml.qnode(dev)
def QCNN(X, params, U_params, embedding_type='amplitude', cost_fn='cross_entropy', num_classes=10, num_layer = 1):


	# Data Embedding
	if embedding_type == 'amplitude':
		AmplitudeEmbedding(X, wires=range(8), normalize=True)
	elif embedding_type == 'angle':
		AngleEmbedding(X, wires=range(8), rotation='Y')
	
	
	QCNN_structure(unitary.CC14, params, U_params, num_classes, num_layer)


	
	if num_classes == 4:
		result = qml.probs(wires=[0, 4])
	elif num_classes == 6:
		result = qml.probs(wires=[0, 2, 4])
	elif num_classes == 8:
		result = qml.probs(wires=[0, 2, 4])
	else:
		result = qml.probs(wires=[0, 2, 4, 6])			
					
	return result
