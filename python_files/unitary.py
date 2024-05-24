import pennylane as qml

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

# Pooling Layer

def Pooling_ansatz1(params, wires): #2 params
	qml.CRZ(params[0], wires=[wires[0], wires[1]])
	qml.PauliX(wires=wires[0])
	qml.CRX(params[1], wires=[wires[0], wires[1]])

