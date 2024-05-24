# This generates the results of the bechmarking code
import argparse
import data
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import Training
import QCNN_circuit

def get_args():
	parser = argparse.ArgumentParser()	 

	parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')

	parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='number of elements in batch size')

	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--opt', type=str, default='Adam', choices=['QNGO', 'Adam', 'GDO'], help = 'optimizer used for training')

	parser.add_argument('--loss', type=str, default = 'cross_entropy', choices=['cross_entropy'], help='loss function')
	
	parser.add_argument('--num_classes', type=int, default = 10, choices=[4, 6, 8, 10], help='num classes')
	
	parser.add_argument('--encoding', type=str, default = "amplitude", choices=['amplitude', 'angle'], help='encoding type')

	parser.add_argument('--num_layer', type=int, default = 1, choices=range(1, 10), help = 'number of circuit repetition')
	parser.add_argument('--load_params', type=bool, default=False, choices=[True, False], help='continue training from saved parameters')

	parser.add_argument('--all_samples', type=bool, default=False, choices=[False, True], help="how many samples of the dataset")

	parser.add_argument('--seed', type = int, default = None, help = "set seed for reproducibility, if not set, it is random")


	return parser.parse_args()
	

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

def main(args):
	
	U_params = 15

	X_train, X_test, Y_train, Y_test = data.data_load_and_process(args.encoding, args.num_classes, args.all_samples, args.seed)

	print("\n")
	print("Loss History for  circuit with " + args.encoding + " with " + args.loss)
	loss_history, trained_params = Training.circuit_training(X_train, Y_train, U_params, args.encoding, args.loss, args.num_classes, args.num_layer, args.load_params,
															 args.opt, args.lr, args.epochs, args.all_samples, args.batch_size, args.seed)

	predictions = []
				
	for x in X_test:	
		predictions.append(QCNN_circuit.QCNN(x, trained_params, U_params, args.encoding, args.loss, args.num_classes, args.num_layer))
							
	accuracy = accuracy_multi(predictions, Y_test, args.num_classes)
	print("Accuracy: " + str(accuracy))
	accuracy_test_multiclass(predictions, Y_test, args.num_classes)

if __name__ == "__main__":
	args = get_args()
	print(args)
	main(args)

