import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

def data_load_and_process(feature_reduction, num_classes, all_samples, seed):
	if seed != None:
		tf.random.set_seed(seed)
		np.random.seed(seed)

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0	 # normalize the data

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

		# Shuffle the data
		shuffle_indices = np.random.permutation(len(x_train_subset))
		x_train = x_train_subset[shuffle_indices]
		y_train = y_train_subset[shuffle_indices]

	print("Shape of subset training data:", x_train.shape)
	print("Shape of subset training labels:", y_train.shape)

	mask_train = np.isin(y_train, range(0, num_classes))
	mask_test = np.isin(y_test, range(0, num_classes))

	X_train = x_train[mask_train]
	X_test = x_test[mask_test]		
	Y_train = y_train[mask_train]
	Y_test = y_test[mask_test]

	print("Shape of subset training data:", X_train.shape)
	print("Shape of subset training labels:", Y_train.shape)

	if feature_reduction == 'amplitude':
		
		X_train_flat = X_train.reshape(X_train.shape[0], -1)
		X_test_flat = X_test.reshape(X_test.shape[0], -1)
		pca = PCA(n_components = 256)
		X_train = pca.fit_transform(X_train_flat)
		X_test = pca.transform(X_test_flat)
		return X_train, X_test, Y_train, Y_test

	elif feature_reduction == 'angle':

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
