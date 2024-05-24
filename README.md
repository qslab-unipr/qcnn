This repository contains the source code of the paper "Multi-Class Quantum Convolutional Neural Networks" https://arxiv.org/abs/2404.12741
submitted to the QUASAR Workshop (https://sites.google.com/view/quasar24/home?authuser=0) in conjunction with the 33rd ACM International
Symposium on High-Performance Parallel and Distributed Computing.

If you want to execute the code step by step run the qcnn.ipynb file. In the first block of the notebook, you can set the different variables.

If you want to execute all the code together, you can run the python_files/main.py file. 
In this case, when you execute the code, you can set different argument to modify the value of the hyperparameters, e.g.: <br />
python ./main.py --epochs 2 --opt 'Adam' --lr 0.01 --batch_size 32 --num_classes 10 <br />
This will execute the QCNN with 2 epochs, with Adam as optimizer with 0.01 as learning rate. The batch size is 64 and it will take all the classes of the MNIST dataset.

The default parameters are: <br />
epochs: 2 <br />
batch_size: 64 <br />
lr: 0.01 <br />
opt: 'Adam' <br />
num_classes: 10 <br />
encoding: 'amplitude' <br />
num_layer: 1 <br />
all_samples: False <br />
seed: None <br />

To get the same results of the paper, run the code with the default value, setting all_samples = True and seed = 43 and changing the number of classes.
In this way, you will obtain the result when all the samples of the dataset are used.
Otherwise, you have to change the number of epochs to 20, 40, 60,... and the seed to 43.
In this way, you will obtain the results when only 250 samples of the dataset are used.


file: <br />
python_files/main.py contains the main file. To execute the qcnn, you need to execute this file. You can set different parameters such as number of epochs,
learning rate, optimizer, etc. <br />
python_files/data.py contains the loading and processing of the MNIST dataset <br />
python_files/QCNN_circuit.py contains the structure of QCNN  <br />
python_files/unitary.py implements the single circuits which implement the convolutional and the pooling layer <br />
python_files/Training.py implements the training process of the network. <br />
qcnn.ipynb contains all the code as a notebook
