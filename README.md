This repository contains the source code of the paper "Multi-Class Quantum Convolutional Neural Networks" https://arxiv.org/abs/2404.12741
submitted to the QUASAR Workshop (https://sites.google.com/view/quasar24/home?authuser=0) in conjunction with the 33rd ACM International
Symposium on High-Performance Parallel and Distributed Computing.

if you want to execute the code step by step run the qcnn.ipynb file. In the first block of the notebook, you can set the different variables.

If you want to execute all the code together, you can run the python_files/main.py file. 
In this case, when you execute the code, you can set different argument to modify the value of the hyperparameters: 
e.g.
python ./main.py --epochs 2 --opt 'Adam' --lr 0.01 --batch_size 32 --num_classes 10
this will execute the QCNN with 2 epochs, with Adam as optimizer with 0.01 as learning rate. The batch size is 32 and it will take all the classes of the MNIST dataset.

The default parameters are:
epochs: 2
batch_size: 64
lr: 0.01
opt: 'Adam'
num_classes: 10
encoding: 'amplitude'
num_layer: 1
all_samples: False
seed: None

To execute the result of the paper, run the code with the default value, setting all_samples = True and seed =  and changing the number of classes.
In this way, you will obtain the result when all the samples of the dataset are used.
Otherwise, you have to change the number of epochs to 20, 40, 60,... and the seed to .
In this way, you will obtain the results when only 250 samples of the dataset are used.


file:
python_files/main.py contains the main file. to execute the qcnn, you need to execute this file, you can set different parameters such as number of epochs,
learning rate, optimizer etc.

python_files/data.py contains the loading and processing of the MNIST dataset

python_files/QCNN_circuit.py contains the structure of QCNN QCNN_circuit

python_files/unitary.py implements the single circuits which implement the convolutional and the pooling layer

python_files/Training.py implements the training process of the network.

qcnn.ipynb contains all the code as a notebook
