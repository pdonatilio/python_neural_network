# Neural Network with PyBrain in Python

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer


# dataset input to training, we need give the dimentions
dataset = SupervisedDataSet(2,1)

# adding the samples
dataset.addSample([1,1],[0])
dataset.addSample([1,0],[1])
dataset.addSample([0,1],[1])
dataset.addSample([1,1],[0])

# Building a FeedFoward network
network = buildNetwork(dataset.indim, 2, dataset.outdim, bias=True)

# Use the back propagation to train the network
trainer = BackpropTrainer(network, dataset, learningrate=0.01, momentum=0.99)

# Training the network
for epoch in range(10000):
    trainer.train()

# Testing the network
test_data = SupervisedDataSet(2,1)
test_data.addSample([1,1],[0])
test_data.addSample([1,0],[1])
test_data.addSample([0,1],[1])
test_data.addSample([1,1],[0])

trainer.testOnData(test_data, verbose=True)