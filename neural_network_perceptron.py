# Neural Network Perceptron Implementation in Python
'''
    u = w * x(k)
    u = g(u) -> y

    The algorithm:
    w = w + N * (d(k) -y) * x(k)
'''

import random
import copy

class Perceptron:

    def __init__(self, samples, outputs, l_rate=0.1, n_epoch=1000, threshold=-1):

        self.samples = samples  # All Samples
        self.outputs = outputs  # Outputs about all samples
        self.l_rate = l_rate  # Learning Rate between 0 and 1
        self.n_epoch = n_epoch  # Number of Epochs
        self.threshold = threshold  # threshold

        self.t_samples = len(samples)  # quantity total of samples - Total of Samples
        self.tb_sample = len(samples[0])  # quantity total by one sample - Total By Sample
        self.weights = []  # Weight array

    # Training the Neural Network
    def training(self):
        # add -1 for each one sample
        for sample in self.samples:
            sample.insert(0, -1)

        # Start the weights array with random values
        for i in range(self.tb_sample):
            self.weights.append(random.random())

        # Insert the threshold in weights array
        self.weights.insert(0, self.threshold)

        # Start the epoch counter
        epoch_counter = 0

        while True:
            error = False  # At the start the error not exists

            for i in range(self.t_samples):
                u = 0

                # We need to sum the limit because in the samples we init with -1 (line 30)
                for j in range(self.tb_sample + 1):
                    u += self.weights[j] * self.samples[i][j]

                # Get the network output using the activation function (Unit Step Function)
                y = self.signal(u)

                # Check if the output predicted is different of the output expected
                if y != self.outputs[i]:
                    error_result = self.outputs[i] - y  # calculate the error expected - predicted: (d(k) -y)

                    # Make the adjustments for each sample element
                    for j in range(self.tb_sample + 1):
                        # w = w + N * (d(k) -y) * x(k)
                        self.weights[j] = self.weights[j] + self.l_rate * error_result * self.samples[i][j]

                    error = True  # Still have an error

            # Increase the epoch counter
            epoch_counter += 1

            # How to stop the loop? If we bit the epoch quantity or no find errors anymore
            if epoch_counter > self.n_epoch or not error:
                break

    # get the samples to be classified and the name of classes
    # use the singal function to identify the class about each sample
    def test(self, sample, class1, class2):

        # Insert the -1
        sample.insert(0, -1)

        # Get the weight array after the training step
        u = 0
        for i in range(self.tb_sample + 1):
            u += self.weights[i] * sample[i]

        # Calculate the network output
        y = self.signal(u)

        if y == -1:
            print('The sample belongs to %s' % class1)
        else:
            print('The sample belongs to %s' % class2)

    # Activation Function: Unit Step Function (the output signal)
    def signal(self, u):
        return 1 if u >= 0 else -1


# Testing the class
print("\nA or B?\n")

# Using 4 samples to test
samples = [[0.1, 0.4, 0.7], [0.3, 0.7, 0.2], [0.6, 0.9, 0.8], [0.5, 0.7, 0.1]]

# The outputs expecteds for each sample
outputs = [1, -1, -1, 1]

# 
tests = copy.deepcopy(samples)

# create the perceptron network
network = Perceptron(samples = samples, outputs = outputs)

network.training()

for test in tests:
    network.test(test, 'A', 'B')