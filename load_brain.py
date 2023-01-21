import numpy as np
import environment
import test_environment


class Candidate():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias


track = input("Track: ")  # train/test
while True:
    generation = input("Generation: ")
    weights = np.load("brains/generation"+generation +
                      "-weights.npy", allow_pickle=True)
    bias = np.load("brains/generation"+generation +
                   "-bias.npy", allow_pickle=True)
    candidate = Candidate(weights, bias)
    if track == "train":
        environment.simulate([candidate])
    elif track == "test":
        test_environment.simulate([candidate])
