import torch
import numpy as np
import sys

class Hebbian_Learning():
    def __init__(self, hebbian_learning_rate):
        self.iterating_counts = 0
        self.past_products = 0
        self.learning_rate = hebbian_learning_rate

    def average_products(self):
        if self.iterating_counts == 0:
            return 0
        else:
            return self.past_products / self.iterating_counts

    def add_count(self):
        self.iterating_counts += 1

    def add_product(self, product):
        self.past_products += product

    def get_learning_rate(self, sample_size):
        return self.learning_rate * pow(0.95, self.iterating_counts // sample_size)

    def hebbian_train(self, model, inputs, sample_size, device):
        for input in inputs:
            hidden_feature = model(input, path='half1')
            hidden_feature = hidden_feature.cpu().detach().numpy()[0].reshape(model.n_hidden_neuron, 1)
            input = input.cpu().detach().numpy().reshape(1, 784)

            # Activation Threshold
            delta = self.get_learning_rate(sample_size) * np.subtract(hidden_feature * input, self.average_products())

            # Gradient Threshold
            threshold_1 = np.percentile(delta, 90)
            threshold_2 = np.percentile(delta, 10)

            delta = np.add(np.where(delta >= threshold_1, delta, 0), np.where(delta <= threshold_2, delta, 0))

            # Replace parameters in-place
            state_dict = model.state_dict()
            parameters = state_dict['features.1.weight'].cpu().detach().numpy()
            state_dict['features.1.weight'] = torch.from_numpy(np.subtract(parameters, delta)).to(device)
            model.load_state_dict(state_dict)

            if self.iterating_counts % 500 == 0:
                np.set_printoptions(threshold=sys.maxsize)
                print(self.iterating_counts, self.get_learning_rate(sample_size))
                print(np.count_nonzero(delta < 0), np.count_nonzero(parameters < 0))
                print(delta.mean(), parameters.mean())
                print()

            # Add count
            self.add_count()
            self.add_product(hidden_feature * input)

        return model
