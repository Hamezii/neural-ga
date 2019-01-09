"""
Contains logic for the genetic algorithm.
"""

from operator import attrgetter

import numpy as np



# Notes to self:
# - Consider using LReLU or ReLU for hidden layers and something like Maxout for output layer.

# James Lecomte
# Started     31/03/18
# Last edit   08/06/18

# py -m cProfile -s tottime nn.py



def relu(x):
    """ReLU activation function."""
    return max(0, x)

def lrelu(x):
    """Leaky ReLU activation function."""
    return max(0.1*x, x)



class NeuralNetLayer:
    """A layer of a neural net.

    Weight matrix:
     Rows- Output nodes
     Columns- Input nodes, bias weight is last column
    """

    def __init__(self, input_size, output_size, weights=None):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        if not weights is None:
            self._set_weights(weights)
        self._activation_function = lrelu

    def randomize_weights(self):
        """Randomize the weights."""
        self.weights = np.random.uniform(-1, 1, (self.output_size, self.input_size+1))

    def calculate(self, inputs):
        """Calculate an output using weights and the activation function."""
        inputs_with_bias = (*inputs, 1)
        return [self._activation_function(num) for num in self.weights @ inputs_with_bias] # @ means matrix multiplication

    def _set_weights(self, weights):
        """Set the weights of the layer given a 1d array."""
        self.weights = weights.reshape(self.output_size, self.input_size+1)

    def copy(self):
        """Return a NeuralNetLayer with the same properties."""
        clone = NeuralNetLayer(self.input_size, self.output_size)
        clone.weights = self.weights.copy()
        return clone

class NeuralNet:
    """A neural net containing layers and a fitness."""

    def __init__(self):
        self.layers = []
        self.fitness = 0

    def calculate(self, inputs):
        """Pass an input through all the layers."""
        output = inputs
        for layer in self.layers:
            output = layer.calculate(output)
        return output

    def mutate(self, chance, amount):
        """Mutate the weights of the layers."""
        for layer in self.layers:
            for row in range(layer.output_size):
                for col in range(layer.input_size+1):
                    if np.random.rand() < chance:
                        new_val = layer.weights[row, col] + np.random.uniform(-amount, amount)
                        new_val = min(max(-1, new_val), 1)
                        layer.weights[row, col] = new_val


    def randomize_layers(self, *sizes):
        """Make random layers."""
        self.layers = []
        for i in range(len(sizes)-1):
            layer = NeuralNetLayer(sizes[i], sizes[i + 1])
            layer.randomize_weights()
            self.layers.append(layer)

    def copy(self):
        """Return a neural net with the same layers."""
        clone = NeuralNet()
        for layer in self.layers:
            clone.layers.append(layer.copy())
        return clone

class GeneticAlgorithmStats:
    """Stores stats about a genetic algorithm."""
    def __init__(self):
        self.mean_fitness = 0
        self.median_fitness = 0
        self.max_fitness = 0
        self.min_fitness = 0
        self.generation = 0

    def calculate(self, genetic_algorithm):
        """Calculate stats about a genetic algorithm."""
        # Returning if stats have already been calculated
        if self.generation == genetic_algorithm.current_generation:
            return
        # Otherwise calculating stats
        self.max_fitness = genetic_algorithm.sorted_population[0].fitness
        self.min_fitness = genetic_algorithm.sorted_population[-1].fitness
        self.mean_fitness = np.mean(tuple(net.fitness for net in genetic_algorithm.population))
        self.median_fitness = np.median(tuple(net.fitness for net in genetic_algorithm.population))

        self.generation = genetic_algorithm.current_generation

class GeneticAlgorithm:
    """The main system."""

    DEFAULT_SETTINGS = {
        "elitism_offspring":    0.2,
        "random_offspring":     0,

        "mutation_rate":        0.1,
        "mutation_chance":      0.1,
        "mutation_amount":      0.1,
    }


    def __init__(self, population_size, layer_sizes, **settings):
        self.population_size = population_size
        self.layer_sizes = layer_sizes
        self.population = tuple(self._random_child() for _ in range(self.population_size))
        self.current_generation = 1

        self.settings = self.DEFAULT_SETTINGS
        self.set_settings(**settings)

        self.stats = GeneticAlgorithmStats()

        self.sorted_population = ()


    def set_settings(self, **settings):
        """Change the value of the settings."""
        for key in settings:
            if not key in self.DEFAULT_SETTINGS:
                raise ValueError(str(key) + " is not a valid setting")
        self.settings = {**self.settings, **settings}


    def calculate_stats(self):
        """Calculate stats on the current generation."""
        # Returning if stats have already been calculated
        if self.current_generation == self.stats.generation:
            return
        self.sorted_population = sorted(self.population, key=attrgetter("fitness"), reverse=True)
        self.stats.calculate(self)


    def next_generation(self):
        """Simulate natural selection and gene crossover to get a new population."""
        self.calculate_stats()

        self.population = []

        # Getting amounts for different types of neural net replacements
        random_size = self.random_round(self.population_size * self.settings["random_offspring"])
        elitism_size = self.random_round(self.population_size * self.settings["elitism_offspring"])
        crossover_size = self.population_size - random_size - elitism_size

        # Keeping best neural nets (elitism)
        self.population.extend(self.sorted_population[i].copy() for i in range(elitism_size))

        # Adding neural nets with crossover

        probs = self._get_selection_probabilities()
        crossovers = (self._uniform_crossover(*np.random.choice(self.sorted_population, 2, replace=False, p=probs)) for _ in range(crossover_size))
        self.population.extend(crossovers)

        # Mutating neural nets
        for neural_net in self.population:
            if np.random.rand() < self.settings["mutation_rate"]:
                neural_net.mutate(self.settings["mutation_chance"], self.settings["mutation_amount"])

        # Adding random nets
        self.population.extend(self._random_child() for _ in range(random_size))

        # Shuffling new population
        np.random.shuffle(self.population)

        # Increment current generation
        self.current_generation += 1

    def _get_selection_probabilities(self):
        """Return the array of probabilities of each creature in the population."""
        probabilities = np.arange(1, self.population_size+1, dtype=float)[::-1]
        probabilities /= probabilities.sum()
        return probabilities

    def _slice_crossover(self, net_a: NeuralNet, net_b: NeuralNet):
        """Return a neural net by using single point crossover."""
        child_net = NeuralNet()

        crossover_layer = np.random.randint(len(self.layer_sizes)-1)

        # Adding layers before crossover layer
        child_net.layers.extend(net_a.layers[i].copy() for i in range(crossover_layer))

        # Slicing layer and putting together the halves
        weights_a = net_a.layers[crossover_layer].weights.flatten()
        weights_b = net_b.layers[crossover_layer].weights.flatten()
        layer_size = (self.layer_sizes[crossover_layer:crossover_layer+2])
        crossover_weight = np.random.randint(1, weights_a.size)
        child_weights = np.append(weights_a[:crossover_weight], weights_b[crossover_weight:])
        child_layer = NeuralNetLayer(*layer_size, child_weights)
        child_net.layers.append(child_layer)

        # Adding layers after crossover layer
        child_net.layers.extend(net_b.layers[i].copy() for i in range(crossover_layer+1, len(self.layer_sizes)-1))
        return child_net

    def _uniform_crossover(self, net_a: NeuralNet, net_b: NeuralNet):
        """Return a neural net by using uniform crossover."""
        child_net = NeuralNet()
        for layer_num in range(len(self.layer_sizes)-1):
            layer_size = (self.layer_sizes[layer_num:layer_num+2])
            child_layer = NeuralNetLayer(*layer_size)

            weights_a = net_a.layers[layer_num].weights
            weights_b = net_b.layers[layer_num].weights
            random_mask = np.random.binomial(1, 0.5, (layer_size[1], layer_size[0]+1))

            child_layer.weights = weights_a.copy()
            child_layer.weights[random_mask == 1] = weights_b[random_mask == 1]
            child_net.layers.append(child_layer)
        return child_net

    def _random_child(self):
        """Return a neural net with random weights."""
        child_net = NeuralNet()
        child_net.randomize_layers(*self.layer_sizes)
        return child_net

    @staticmethod
    def random_round(x):
        """Round in a random direction. Only works for positive numbers."""
        prob = x - int(x)
        int_x = int(x)
        return int_x + (prob > np.random.rand())

def multiplication_test():
    """Run a multiplication test example for the learner system."""

    def fitness_function(neural_net):
        """Calculate the fitness of a neural_net."""
        fitness = 25
        for i in range(1, 6):
            for j in range(1, 6):
                answer = np.exp(neural_net.calculate([np.log(i), np.log(j)])[0])
                result = i*j
                fitness -= abs(answer - result)

        return fitness

    gen_size = 50
    net_size = (2, 1)
    genetic_algorithm = GeneticAlgorithm(gen_size, net_size, mutation_rate=0.3, mutation_chance=0.5)

    highest_so_far = 0
    while True:
        # Testing creatures
        for neural_net in genetic_algorithm.population:
            neural_net.fitness = fitness_function(neural_net)

        # Sorting creatures
        genetic_algorithm.calculate_stats()

        print("Gen", genetic_algorithm.current_generation, ":")
        print("Max fitness", genetic_algorithm.stats.max_fitness)
        print("Mean fitness", genetic_algorithm.stats.mean_fitness)
        highest_so_far = max(genetic_algorithm.stats.max_fitness, highest_so_far)
        print("Highest so far", highest_so_far)


        # Starting next generation
        if genetic_algorithm.stats.max_fitness < 24.9 and genetic_algorithm.current_generation < 1000:
            genetic_algorithm.next_generation()
        else:
            break


    quit()


    for net in genetic_algorithm.sorted_population:
        print(net.fitness)
    best_neural_net = genetic_algorithm.sorted_population[0]
    print("Weights:")
    print(best_neural_net.layers[0].weights[0])
    while True:
        print()
        in_a = input("Give net first number: ")
        in_b = input("Give net second number: ")
        answer = best_neural_net.calculate([np.log(float(in_a)), np.log(float(in_b))])[0]
        print("Net's answer:", np.exp(answer))



if __name__ == "__main__":
    multiplication_test()
