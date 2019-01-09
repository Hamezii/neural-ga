"""
Testing the genetic algorithm as well as the drawing module using adding.
"""

import pygame
from numpy import sqrt
from numpy.random import randint

import drawing
import nn



# James Lecomte
# Started     04/06/18
# Last edit   06/06/18


def main():
    """Run the program."""
    pygame.init()
    screen = pygame.display.set_mode((700, 700))
    clock = pygame.time.Clock()

    gen_size = 49
    net_size = (2, 1)
    genetic_algorithm = nn.GeneticAlgorithm(gen_size, net_size, mutation_rate=0.4, mutation_chance=0.2, mutation_amount=0.1, elitism_offspring=0)

    done = False
    while True:
        clock.tick(sqrt(genetic_algorithm.current_generation)+1)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit(0)
            if event.type == pygame.QUIT:
                quit(0)

        if not done:
            do_generation(genetic_algorithm)

            screen.fill((220, 220, 220))
            draw_neural_nets(screen, genetic_algorithm)
            pygame.display.update()

            print("Gen", genetic_algorithm.stats.generation)
            print("Best fitness", genetic_algorithm.stats.max_fitness)

            if genetic_algorithm.stats.max_fitness > -0.1:
                done = True
                for net in genetic_algorithm.sorted_population:
                    print(net.fitness)
                best_neural_net = genetic_algorithm.sorted_population[0]
                print("Weights:")
                for i, layer in enumerate(best_neural_net.layers, 1):
                    print("Layer", i)
                    print(layer.weights)

def fitness_function(neural_net):
    """Calculate the fitness of a neural net."""
    fitness = 0
    for _ in range(25):
        i = randint(6)
        j = randint(6)
        calculation = neural_net.calculate((i, j))[0]
        answer = i + j
        fitness -= abs(answer - calculation)
    return fitness

def do_generation(genetic_algorithm):
    """Run a generation of the genetic algorithm."""
    for neural_net in genetic_algorithm.population:
        neural_net.fitness = fitness_function(neural_net)

    genetic_algorithm.next_generation()

def draw_neural_nets(surface, genetic_algorithm):
    """Draw the whole sorted population."""
    for i in range(genetic_algorithm.population_size):
        x = i%7
        y = i//7
        drawing.draw_neural_net(surface, genetic_algorithm.sorted_population[x + y*7], (10 + x*100, 10 + y*100), (80, 80))


if __name__ == "__main__":
    main()
