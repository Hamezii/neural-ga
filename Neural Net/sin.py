"""
A sin approximation program.
"""

import pygame
from numpy import sqrt, sin, pi
from numpy.random import randint

import drawing
import graphing
import nn

WHITE = (220, 220, 220)
BLACK = (25, 25, 25)
DARK_GRAY = (55, 55, 55)
GRAY = (120, 120, 120)

ACCURACY = 20

GA_STATS = {
    "mutation_rate": 0.4,
    "mutation_chance": 0.1,
    "mutation_amount": 0.3,
    "elitism_offspring": 0.05
    }

MAX_FITNESS = (ACCURACY+1) * 2

def main():
    """Run the program."""
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()

    gen_size = 100
    net_size = (1, 20, 20, 1)
    genetic_algorithm = nn.GeneticAlgorithm(gen_size, net_size)
    genetic_algorithm.set_settings(**GA_STATS)

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
            if genetic_algorithm.stats.max_fitness > MAX_FITNESS - 2:
                difference = MAX_FITNESS + 0.1 - genetic_algorithm.stats.max_fitness
                genetic_algorithm.set_settings(mutation_chance = difference * 0.04, mutation_amount = difference * 0.05)
            else:
                genetic_algorithm.set_settings(**GA_STATS)

            do_generation(genetic_algorithm)

            screen.fill((220, 220, 220))
            draw_neural_nets(screen, genetic_algorithm)
            pygame.display.update()

            print("Gen", genetic_algorithm.stats.generation)
            print("Best fitness", genetic_algorithm.stats.max_fitness)

            if genetic_algorithm.stats.max_fitness > MAX_FITNESS - 0.1:
                done = True
                for net in genetic_algorithm.sorted_population[:10]:
                    print(net.fitness)

def fitness_function(neural_net):
    """Calculate the fitness of a neural net."""
    fitness = 0
    for i in range(ACCURACY + 1):
        val = i * pi * (2 / ACCURACY)
        calculation = calculate(neural_net, val)
        answer = sin(val) + 1
        fitness += 2 - abs(answer - calculation)
    return fitness

def calculate(neural_net, x):
    """Calculate with a neural net."""
    return neural_net.calculate([x])[0]

def do_generation(genetic_algorithm):
    """Run a generation of the genetic algorithm."""
    for neural_net in genetic_algorithm.population:
        neural_net.fitness = fitness_function(neural_net)

    genetic_algorithm.next_generation()

def draw_neural_nets(surface, genetic_algorithm):
    """Draw the whole sorted population."""
    for i in range(3):
        net = genetic_algorithm.sorted_population[i]
        drawing.draw_neural_net(surface, net, (20, 20 + i*200), (160, 160))
        vals = []
        for j in range(ACCURACY + 1):
            x = j * pi * (2 / ACCURACY)
            vals.append(calculate(net, x))
        rect = pygame.Rect((200, 20 + i*200), (400, 160))
        graphing.draw_line_graph(surface, (WHITE, GRAY, DARK_GRAY), vals, rect=rect)
        


if __name__ == "__main__":
    main()
