'''
Using a genetic algorithm to get a neural net to play the prison dilemma.
'''

import nn


# James Lecomte
# Started     06/06/18
# Last edit   06/06/18


def main():
    '''Run the program.'''
    genetic_algorithm = nn.GeneticAlgorithm(100, (18, 14, 9))
    genetic_algorithm.next_generation()

if __name__ == "__main__":
    main()
