'''
A bunch of creatures going around using a genetic algorithm.
'''

import sys
import numpy as np
import pygame

import drawing
import graphing
import nn


pygame.init()
pygame.display.set_caption('Gen 1 - Creatures')

WIDTH = 1200
HEIGHT = 900
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

WHITE = (220, 220, 220)
BLACK = (25, 25, 25)
DARK_GRAY = (55, 55, 55)
GRAY = (120, 120, 120)
GREEN = (25, 220, 25)
RED = (150, 25, 25)
DARK_BLUE = (25, 25, 100)
BG_COLOR = (180, 180, 180)

class Creature:
    '''A creature.'''
    def __init__(self, index):
        self.x = None
        self.y = None
        self.color = (100, 100, 100)
        self.memory = 0
        self.index = index

    def move(self, pos):
        """Move the creature."""
        self.x, self.y = pos

class Grid:
    '''A cache of food positions which can be accessed.'''
    def __init__(self, size):
        self.size = size
        self.food_mask = np.full(size, False)

    def clear(self):
        '''Clear the cache.'''
        self.food_mask.fill(False)

    def food_pos(self):
        '''Return the positions of all food.'''
        for pos in np.transpose(np.nonzero(self.food_mask)):
            yield pos

    def add_food(self, pos):
        '''Add a food to the cache.'''
        self.food_mask[pos] = True

    def remove_food(self, pos):
        '''Remove food from the cache.'''
        self.food_mask[pos] = False

    def is_food_at(self, pos):
        """Return True if there is food at a specific pos."""
        return self.food_mask[pos]

    def distance_to_food(self, pos, direction):
        '''Return the distance to the nearest food from pos in a direction.'''
        if direction[0] == 1:
            array = self.food_mask[pos[0]+1:, pos[1]]
            pos = pos[0]
            func = np.min
            right = True
        if direction[0] == -1:
            array = self.food_mask[:pos[0], pos[1]]
            pos = pos[0]
            func = np.max
            right = False
        if direction[1] == 1:
            array = self.food_mask[pos[0], pos[1]+1:]
            pos = pos[1]
            func = np.min
            right = True
        if direction[1] == -1:
            array = self.food_mask[pos[0], :pos[1]]
            pos = pos[1]
            func = np.max
            right = False

        indexes = np.nonzero(array)[0]
        if indexes.size == 0:
            return None

        val = func(indexes)
        if right:
            return val + 1

        return array.size - val


class World:
    '''Stores the creatures and the food.'''

    DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    GRID_SIZE = (9, 9)
    POPULATION_SIZE = 50
    MIN_FOOD_AMOUNT = 30
    MAX_FOOD_AMOUNT = 30
    NET_SIZE = (9, 9, 5)
    MAX_STEPS = 90
    MAX_MUTATION_AMOUNT = 0.4
    MUTATION_RATE = 0.2

    def __init__(self):
        self.genetic_algorithm = nn.GeneticAlgorithm(self.POPULATION_SIZE, self.NET_SIZE)
        self.genetic_algorithm.set_settings(mutation_rate=self.MUTATION_RATE, mutation_amount=self.MAX_MUTATION_AMOUNT)

        self.creature = None
        self.grid = Grid(self.GRID_SIZE)
        self.steps = 0

        self.food_amount = self.MIN_FOOD_AMOUNT

        self.highscores = [0]
        self.meanscores = [0]
        
        self.perfects = 0
        self.perfect_streak = 0
        self.best_perfect_streak = 0

        self.next_creature()

    def next_creature(self):
        """Refresh the grid and test the next creature in the population."""
        if self.creature is None:
            self.creature = Creature(0)
        else:
            self.creature = Creature(self.creature.index + 1)


        self.creature.move((int(self.GRID_SIZE[0]/2), int(self.GRID_SIZE[1]/2)))
        self._populate_grid()
        self.steps = 0

    def next_generation(self):
        '''Change values depending on how well the generation did and call next generation for the genetic algorithm.'''
        self.genetic_algorithm.calculate_stats()
        if self.genetic_algorithm.stats.max_fitness == self.food_amount:
            # Perfect generation
            new_mutation_amount = self.genetic_algorithm.settings["mutation_amount"]*0.9
            self.genetic_algorithm.set_settings(mutation_amount=new_mutation_amount)

            self.food_amount = min(self.food_amount + 1, self.MAX_FOOD_AMOUNT)

            self.perfect_streak += 1
            self.perfects += 1
            self.best_perfect_streak = max(self.best_perfect_streak, self.perfect_streak)
        else:
            # Not perfect generation
            new_mutation_amount = self.genetic_algorithm.settings["mutation_amount"]*5
            new_mutation_amount = self.clamp(new_mutation_amount, 0.01, self.MAX_MUTATION_AMOUNT)
            self.genetic_algorithm.set_settings(mutation_amount=new_mutation_amount)

            self.food_amount = max(self.food_amount - 2, self.MIN_FOOD_AMOUNT)

            self.perfect_streak = 0

        self.genetic_algorithm.next_generation()
        self.creature = None
        self.next_creature()

    def random_free_pos(self):
        '''Return a random position on the grid which is empty.'''
        pos = tuple(np.random.randint(axis) for axis in self.GRID_SIZE)
        while self.grid.is_food_at(pos) or pos == (self.creature.x, self.creature.y):
            pos = tuple(np.random.randint(axis) for axis in self.GRID_SIZE)
        return pos

    def _populate_grid(self):
        '''Fill the grid with food.'''
        self.grid.clear()
        for _ in range(self.food_amount):
            self.grid.add_food(self.random_free_pos())

    def run_step(self):
        '''Run a step of simulation.'''
        self.make_creature_act()
        self.steps += 1

    def make_creature_act(self):
        '''Simulate a step of a creature.'''
        creature = self.creature

        net = self.genetic_algorithm.population[creature.index]

        inputs = []
        inputs.extend(self.find_distance_to_edge(direction) for direction in self.DIRECTIONS)
        inputs.extend(self.find_distance_to_food(direction) for direction in self.DIRECTIONS)
        inputs.append(creature.memory)

        output = net.calculate(inputs)
        creature.memory = self.clamp(output.pop(), -1, 1)
        creature.color = (100, 100, 100 + creature.memory*100)
        if all(direction <= 0 for direction in output):
            return # Not moving
        else:
            movement_index = output.index(max(output))
            movement_direction = self.DIRECTIONS[movement_index]

        if inputs[movement_index] == 1:
            return # Bumping into edge

        new_pos = (creature.x + movement_direction[0], creature.y + movement_direction[1])

        if inputs[4+movement_index] == 1:
            net.fitness += 1 # Getting food
            self.grid.remove_food(new_pos)

        self.creature.move(new_pos)

    def find_distance_to_edge(self, direction):
        '''Find the distance to the edge of the world in a specific direction from the creature.'''
        creature = self.creature
        if direction[0] == -1:
            max_distance = self.GRID_SIZE[0]
            distance = creature.x + 1
        if direction[0] == 1:
            max_distance = self.GRID_SIZE[0]
            distance = self.GRID_SIZE[0] - creature.x
        if direction[1] == -1:
            max_distance = self.GRID_SIZE[1]
            distance = creature.y + 1
        if direction[1] == 1:
            max_distance = self.GRID_SIZE[1]
            distance = self.GRID_SIZE[1] - creature.y
        return 1 - (distance-1)/max_distance

    def find_distance_to_food(self, direction):
        '''Find the distance to food in a specific direction from a creature.'''
        creature = self.creature
        pos = (creature.x, creature.y)
        if direction[0]:
            max_distance = self.GRID_SIZE[0]
        if direction[1]:
            max_distance = self.GRID_SIZE[1]

        distance = self.grid.distance_to_food(pos, direction)

        if distance is None:
            return 0
        return 1 - (distance-1)/max_distance


    def is_on_grid(self, pos):
        '''Return true if the position is on the grid.'''
        if any(i < 0 for i in pos):
            return False
        if any(pos[i] >= self.GRID_SIZE[i] for i in range(2)):
            return False
        return True

    def get_creature_at(self, pos):
        '''Return the creature at a position in the grid.

        Return None if there is no creature here.
        '''
        if (self.creature.x, self.creature.y) == pos:
            return self.creature
        return None

    @staticmethod
    def clamp(value, small, big):
        '''Return the value clamped between two other values.'''
        return min(max(small, value), big)

class Interface:
    '''The interface in which the user can see and communicate with the World.'''
    TILE_SIZE = 50
    TILE_BORDER = 3

    def __init__(self, screen_size):
        self.world = World()
        self.time_since_step = 0
        self.step_speed = 16
        self.fast_mode = False
        self.paused = False

        self.grid_rect = pygame.Rect(0, 0, *(self.world.GRID_SIZE[i] * self.TILE_SIZE for i in range(2)))

        self.background = self.make_background(screen_size)
        self.frequency_graph = pygame.surface.Surface((400, HEIGHT - self.grid_rect.bottom - 100))
        self.frequency_graph.fill(WHITE)
        self.scores_graph = pygame.surface.Surface((400, HEIGHT - self.grid_rect.bottom - 100))
        self.scores_graph.fill(WHITE)
        self.best_neural_net = pygame.surface.Surface(((280, self.grid_rect.bottom*0.75-40)))


    def make_background(self, screen_size):
        '''Draw the grid.'''
        background = pygame.Surface(screen_size)
        background.fill(BG_COLOR)
        for x in range(self.world.GRID_SIZE[0]):
            for y in range(self.world.GRID_SIZE[1]):
                pygame.draw.rect(background, WHITE, (x*self.TILE_SIZE+self.TILE_BORDER, y*self.TILE_SIZE+self.TILE_BORDER, self.TILE_SIZE-self.TILE_BORDER*2, self.TILE_SIZE-self.TILE_BORDER*2))
        return background

    def update_surfaces(self):
        '''Update all surfaces which are cached per-generation'''

        frequencies = np.bincount(tuple(net.fitness for net in self.world.genetic_algorithm.population), minlength=6)
        graphing.draw_bar_graph(self.frequency_graph, (WHITE, GRAY, DARK_GRAY), frequencies)
        graphing.draw_line_graph(self.scores_graph, (WHITE, GRAY, DARK_GRAY), self.world.highscores, self.world.meanscores)

        # Best neural net
        self.best_neural_net.fill(BG_COLOR)
        drawing.draw_neural_net(self.best_neural_net, self.world.genetic_algorithm.sorted_population[0])



    def screen_to_tile_pos(self, screen_pos):
        '''Convert a screeen position to a tile position.'''
        return tuple(int(num/self.TILE_SIZE) for num in screen_pos)

    def tick(self, clock):
        '''Do per-frame actions.'''
        if self.fast_mode:
            clock.tick()
            return

        self.time_since_step += clock.tick(60) * self.step_speed
        if self.paused:
            self.time_since_step = 0

    def process_keypress(self, keypress):
        '''Respond to a specific key press.'''
        if keypress == pygame.K_RIGHT:
            if not self.fast_mode:
                if self.step_speed == 60:
                    self.fast_mode = True
                elif self.step_speed == 32:
                    self.step_speed = 60
                else:
                    self.step_speed *= 2

        if keypress == pygame.K_LEFT:
            if self.fast_mode:
                self.fast_mode = False
            else:
                if self.step_speed == 60:
                    self.step_speed = 32
                else:
                    self.step_speed = max(1, int(self.step_speed * 0.5))
        if keypress == pygame.K_SPACE:
            self.paused = not self.paused

    def process_click(self, click, click_pos):
        '''Process a click at a position.

        EMPTY
        '''
        pass

    def process_world(self):
        '''Step the world if ready, and also go to the next generation if the world is finished.'''
        ready_to_step = False
        if self.paused:
            return

        if self.fast_mode:
            ready_to_step = True

        if self.time_since_step > 1000:
            self.time_since_step = self.time_since_step % 1000
            ready_to_step = True

        if ready_to_step:
            self.world.run_step()
            if self.world.steps == self.world.MAX_STEPS:
                if self.world.creature.index == self.world.POPULATION_SIZE - 1:
                    self.world.genetic_algorithm.calculate_stats()
                    highscore = self.world.genetic_algorithm.stats.max_fitness
                    self.world.highscores.append(highscore)
                    self.world.meanscores.append(self.world.genetic_algorithm.stats.mean_fitness)
                    self.update_surfaces()
                    self.world.next_generation()
                    pygame.display.set_caption('Gen '+str(self.world.genetic_algorithm.current_generation) +' - Creatures')
                else:
                    self.world.next_creature()


    def draw(self, clock):
        '''Draw the interface.'''
        if self.fast_mode:
            if not self.paused and self.world.steps != 0:
                return

        # Blitting grid
        SCREEN.blit(self.background, (0, 0))

        # Drawing objects on grid
        size = (self.TILE_SIZE*0.6, self.TILE_SIZE*0.6)
        for x, y in self.world.grid.food_pos():
            pygame.draw.rect(SCREEN, GREEN, (((x+0.2)*self.TILE_SIZE, (y+0.2)*self.TILE_SIZE), size))
        if self.world.creature is not None:
            creature = self.world.creature
            pygame.draw.rect(SCREEN, creature.color, (((creature.x+0.2)*self.TILE_SIZE, (creature.y+0.2)*self.TILE_SIZE), size))

        # Drawing simulation text
        display_text(SCREEN, "Generation: "+str(self.world.genetic_algorithm.current_generation), (0, self.grid_rect.bottom), size=40)
        display_text(SCREEN, "Creature "+str(self.world.creature.index+1)+"/"+str(self.world.POPULATION_SIZE), (0, self.grid_rect.bottom+40), size=30)

        settings = self.world.genetic_algorithm.settings
        display_text(SCREEN, "Food: "+str(self.world.food_amount), (0, self.grid_rect.bottom+80), size=20, color=GRAY)
        mutation_percentage = round(settings["mutation_amount"]*100/self.world.MAX_MUTATION_AMOUNT)
        display_text(SCREEN, "Mutation: "+str(mutation_percentage)+"%", (0, self.grid_rect.bottom+100), size=20, color=GRAY)

        display_text(SCREEN, "Perfect streak: "+str(self.world.perfect_streak), (0, self.grid_rect.bottom+130), size=20, color=DARK_GRAY)
        display_text(SCREEN, "Best perfect streak: "+str(self.world.best_perfect_streak), (0, self.grid_rect.bottom+150), size=20, color=DARK_GRAY)
        display_text(SCREEN, "Perfect generations: "+str(self.world.perfects), (0, self.grid_rect.bottom+180), size=20, color=DARK_GRAY)

        # Drawing step speed text
        display_text(SCREEN, "Step " + str(self.world.steps), (WIDTH-100, self.grid_rect.bottom), size=20, color=GRAY)
        if self.fast_mode:
            display_text(SCREEN, "Fast Mode", (WIDTH-100, self.grid_rect.bottom+24), size=15, color=RED, bold=True)
        else:
            display_text(SCREEN, "Speed "+str(self.step_speed), (WIDTH-100, self.grid_rect.bottom+24), size=15, color=GRAY, bold=True)
        if self.paused:
            display_text(SCREEN, "Paused", (WIDTH-100, self.grid_rect.bottom+40), size=15, color=DARK_BLUE, bold=True)

        # Drawing last generation stats
        SCREEN.blit(self.frequency_graph, (350, self.grid_rect.bottom+60))
        SCREEN.blit(self.scores_graph, (770, self.grid_rect.bottom+60))
        display_text(SCREEN, "Last generation scores", (350, self.grid_rect.bottom+20), size=28, color=BLACK)
        stats = self.world.genetic_algorithm.stats
        display_text(SCREEN, "Best score: "+str(stats.max_fitness), (550, self.grid_rect.bottom+60), size=20, color=DARK_GRAY)
        display_text(SCREEN, "Mean score: "+str(stats.mean_fitness), (550, self.grid_rect.bottom+80), size=20, color=DARK_GRAY)

        display_text(SCREEN, "Scores by generation", (770, self.grid_rect.bottom+20), size=28, color=BLACK)

        # Drawing neural net
        if self.world.genetic_algorithm.stats.generation > 0:
            display_text(SCREEN, "Best neural net of last generation:", (self.grid_rect.right+30, 10), size=14, color=DARK_GRAY)
            SCREEN.blit(self.best_neural_net, (self.grid_rect.right+30, 40))

        # Drawing fps
        display_text(SCREEN, clock.get_fps(), (0, HEIGHT - 15))


def display_text(surface, text, pos, size=14, color=BLACK, font_name="verdana", bold=False):
    '''Display text to a surface.'''
    text = str(text)
    font = pygame.font.SysFont(font_name, size, bold=bold, italic=0)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def get_events():
    '''Return a list of events since the last frame.'''
    events = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        events.append(event)
    return events

def get_keypress(events):
    '''Get the value of the keypress, or None if no key was pressed.'''
    for event in events:
        if event.type == pygame.KEYDOWN:
            return event.key
    return None

def get_clicked(events):
    '''Return True if there was a mouse click on this frame.'''
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            return event.button
    return None

def main():
    '''Run the program.'''

    clock = pygame.time.Clock()

    interface = Interface((WIDTH, HEIGHT))

    while True:
        interface.tick(clock)

        events = get_events()

        keypress = get_keypress(events)
        if keypress == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
        interface.process_keypress(keypress)


        click = get_clicked(events)
        if click:
            click_pos = pygame.mouse.get_pos()
            interface.process_click(click, click_pos)

        interface.process_world()

        interface.draw(clock)


        pygame.display.update()


if __name__ == "__main__":
    main()
