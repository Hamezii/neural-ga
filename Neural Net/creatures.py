"""
The module which implements the genetic algorithm as a population of creatures.
"""

import sys
import numpy as np
import pygame

import drawing
import graphing
import nn


pygame.init()
pygame.display.set_caption('Gen 1 - Creatures')

WIDTH = 1200
HEIGHT = 1000
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
    """A creature."""
    def __init__(self, index):
        self.x = None
        self.y = None
        self.color = (100, 100, 100)
        self.memory = 0
        self.index = index

class Grid:
    """A cache of object positions which can be accessed."""
    def __init__(self, size):
        self._food_mask = np.full(size, False)
        self._creature_mask = np.full(size, False)

    def clear(self):
        """Clear the cache."""
        self._food_mask.fill(False)
        self._creature_mask.fill(False)

    def food_pos(self):
        """Return the positions of all food."""
        for pos in np.transpose(np.nonzero(self._food_mask)):
            yield pos

    def move_creature(self, old_pos, new_pos):
        """Move a creature in the cache."""
        self._creature_mask[old_pos] = False
        self._creature_mask[new_pos] = True

    def add_creature(self, pos):
        """Add a creature to the cache."""
        self._creature_mask[pos] = True

    def add_food(self, pos):
        """Add a food to the cache."""
        self._food_mask[pos] = True

    def remove_food(self, pos):
        """Remove food from the cache."""
        self._food_mask[pos] = False

    def is_empty_tile(self, pos):
        """Return True if tile is empty."""
        if self._creature_mask[pos]:
            return False
        if self._food_mask[pos]:
            return False
        return True

    def distance_to_creature(self, pos, direction):
        """Return the distance to the nearest creature from pos in a direction."""
        return self._distance(self._creature_mask, pos, direction)

    def distance_to_food(self, pos, direction):
        """Return the distance to the nearest food from pos in a direction."""
        return self._distance(self._food_mask, pos, direction)

    @staticmethod
    def _distance(matrix, pos, direction):
        """Return the distance to the nearest object from pos in a direction."""
        if direction[0] == 1:
            array = matrix[pos[0]+1:, pos[1]]
            pos = pos[0]
            func = np.min
            right = True
        if direction[0] == -1:
            array = matrix[:pos[0], pos[1]]
            pos = pos[0]
            func = np.max
            right = False
        if direction[1] == 1:
            array = matrix[pos[0], pos[1]+1:]
            pos = pos[1]
            func = np.min
            right = True
        if direction[1] == -1:
            array = matrix[pos[0], :pos[1]]
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
    """Stores the creatures and the food."""

    DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    GRID_SIZE = (50, 35)
    POPULATION_SIZE = 50
    MIN_FOOD_AMOUNT = 1400
    MAX_FOOD_AMOUNT = 1400
    NET_SIZE = (13, 9, 5)
    MAX_STEPS = 150
    MAX_MUTATION_AMOUNT = 0.4

    def __init__(self):
        self.genetic_algorithm = nn.GeneticAlgorithm(self.POPULATION_SIZE, self.NET_SIZE)
        self.genetic_algorithm.set_settings(mutation_rate=0.2, mutation_amount=self.MAX_MUTATION_AMOUNT)

        self.creatures = ()
        self.grid = Grid(self.GRID_SIZE)
        self.steps = 0

        self.food_amount = self.MIN_FOOD_AMOUNT

        self.food_percentages = [0]
        self.perfects = 0
        self.perfect_streak = 0
        self.best_perfect_streak = 0

        self._populate_grid()

    def next_generation(self):
        """Simulate evolution and also reset positions of population and food."""
        self.genetic_algorithm.calculate_stats()
        if self.genetic_algorithm.stats.mean_fitness == self.food_amount / self.POPULATION_SIZE:
            # Perfect generation
            new_mutation_amount = self.genetic_algorithm.settings["mutation_amount"]*0.9
            self.genetic_algorithm.set_settings(mutation_rate=0.1, mutation_amount=new_mutation_amount)

            self.food_amount = min(self.food_amount + 10, self.MAX_FOOD_AMOUNT)

            self.perfect_streak += 1
            self.perfects += 1
            self.best_perfect_streak = max(self.best_perfect_streak, self.perfect_streak)
        else:
            # Not perfect generation
            new_mutation_amount = self.genetic_algorithm.settings["mutation_amount"]*5
            new_mutation_amount = self.clamp(new_mutation_amount, 0.01, self.MAX_MUTATION_AMOUNT)
            self.genetic_algorithm.set_settings(mutation_rate=0.2, mutation_amount=new_mutation_amount)

            self.food_amount = max(self.food_amount - 20, self.MIN_FOOD_AMOUNT)

            self.perfect_streak = 0

        self.genetic_algorithm.next_generation()
        self._populate_grid()
        self.steps = 0

    def _random_free_pos(self):
        """Return a random position on the grid which is empty."""
        pos = tuple(np.random.randint(axis) for axis in self.GRID_SIZE)
        while not self.grid.is_empty_tile(pos):
            pos = tuple(np.random.randint(axis) for axis in self.GRID_SIZE)
        return pos

    def _move_creature(self, creature, pos):
        """Move or place a creature to a position on the world."""
        if creature.x is None:
            self.grid.add_creature(pos)
        else:
            self.grid.move_creature((creature.x, creature.y), pos)
        creature.x, creature.y = pos

    def _populate_grid(self):
        """Fill the grid with food and move the creatures randomly."""
        self.grid.clear()
        self.creatures = tuple(Creature(i) for i in range(self.genetic_algorithm.population_size))
        for creature in self.creatures:
            self._move_creature(creature, self._random_free_pos())

        for _ in range(self.food_amount):
            self.grid.add_food(self._random_free_pos())

    def run_step(self):
        """Run a step of simulation."""
        for i in range(self.genetic_algorithm.population_size):
            self._make_creature_act(i)
        self.steps += 1

    def _make_creature_act(self, creature_id):
        """Simulate a step of a creature given its id."""

        creature = self.creatures[creature_id]
        net = self.genetic_algorithm.population[creature_id]

        inputs = []
        inputs.extend(self._find_distance_to_edge(creature_id, direction) for direction in self.DIRECTIONS)
        inputs.extend(self._find_distance_to_object("creature", creature_id, direction) for direction in self.DIRECTIONS)
        inputs.extend(self._find_distance_to_object("food", creature_id, direction) for direction in self.DIRECTIONS)
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
        if inputs[4+movement_index] == 1:
            return # Bumping into creature

        new_pos = (creature.x + movement_direction[0], creature.y + movement_direction[1])

        if inputs[8+movement_index] == 1:
            net.fitness += 1 # Getting food
            self.grid.remove_food(new_pos)

        self._move_creature(creature, new_pos)

    def _find_distance_to_edge(self, creature_id, direction):
        """Find the distance to the edge of the world in a specific direction from a creature."""
        creature = self.creatures[creature_id]
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

    def _find_distance_to_object(self, object_type, creature_id, direction):
        """Find the distance to an object in a specific direction from a creature."""
        creature = self.creatures[creature_id]
        pos = (creature.x, creature.y)
        if direction[0]:
            max_distance = self.GRID_SIZE[0]
        if direction[1]:
            max_distance = self.GRID_SIZE[1]
        if object_type == "creature":
            distance = self.grid.distance_to_creature(pos, direction)
        if object_type == "food":
            distance = self.grid.distance_to_food(pos, direction)
        if distance is None:
            return 0
        return 1 - (distance-1)/max_distance


    def _is_on_grid(self, pos):
        """Return true if the position is on the grid."""
        if any(i < 0 for i in pos):
            return False
        if any(pos[i] >= self.GRID_SIZE[i] for i in range(2)):
            return False
        return True

    def get_creature_at(self, pos):
        """Return the creature at a position in the grid.

        Return None if there is no creature here.
        """
        if not self._is_on_grid(pos):
            return None
        for creature in self.creatures:
            if (creature.x, creature.y) == pos:
                return creature
        return None

    @staticmethod
    def clamp(value, small, big):
        """Return the value clamped between two other values."""
        return min(max(small, value), big)

class Interface:
    """The interface in which the user can see and communicate with the World."""
    TILE_SIZE = 16

    def __init__(self, screen_size):
        self._world = World()
        self._time_since_step = 0
        self._step_speed = 16
        self._fast_mode = False
        self._paused = False

        self._grid_rect = pygame.Rect(0, 0, *(self._world.GRID_SIZE[i] * self.TILE_SIZE for i in range(2)))

        self._background = self._make_background(screen_size)
        self._frequency_graph = pygame.surface.Surface((400, HEIGHT - self._grid_rect.bottom - 100))
        self._frequency_graph.fill(WHITE)
        self._food_percentage_graph = pygame.surface.Surface((400, HEIGHT - self._grid_rect.bottom - 100))
        self._food_percentage_graph.fill(WHITE)
        self._best_neural_net = pygame.surface.Surface(((280, self._grid_rect.bottom*0.5-40)))


    def _make_background(self, screen_size):
        """Draw the grid."""
        background = pygame.Surface(screen_size)
        background.fill(BG_COLOR)
        for x in range(self._world.GRID_SIZE[0]):
            for y in range(self._world.GRID_SIZE[1]):
                pygame.draw.rect(background, WHITE, (x*self.TILE_SIZE+1, y*self.TILE_SIZE+1, self.TILE_SIZE-2, self.TILE_SIZE-2))
        return background

    def _update_surfaces(self):
        """Update all surfaces which are cached per-generation"""

        frequencies = np.bincount(tuple(net.fitness for net in self._world.genetic_algorithm.population), minlength=6)
        graphing.draw_bar_graph(self._frequency_graph, (WHITE, GRAY, DARK_GRAY), frequencies)
        graphing.draw_line_graph(self._food_percentage_graph, (WHITE, GRAY, DARK_GRAY), self._world.food_percentages)

        # Best neural net
        self._best_neural_net.fill(BG_COLOR)
        drawing.draw_neural_net(self._best_neural_net, self._world.genetic_algorithm.sorted_population[0])

    def _screen_to_tile_pos(self, screen_pos):
        """Convert a screeen position to a tile position."""
        return tuple(int(num/self.TILE_SIZE) for num in screen_pos)

    def tick(self, clock):
        """Do per-frame actions."""
        if self._fast_mode:
            clock.tick()
            return

        self._time_since_step += clock.tick(60) * self._step_speed
        if self._paused:
            self._time_since_step = 0

    def process_keypress(self, keypress):
        """Respond to a specific key press."""
        if keypress == pygame.K_RIGHT:
            if not self._fast_mode:
                if self._step_speed == 60:
                    self._fast_mode = True
                elif self._step_speed == 32:
                    self._step_speed = 60
                else:
                    self._step_speed *= 2

        if keypress == pygame.K_LEFT:
            if self._fast_mode:
                self._fast_mode = False
            else:
                if self._step_speed == 60:
                    self._step_speed = 32
                else:
                    self._step_speed = max(1, int(self._step_speed * 0.5))
        if keypress == pygame.K_SPACE:
            self._paused = not self._paused

    def process_click(self, click, click_pos):
        """Process a click at a position."""
        clicked_tile = self._screen_to_tile_pos(click_pos)
        obj = self._world.get_creature_at(clicked_tile)
        if obj is not None:
            if click == 1:
                self._world.genetic_algorithm.population[obj.index].fitness = 0
            if click == 3:
                self._world.genetic_algorithm.population[obj.index].fitness += 1

    def process_world(self):
        """Step the world if ready, and also go to the next generation if the world is finished."""
        ready_to_step = False
        if self._paused:
            return

        if self._fast_mode:
            ready_to_step = True

        if self._time_since_step > 1000:
            self._time_since_step = self._time_since_step % 1000
            ready_to_step = True

        if ready_to_step:
            self._world.run_step()
            if self._world.steps == self._world.MAX_STEPS:
                self._world.genetic_algorithm.calculate_stats()
                food_percentage = min(1, (self._world.genetic_algorithm.stats.mean_fitness*self._world.POPULATION_SIZE) / self._world.food_amount)
                self._world.food_percentages.append(food_percentage)
                self._update_surfaces()
                self._world.next_generation()
                pygame.display.set_caption('Gen '+str(self._world.genetic_algorithm.current_generation) +' - Creatures')


    def draw(self, clock):
        """Draw the interface."""
        if self._fast_mode:
            if not self._paused and self._world.steps != 0:
                return

        # Blitting grid
        SCREEN.blit(self._background, (0, 0))

        # Drawing objects on grid
        size = (self.TILE_SIZE*0.6, self.TILE_SIZE*0.6)
        for x, y in self._world.grid.food_pos():
            pygame.draw.rect(SCREEN, GREEN, (((x+0.2)*self.TILE_SIZE, (y+0.2)*self.TILE_SIZE), size))
        for creature in self._world.creatures:
            pygame.draw.rect(SCREEN, creature.color, (((creature.x+0.2)*self.TILE_SIZE, (creature.y+0.2)*self.TILE_SIZE), size))

        # Drawing simulation text
        display_text(SCREEN, "Generation: "+str(self._world.genetic_algorithm.current_generation), (0, self._grid_rect.bottom), size=40)

        settings = self._world.genetic_algorithm.settings
        display_text(SCREEN, "Food: "+str(self._world.food_amount), (0, self._grid_rect.bottom+50), size=20, color=GRAY)
        mutation_percentage = round(settings["mutation_amount"]*100/self._world.MAX_MUTATION_AMOUNT)
        display_text(SCREEN, "Mutation: "+str(mutation_percentage)+"%", (0, self._grid_rect.bottom+70), size=20, color=GRAY)

        display_text(SCREEN, "Perfect streak: "+str(self._world.perfect_streak), (0, self._grid_rect.bottom+100), size=20, color=DARK_GRAY)
        display_text(SCREEN, "Best perfect streak: "+str(self._world.best_perfect_streak), (0, self._grid_rect.bottom+120), size=20, color=DARK_GRAY)
        display_text(SCREEN, "Perfect generations: "+str(self._world.perfects), (0, self._grid_rect.bottom+150), size=20, color=DARK_GRAY)

        # Drawing step speed text
        display_text(SCREEN, "Step " + str(self._world.steps), (WIDTH-100, self._grid_rect.bottom), size=20, color=GRAY)
        if self._fast_mode:
            display_text(SCREEN, "Fast Mode", (WIDTH-100, self._grid_rect.bottom+24), size=15, color=RED, bold=True)
        else:
            display_text(SCREEN, "Speed "+str(self._step_speed), (WIDTH-100, self._grid_rect.bottom+24), size=15, color=GRAY, bold=True)
        if self._paused:
            display_text(SCREEN, "Paused", (WIDTH-100, self._grid_rect.bottom+40), size=15, color=DARK_BLUE, bold=True)

        # Drawing last generation stats
        SCREEN.blit(self._frequency_graph, (350, self._grid_rect.bottom+60))
        SCREEN.blit(self._food_percentage_graph, (770, self._grid_rect.bottom+60))
        display_text(SCREEN, "Last generation scores", (350, self._grid_rect.bottom+20), size=28, color=BLACK)
        stats = self._world.genetic_algorithm.stats
        display_text(SCREEN, "Best score: "+str(stats.max_fitness), (550, self._grid_rect.bottom+60), size=20, color=DARK_GRAY)
        display_text(SCREEN, "Mean score: "+str(stats.mean_fitness), (550, self._grid_rect.bottom+80), size=20, color=DARK_GRAY)

        display_text(SCREEN, "Food eaten", (770, self._grid_rect.bottom+20), size=28, color=BLACK)
        food_percentage = round(self._world.food_percentages[-1]*100 , 1)
        display_text(SCREEN, "Percentage: "+str(food_percentage)+"%", (950, self._grid_rect.bottom+60), size=20, color=DARK_GRAY)

        # Drawing neural net
        if self._world.genetic_algorithm.stats.generation > 0:
            display_text(SCREEN, "Best neural net of last generation:", (self._grid_rect.right+30, 10), size=14, color=DARK_GRAY)
            SCREEN.blit(self._best_neural_net, (self._grid_rect.right+30, 40))

        # Drawing on tile her
        hovered_tile = self._screen_to_tile_pos(pygame.mouse.get_pos())
        hovered_object = self._world.get_creature_at(hovered_tile)
        if hovered_object is not None:
            fitness = self._world.genetic_algorithm.population[hovered_object.index].fitness
            display_text(SCREEN, fitness, (hovered_tile[0]*self.TILE_SIZE, hovered_tile[1]*self.TILE_SIZE), size=14, color=BLACK, bold=True)

        # Drawing fps
        display_text(SCREEN, clock.get_fps(), (0, HEIGHT - 15))



def display_text(surface, text, pos, size=14, color=BLACK, font_name="verdana", bold=False):
    """Display text to a surface."""
    text = str(text)
    font = pygame.font.SysFont(font_name, size, bold=bold, italic=0)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def get_events():
    """Return a list of events since the last frame."""
    events = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        events.append(event)
    return events

def get_keypress(events):
    """Get the value of the keypress, or None if no key was pressed."""
    for event in events:
        if event.type == pygame.KEYDOWN:
            return event.key
    return None

def get_clicked(events):
    """Return True if there was a mouse click on this frame."""
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            return event.button
    return None

def main():
    """Run the program."""

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
