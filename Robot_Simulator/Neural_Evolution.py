import pygame, pygame.gfxdraw, math
from shapely.geometry import LineString
from shapely.geometry import Point as pnt

import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as sigmoid_activation_function
import random
from operator import attrgetter

import pandas as pd

import time

#from .Definitions import *
#from Definitions import *

global win

maxX = 800
maxY = 600

WHITE = (255, 255, 255)
WHITE_2 = (245, 245, 245)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 238, 0)
BROWN = (255, 222, 173)


class GeneticAlgorithm(object):

    def __init__(self):
        pass

    #Population is created from NNs in the Robot

    def selRandom(self, individuals, k):
        """Select *k* individuals at random from the input *individuals* with
        replacement. The list returned contains references to the input
        *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.
        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        return [random.choice(individuals) for i in range(k)]

    def selBest(self, individuals, k, fit_attr="fitnessScore"):
        """Select the *k* best individuals among the input *individuals*. The
        list returned contains references to the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list containing the k best individuals.
        """
        return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]

    def selRoulette(self, individuals, k, fit_attr="fitnessScore"):
        """Select *k* individuals from the input *individuals* using *k*
        spins of a roulette. The selection is made by looking only at the first
        objective of each individual. The list returned contains references to
        the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        .. warning::
           The roulette selection by definition cannot be used for minimization
           or when the fitness can be smaller or equal to 0.
        """

        s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
        sum_fits = sum(getattr(ind, fit_attr) for ind in individuals)
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += getattr(ind, fit_attr)
                if sum_ > u:
                    chosen.append(ind)
                    break

        return chosen

    def selTournament(self, individuals, k, tournsize, fit_attr="fitnessScore"):
        """Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants = self.selRandom(individuals, tournsize)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    def one_point_crossover(self, parent_1, parent_2):

        offspring_1, offspring_2 = parent_1, parent_2
        size = min(len(parent_1), len(parent_2))
        crossover_point = random.randint(1, size - 1)
        offspring_1[crossover_point:], offspring_2[crossover_point:] = parent_2[crossover_point:], parent_1[crossover_point:]

        return offspring_1, offspring_2

    def two_point_crossover(self, parent_1, parent_2):

        offspring_1, offspring_2 = parent_1, parent_2
        size = min(len(parent_1), len(parent_2))
        crossover_point_1 = random.randint(1, size)
        crossover_point_2 = random.randint(1, size - 1)

        if crossover_point_2 >= crossover_point_1:
            crossover_point_2 += 1
        else:
            crossover_point_1, crossover_point_2 = crossover_point_2, crossover_point_1

        offspring_1[crossover_point_1:crossover_point_2] = parent_2[crossover_point_1:crossover_point_2]
        offspring_2[crossover_point_1:crossover_point_2] = parent_1[crossover_point_1:crossover_point_2]

        return offspring_1, offspring_2

    def mutShuffleIndexes(self, individual, indpb):
        """Shuffle the attributes of the input individual and return the mutant.
        The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
        probability of each attribute to be moved. Usually this mutation is applied on
        vector of indices.
        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be exchanged to
                      another position.
        :returns: A tuple of one individual.
        This function uses the :func:`~random.random` and :func:`~random.randint`
        functions from the python base :mod:`random` module.
        """
        size = len(individual)
        for i in range(size):
            if random.random() < indpb:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                individual[i], individual[swap_indx] = \
                    individual[swap_indx], individual[i]

        return individual

    def GAPipeLine(self, individuals, proportion):
        #parents and children number
        pk = int(len(individuals) * proportion)
        ck = int(len(individuals) - pk)
        chosen_1 = self.selRoulette(individuals, pk)
        chosen_2 = self.selRoulette(chosen_1, ck)
        offsprings = []

        print("len individuals:", len(individuals))

        i_ = 0
        step = 2
        while i_ < ck:
            print("i_ before parents:", i_)
            parent_1 = chosen_2[i_].NN.create_chromosome()
            parent_2 = chosen_2[i_ + 1].NN.create_chromosome()
            offs_1, offs_2 = self.one_point_crossover(parent_1, parent_2)
            offsprings.append(offs_1)
            offsprings.append(offs_2)
            i_ = i_ + step
            print("i_ at end:", i_)

        mutated_offsprings = []
        for i_ in range(len(offsprings)):
            mo = self.mutShuffleIndexes(offsprings[i_], 0.1)
            mutated_offsprings.append(mo)

        for i_ in range(len(chosen_2)):
            chosen_2[i_].NN.update_weights(mutated_offsprings[i_])

        new_population = chosen_1 + chosen_2
        print("len new population:", len(new_population))
        return new_population


class NeuralNetwork(object):

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.fitness = 0
        self.bias = bias
        self.weights_in_hidden = np.zeros(shape=(self.no_of_hidden_nodes, self.no_of_in_nodes))
        self.weights_hidden_out = np.zeros(shape=(self.no_of_out_nodes, self.no_of_hidden_nodes))
        self.create_weight_matrices()

    def create_weight_matrices(self):
        '''
        A method to initialize the weight matrices of the neural network with optional bias nodes.
        good idea to initialize weight matrices with values in the interval
        (-1/sqrt(n), 1/sqrt(n)), where n denotes the number of input nodes for each
        weight matrice.
        '''

        bias_node = 1 if self.bias else 0
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes + bias_node))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes + bias_node))

    def truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def create_chromosome(self):
        wih = self.weights_in_hidden.flatten()
        woh = self.weights_hidden_out.flatten()
        chromosome = np.concatenate((wih, woh), axis=None)
        return chromosome

    def update_weights(self, chromosome):
        self.weights_in_hidden = chromosome[:self.no_of_in_nodes * self.no_of_hidden_nodes].reshape(
            self.no_of_hidden_nodes,
            self.no_of_in_nodes)
        self.weights_hidden_out = chromosome[self.no_of_in_nodes * self.no_of_hidden_nodes:].reshape(
            self.no_of_out_nodes,
            self.no_of_hidden_nodes)

    def run(self, input_vector):
        '''
        Running the network with an input vector input_vector, which can be a tuple,
        list or ndarray.
        '''
        #turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = np.tanh(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = np.tanh(output_vector) * 6

        return [item for sublist in output_vector for item in sublist]


def CorrY(y):
    return (maxY - y)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, object):
        return math.sqrt((object.x - self.x) ** 2 + (object.y - self.y) ** 2)


class dust(object):
    """
    Creates the dust
    """

    def __init__(self, point, radius):
        self.x = point.x
        self.y = point.y
        self.rad = radius

    def draw(self):
        pygame.gfxdraw.filled_circle(win, self.x, CorrY(self.y), self.rad, BROWN)


class Wall(object):
    def __init__(self, startpoint, endpoint):
        self.x1 = startpoint.x
        self.y1 = startpoint.y
        self.x2 = endpoint.x
        self.y2 = endpoint.y
        if self.x1 == self.x2:
            self.angleDeg = 90
            self.angleRad = math.radians(self.angleDeg)
        else:
            self.angleRad = math.atan((self.y2 - self.y1) / (self.x2 - self.x1))
            self.angleDeg = math.degrees(self.angleRad)
        self.bound = LineString([(self.x1, self.y1), (self.x2, self.y2)])

    def draw(self):
        pygame.draw.line(win, (0, 0, 0), (self.x1, CorrY(self.y1)), (self.x2, CorrY(self.y2)), 5)


class Sensor(object):
    def __init__(self, startpoint, endpoint, width):
        self.x1 = startpoint.x
        self.y1 = startpoint.y
        self.x2 = endpoint.x
        self.y2 = endpoint.y
        self.width = width
        self.distance = 0

    def set_sensor_direction(self, newpoint, deg, radius):
        self.x1 = newpoint.x
        self.y1 = newpoint.y
        theta = math.radians(deg)
        x2_rot = ((self.x2 - self.x1) * math.cos(theta) + (self.y2 - self.y1) * math.sin(theta)) + self.x1
        y2_rot = (-(self.x2 - self.x1) * math.sin(theta) + (self.y2 - self.y1) * math.cos(theta)) + self.y1
        self.x2 = x2_rot
        self.y2 = y2_rot

        d = math.sqrt((self.x2 - self.x1) ** 2 + (self.y1 - self.y2) ** 2)
        t = radius / d
        x1 = (1 - t) * self.x1 + t * self.x2
        y1 = (1 - t) * self.y1 + t * self.y2
        self.x1 = x1
        self.y1 = y1

    def update_sensor_position(self, biaspoint):
        self.x1 += biaspoint.x
        self.y1 += biaspoint.y
        self.x2 += biaspoint.x
        self.y2 += biaspoint.y

    def update_rotate_sensor_line(self, centerx, centery, deg):
        cx = centerx
        cy = centery
        theta = math.radians(deg)
        x1_rot = centerx + (self.x1 - centerx) * math.cos(theta) - (self.y1 - centery) * math.sin(theta)
        y1_rot = centery + (self.x1 - centerx) * math.sin(theta) + (self.y1 - centery) * math.cos(theta)
        x2_rot = centerx + (self.x2 - centerx) * math.cos(theta) - (self.y2 - centery) * math.sin(theta)
        y2_rot = centery + (self.x2 - centerx) * math.sin(theta) + (self.y2 - centery) * math.cos(theta)
        self.x1 = x1_rot
        self.y1 = y1_rot
        self.x2 = x2_rot
        self.y2 = y2_rot

    def calculate_distance(self, object):
        self.distance = int(Point(self.x1, self.y1).distance(Point(object.x, object.y)))


class Robot(object):
    def __init__(self, point, radius, circle_width, num_sensors, sensor_range, velocity, sensor_width):
        self.x = point.x
        self.y = point.y
        self.radius = radius
        self.circle_width = circle_width
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range
        self.sensor_width = sensor_width
        self.velocity = velocity
        self.sensors = []
        self.prev_x = self.x
        self.prev_y = self.y
        # Nik's attributes
        self.prev_theta = 0
        self.width = self.radius * 2
        self.velR = 0
        self.velL = 0
        self.velMax = 8
        self.velMin = -8
        self.acc = 0.1
        self.iccX = 0
        self.iccY = 0
        self.omega = 0
        self.R = 0
        self.theta = 0
        self.dt = 1
        self.font = pygame.font.SysFont('comicsans', 16, False, False)
        self.center = pnt(self.x, self.y)
        self.circle = self.center.buffer(self.radius).boundary
        self.angleDeg = 0
        self.collision = False
        self.collisionScore = 0
        self.velocityScore = 0
        self.fitnessScore = 0
        #Neural Network
        self.NN = NeuralNetwork(12, 2, 24)


    def create_adjust_sensors(self):
        self.sensor_range += self.radius
        for i in range(self.num_sensors):
            s = Sensor(Point(self.x, self.y), Point(self.x, self.y - self.sensor_range), self.sensor_width)
            angle = 360 / self.num_sensors * i
            s.set_sensor_direction(Point(self.x, self.y), angle, self.radius)
            self.sensors.append(s)

    def update_sensors(self, biaspoint, d_theta):
        for sen in self.sensors:
            sen.update_sensor_position(biaspoint)
            # last parameter the rotation degree
            sen.update_rotate_sensor_line(self.x, self.y, d_theta)

    def calculate_fitness(self):
        """
               Calculates the fitness of the robot.
               Each timestep it increases the score by the absolure value of the velocity, times a constant beta,
               and reduces the score if the object is colliding by alpha
               """
        alpha = -50  # constant to adjust weight of collisions
        beta = 1  # constant to adjust weight of velocity
        col = 0  # col is used to completely discount the velocity contribution if the object is colliding
        if self.collision:
            self.collisionScore += 1  # total dt that the robot has been colliding
            col = 1
        self.velocityScore += (abs(self.velocity)) * (1 - col)  # total positive score from the velocity
        self.fitnessScore = alpha * self.collisionScore + beta * self.velocityScore
        #print(self.fitnessScore)

    def update_velocity(self):
        distances = []
        for sen in self.sensors:
            distances.append(sen.distance)
        velocities = self.NN.run(distances)
        self.velL, self.velR = velocities[0], velocities[1]

    def calculate_intersection(self, walls):

        for sen in self.sensors:
            s2 = LineString([(sen.x1, sen.y1), (sen.x2, sen.y2)])

            for wall_ in walls:
                # FIX_AFTER
                # s1 = LineString([(wall_.x1, wall_.y1), (wall_.x2, wall_.y2)])
                ip = s2.intersection(wall_.bound)
                if ip:
                    sen.calculate_distance(ip)
                    break
                    # print("distance:", sen.distance)
                else:
                    sen.distance = int(self.sensor_range / 2)

    def imminent_collision(self, wall):

        self.center = pnt(self.x, self.y)
        self.circle = self.center.buffer(self.radius + 2).boundary
        collision = self.circle.intersection(wall.bound)
        return collision

    def key_input(self):

        if keys[pygame.K_w]:
            self.velL += self.acc
        if keys[pygame.K_s]:
            self.velL -= self.acc
        if keys[pygame.K_o]:
            self.velR += self.acc
        if keys[pygame.K_l]:
            self.velR -= self.acc
        if keys[pygame.K_x]:
            self.velL = 0
            self.velR = 0
        if keys[pygame.K_t]:
            self.velL += self.acc
            self.velR += self.acc
        if keys[pygame.K_g]:
            self.velL -= self.acc
            self.velR -= self.acc
        # Constrain velocity    -NEEDS WORK-constrains wrongly for top-also limit negative speeds
        if self.velL >= self.velMax:
            self.velL = self.velMax

        if self.velR >= self.velMax:
            self.velR = self.velMax

        if self.velL <= self.velMin:
            self.velL = self.velMin

        if self.velR <= self.velMin:
            self.velR = self.velMin

    def move(self):

        self.velocity = (self.velL + self.velR) / 2
        # Rotation
        if self.velR == self.velL:
            self.omega = 0
            self.R = 0
        else:
            self.omega = (self.velR - self.velL) / self.width
            self.R = (self.width / 2) * (self.velR + self.velL) / (self.velR - self.velL)
            iccX = (self.x - self.R * math.sin(self.theta))
            iccY = (self.y + self.R * math.cos(self.theta))
            self.theta = self.theta + self.omega * self.dt

        self.angleDeg = math.degrees(self.theta) % 360

        # Movement-Update position
        self.x += self.velocity * math.cos(self.theta) * self.dt
        self.y += self.velocity * math.sin(self.theta) * self.dt

        # COLLISION

        # doubleCollision = False
        # singleCollision = False
        otherWalls = walls.copy()
        for wall_ in walls:
            otherWalls.remove(wall_)
            col = robot.imminent_collision(wall_)
            velSign = self.velocity / (abs(self.velocity) + 0.000001)

            self.collision = False

            if col:
                self.x -= self.velocity * math.cos(self.theta) * self.dt
                self.y -= self.velocity * math.sin(self.theta) * self.dt

                self.x += 0.5 * self.velocity * math.cos(self.theta) * self.dt
                self.y += 0.5 * self.velocity * math.sin(self.theta) * self.dt

                col1 = robot.imminent_collision(wall_)  # NEED to just check for previous wall.

                if col1:

                    # singleCollision = True

                    self.x -= 0.5 * self.velocity * math.cos(self.theta) * self.dt
                    self.y -= 0.5 * self.velocity * math.sin(self.theta) * self.dt

                    self.x += self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (
                        math.cos(math.radians(wall_.angleDeg)))
                    self.y += self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (
                        math.sin(math.radians(wall_.angleDeg)))

                    self.collision = True

                    for otherWall in otherWalls:

                        col2 = robot.imminent_collision(otherWall)
                        if col2:
                            self.x -= self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (
                                math.cos(math.radians(wall_.angleDeg)))
                            self.y -= self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (
                                math.sin(math.radians(wall_.angleDeg)))
                            doubleCollision = True

        # END COLLISION

    def draw(self):

        # TEXT ALONG CIRCLE
        radToDeg = 57.2958
        currAngle = math.floor((360 - (radToDeg * self.theta)) % 360)
        textDistance = 20 + self.width / 2
        textPos = (self.x + (math.cos(self.theta) - 0.15) * textDistance,
                   CorrY(self.y + (math.sin(self.theta)) * textDistance - math.sin(self.theta) * 5 - 5))
        text = self.font.render(str(math.floor(self.angleDeg)), 1, BLUE)
        win.blit(text, textPos)

        # END TEXT ALONG CIRCLE
        outerRad = math.floor(self.width / 2)
        pos = (math.floor(self.x), CorrY(math.floor(self.y)))
        lineEnd = (math.floor(self.x) + math.floor(outerRad * math.cos(self.theta)),
                   CorrY(math.floor(self.y) + math.floor(outerRad * math.sin(self.theta))))

        polygonPoints = (pos, lineEnd, pos)
        pygame.gfxdraw.filled_circle(win, pos[0], pos[1], outerRad, YELLOW)
        pygame.gfxdraw.aacircle(win, pos[0], pos[1], outerRad, BLACK)
        pygame.gfxdraw.aapolygon(win, polygonPoints, BLACK)

        for sen in self.sensors:
            pygame.draw.line(win, (255, 0, 0), (sen.x1, CorrY(sen.y1)), (sen.x2, CorrY(sen.y2)), sen.width)
            text_ = self.font.render(str(sen.distance), 1, BLUE)
            win.blit(text_, (sen.x1, CorrY(sen.y1)))
        # WHEELS
        textPosL = (self.x + (math.cos(self.theta + math.radians(90)) - 0.15) * textDistance,
                    CorrY(self.y + (math.sin(self.theta + math.radians(90))) * textDistance - math.sin(
                        self.theta) * 5 - 5))
        textPosR = (self.x + (math.cos(self.theta + math.radians(-90)) - 0.15) * textDistance,
                    CorrY(self.y + (math.sin(self.theta + math.radians(-90))) * textDistance - math.sin(
                        self.theta) * 5 - 5))

        textL = self.font.render(format(self.velL, '.2f'), 1, RED)
        textR = self.font.render(format(self.velR, '.2f'), 1, RED)
        win.blit(textL, textPosL)
        win.blit(textR, textPosR)


def redrawGameWindow():
    win.fill((WHITE_2))
    for wall in walls:
        wall.draw()
    robot.draw()
    dust1.draw()
    pygame.display.update()


def run_GA(individuals, proportion, df, GA):

    lst_dict = []
    col = 'Best_fitness_score'
    val = max(getattr(ind, 'fitnessScore') for ind in individuals)
    df_ = pd.DataFrame([{col: val}])
    df = df.append(df_, ignore_index=True)
    if len(df) % 100 == 0:
        ax = df.plot()
        fig = ax.get_figure()
        fig.savefig('Fitness_score_evolution.jpg')

    new_population = GA.GAPipeLine(individuals, proportion)

    return new_population


window_width = 800
window_height = 600

pygame.init()

win = pygame.display.set_mode((window_width, window_height))

pygame.display.set_caption("BumbleBeeN'TheHood")

start_point = Point(200, 200)

number_of_individuals = 20
robots = []
for i in range(number_of_individuals):
    robots.append(Robot(start_point, 40, 1, 12, 40, 10, 1))
    robots[i].create_adjust_sensors()

#Genetic Algorithm class instance
GA = GeneticAlgorithm()
#DataFrame that stores Historical data for plotting
df = pd.DataFrame(columns=['Best_fitness_score'])
#parents and children proportion
proportion = 0.6

dust1 = dust(Point(10, 10), 15)

wall_right = Wall(Point(750, 50), Point(750, 550))
wall_left = Wall(Point(50, 50), Point(50, 550))
wall_top = Wall(Point(50, 50), Point(750, 50))
wall_bottom = Wall(Point(50, 550), Point(750, 550))
wall1 = Wall(Point(300, 0), Point(700, 692.8204))

walls = [wall_right, wall_left, wall_top, wall_bottom, wall1]

generation = 0
run = True
while run:

    pygame.time.delay(30)  # milliseconds delay

    for robot in robots:

        dt = 0
        while dt < 100:

            bias_x = 0
            bias_y = 0
            d_theta = 0

            robot.update_velocity()
            robot.move()
            robot.calculate_fitness()

            bias_x = robot.x - robot.prev_x
            bias_y = robot.y - robot.prev_y
            d_theta = robot.angleDeg - robot.prev_theta

            robot.prev_theta = robot.angleDeg
            robot.prev_x = robot.x
            robot.prev_y = robot.y
            biasPoint = Point(bias_x, bias_y)

            robot.update_sensors(biasPoint, d_theta)
            robot.calculate_intersection(walls)
            redrawGameWindow()

            dt += 1

    robots = run_GA(robots, proportion, df, GA)

    for robot in robots:
        robot.fitnessScore = 0

    generation += 1
    print("generation:", generation)
    #redrawGameWindow()

    #time.sleep(0.5)

pygame.quit()