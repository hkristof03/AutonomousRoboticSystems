#%matplotlib inline     for jupyter notebook
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import *
import cv2
import os
import re

global x_begin, y_begin, x_end, y_end, z_begin, z_end

x_begin = y_begin = z_begin = -10
x_end = y_end = z_end = 10

def function_rastrigin(position):

    x = position[0]
    y = position[1]

    return (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20


def function_rosenbrock(position):

    x = position[0]
    y = position[1]
    #a=0, b=10
    return (-x)**2 + 10*(y-x**2)**2


class Particle(object):

    def __init__(self, x, v):
        """Particle class' constructor
        Keyword arguments:
        x -- double[d] coordinates of particles in d dimensions
        v -- double[d] velocity vector of particles in d dimensions
        """
        self.position = x
        self.velocity = v
        self.best_value = 0
        self.best_position = self.position
        self.value = 0

    def get_value(self, func):
        """
        Method to get each particle's value with function fun, at given position.
        Keyword arguments:
        fun -- Benchmark function, use function_rastrigin or function_rosenbrock
        """
        value = func(self.position)
        if value < self.best_value:   # minimisation option
            self.best_value = value
            self.best_position = self.position
        #check if value is in the space limits
        if value > z_end:
            self.value = z_end
        if value < z_begin:
            self.value = z_begin
        else:
            self.value = value


    def update_position(self, dt=1):
        """This method updates the particle's positon according to the given POS equation.
        Keyword arguments:
        dt -- time step, default=1
        """
        #Lets suppose that space is cubic for now...
        #If position is not in the given space, set it to the minimum or maximum
        for dim in range(len(self.position)):
            if (self.position[dim] + self.velocity[dim] * dt) > x_end:
                self.position[dim] = x_end

            if (self.position[dim] + self.velocity[dim] * dt) < x_begin:
                self.position[dim] = x_begin

            else:
                self.position[dim] += self.velocity[dim] * dt


    def update_velocity(self, best_position, coefficients):
        """This method updates the particle's velocity according to the given POS equation.
        Keyword arguments:
        best_position - Best position of the population, d-dimensions
        coefficients - double[3] vector of three coefficients of simulation: 0-inertial 1-egoism 2-group terms
        """
        for i in range(len(best_position)):
            self.velocity[i] = self.velocity[i] * coefficients[0] + \
                coefficients[1] * np.random.random() * (self.best_position[i] - self.position[i]) + \
                coefficients[2] * np.random.random() * (best_position[i] - self.position[i])



def atoi(text):

    return int(text) if text.isdigit() else text


def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
