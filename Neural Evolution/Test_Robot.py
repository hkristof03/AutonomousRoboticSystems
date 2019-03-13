import Definitions as defs_
from Definitions import pygame


window_width = 800
window_height = 600

maxX = window_width
maxY = window_height

defs_.pygame.init()
clock = pygame.time.Clock()
win = pygame.display.set_mode((window_width, window_height))
defs_.pygame.display.set_caption("BumbleBeeN'TheHood")
FPS = 30

file_json = 'Generations_data.json'

NN = defs_.RecurrentNeuralNetwork(12, 2, 12)
start_point = defs_.Point(100, 100)
robot = defs_.Robot(start_point, 30, 1, 12, 80, 10, 1, NN)
robot.create_adjust_sensors()

#Reading the data from json, 90th generation Best fitness
chromosome = defs_.read_weights_from_json(file_json, 90, 1)
robot.NN.update_weights(chromosome)


layout = 'box'

walls = defs_.Layout(layout)
dust = defs_.Dust(150, maxX, maxY, 8, win)

run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    bias_x = 0
    bias_y = 0
    d_theta = 0

    robot.update_velocity()
    robot.move(walls)
    robot.eat_dust(dust)

    bias_x = robot.x - robot.prev_x
    bias_y = robot.y - robot.prev_y
    d_theta = robot.angleDeg - robot.prev_theta

    robot.prev_theta = robot.angleDeg
    robot.prev_x = robot.x
    robot.prev_y = robot.y
    biasPoint = defs_.Point(bias_x, bias_y)

    robot.update_sensors(biasPoint, d_theta)
    robot.calculate_intersection(walls)

    robot.fitness_function()
    terminate = robot.terminate_check()
    defs_.redrawGameWindow(win, robot, walls, dust)
    clock.tick(FPS)

pygame.quit()