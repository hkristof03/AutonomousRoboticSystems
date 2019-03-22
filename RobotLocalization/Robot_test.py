import Definitions as defs_
from Definitions import pygame, Point, Wall, Robot, Landmark, redrawGameWindow


window_width = 800
window_height = 600

pygame.init()
win = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("BumbleBeeN'TheHood")
clock = pygame.time.Clock()
FPS = 120

start_point = defs_.Point(100, 100)
robot = Robot(start_point, 30, 1, 12, 80, 0, 1)
robot.create_adjust_sensors()

wall_right = Wall(Point(750, 50), Point(750, 550))
wall_left = Wall(Point(50, 50), Point(50, 550))
wall_top = Wall(Point(50, 50), Point(750, 50))
wall_bottom = Wall(Point(50, 550), Point(750, 550))

walls = [wall_right, wall_left, wall_top, wall_bottom]

landmark1 = Landmark(Point(500, 250))
landmark2 = Landmark(Point(300, 450)) 
landmark3 = Landmark(Point(100, 250))
landmark4 = Landmark(Point(600, 150))
landmark5 = Landmark(Point(200, 400))
landmark6 = Landmark(Point(600, 350))


landmarks = [landmark1, landmark2, landmark3, landmark4, landmark5, landmark6]

run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    keys = pygame.key.get_pressed()
    robot.key_input(keys)
    robot.move(walls)

    bias_x = robot.x - robot.prev_x
    bias_y = robot.y - robot.prev_y
    d_theta = robot.angleDeg - robot.prev_theta

    robot.prev_theta = robot.angleDeg
    robot.prev_x = robot.x
    robot.prev_y = robot.y
    biasPoint = Point(bias_x, bias_y)

    robot.update_sensors(biasPoint, d_theta)
    robot.calculate_intersection(walls)

    redrawGameWindow(win, robot, walls, landmarks)
    clock.tick(FPS)




