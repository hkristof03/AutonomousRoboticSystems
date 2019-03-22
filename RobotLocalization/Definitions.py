import pygame, pygame.gfxdraw, math
from shapely.geometry import LineString
from shapely.geometry import Point as pnt
import numpy as np
import random
import math

global win

global maxX, maxY

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


def CorrY(y):
    return (maxY - y)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, object):
        return math.sqrt((object.x - self.x) ** 2 + (object.y - self.y) ** 2)


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

    def draw(self, win):
        pygame.draw.line(win, (0, 0, 0), (self.x1, CorrY(self.y1)), (self.x2, CorrY(self.y2)), 5)

class Landmark(object):
    """

    """

    def __init__(self, point):
        self.x = point.x
        self.y = point.y
        self.rad = 10

    def draw(self, win):
        pygame.gfxdraw.filled_circle(win, self.x, CorrY(self.y), self.rad, BLUE)


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
        self.omega = 0
        self.theta = 0
        self.dt = 1
        self.font = pygame.font.SysFont('comicsans', 16, False, False)
        self.center = pnt(self.x, self.y)
        self.circle = self.center.buffer(self.radius).boundary
        self.angleDeg = 0
        self.collision = False
        #self.time = 0

        # Neural Network
        # self.NN = NeuralNetwork(12, 2, 12)
        #self.NN = RecurrentNeuralNetwork(12, 2, hidden_neurons)





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

        # def terminate_check_orig(self):
        """
        Checks (after 10 timesteps)if the fitnessScore has not increased. 
        If terminateLimit timesteps have passed, where the timestep does not increase, it 
        terminates the simulation.
        It also terminates the simulation after 2000 timesteps
        """



    def update_velocity(self):
        distances = []
        prev_velocities = []
        for sen in self.sensors:
            distances.append(sen.distance)
        prev_velocities.append(self.velL)
        prev_velocities.append(self.velR)
        velocities = self.NN.run(distances, prev_velocities)
        self.velL, self.velR = velocities[0], velocities[1]

    def calculate_intersection(self, walls):

        for sen in self.sensors:
            s2 = LineString([(sen.x1, sen.y1), (sen.x2, sen.y2)])

            for wall_ in walls:
                ip = s2.intersection(wall_.bound)
                if ip:
                    sen.calculate_distance(ip)
                    break
                else:
                    sen.distance = int(self.sensor_range - self.radius)

    def imminent_collision(self, wall):

        self.center = pnt(self.x, self.y)
        self.circle = self.center.buffer(self.radius).boundary
        collision = self.circle.intersection(wall.bound)
        return collision

    def key_input(self, keys):

        if keys[pygame.K_w]:
            self.velL += self.acc
            self.velR += self.acc
        if keys[pygame.K_s]:
            self.velL -= self.acc
            self.velR -= self.acc
        if keys[pygame.K_d]:
            self.velL += self.acc/5
            self.velR -= self.acc/5
        if keys[pygame.K_a]:
            self.velL -= self.acc/5
            self.velR += self.acc/5
        if keys[pygame.K_x]:
            self.velL = 0
            self.velR = 0

        # Constrain velocity    -NEEDS WORK-constrains wrongly for top-also limit negative speeds
        if self.velL >= self.velMax:
            self.velL = self.velMax

        if self.velR >= self.velMax:
            self.velR = self.velMax

        if self.velL <= self.velMin:
            self.velL = self.velMin

        if self.velR <= self.velMin:
            self.velR = self.velMin




    def move(self, walls):
        """
        move resolves the kinematics of the robot
        it also updates the time attribute
        """
        self.velocity = (self.velL + self.velR) / 2
        # Rotation


        self.omega = (self.velR - self.velL) / self.width
        self.theta = self.theta + self.omega * self.dt


        self.angleDeg = math.degrees(self.theta) % 360
        self.theta = math.radians(self.angleDeg)


        # Movement-Update position
        self.x += self.velocity * math.cos(self.theta) * self.dt
        self.y += self.velocity * math.sin(self.theta) * self.dt

        # COLLISION

        # doubleCollision = False
        # singleCollision = False
        otherWalls = walls.copy()
        for wall_ in walls:
            otherWalls.remove(wall_)
            col = self.imminent_collision(wall_)
            velSign = self.velocity / (abs(self.velocity) + 0.000001)

            self.collision = False

            if col:
                self.x -= self.velocity * math.cos(self.theta) * self.dt
                self.y -= self.velocity * math.sin(self.theta) * self.dt

                self.x += 0.5 * self.velocity * math.cos(self.theta) * self.dt
                self.y += 0.5 * self.velocity * math.sin(self.theta) * self.dt

                col1 = self.imminent_collision(wall_)  # NEED to just check for previous wall.

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

                        col2 = self.imminent_collision(otherWall)
                        if col2:
                            self.x -= self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (
                                math.cos(math.radians(wall_.angleDeg)))
                            self.y -= self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (
                                math.sin(math.radians(wall_.angleDeg)))
                            doubleCollision = True
        # increase time
       # self.time += self.dt
        # END COLLISION

    def draw(self, win):

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



def redrawGameWindow(win, robot, walls, landmarks):
    win.fill((WHITE_2))
    for wall in walls:
        wall.draw(win)
    for landmark in landmarks:
        landmark.draw(win)
    robot.draw(win)
    pygame.display.update()