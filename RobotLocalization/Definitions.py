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
        self.velocity = velocity
        self.velMax = 8
        self.velMin = -8
        self.acc = 0.1
        self.omega = 0
        self.omega_acc = 0.0174533
        self.R = 0
        self.theta = 0
        self.dt = 1
        self.font = pygame.font.SysFont('comicsans', 16, False, False)
        self.center = pnt(self.x, self.y)
        self.circle = self.center.buffer(self.radius).boundary
        self.angleDeg = 0
        self.collision = False

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


    def update_velocity(self):
        distances = []
        for sen in self.sensors:
            distances.append(sen.distance)


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
        #increment v
        if keys[pygame.K_w]:
            self.velocity += self.acc
        #decrement v
        if keys[pygame.K_s]:
            self.velocity -= self.acc
        #decrement omega
        if keys[pygame.K_a]:
            self.omega -= self.omega_acc
        #increment omega
        if keys[pygame.K_d]:
            self.omega -= self.omega_acc
        if keys[pygame.K_x]:
            self.velocity = 0
            self.omega = 0
        # Constrain velocity    -NEEDS WORK-constrains wrongly for top-also limit negative speeds
        if self.velocity >= self.velMax:
            self.velocity = self.velMax

        if self.velocity <= self.velMin:
            self.velocity = self.velMin


    def move(self, walls):
        """
        move resolves the kinematics of the robot
        it also updates the time attribute
        """
        oldX = self.x
        oldY = self.y
        oldTheta = self.theta
        state = np.array([oldX, oldY, oldTheta])
        rotation = np.array([[self.dt * math.cos(oldTheta), 0], [self.dt * math.sin(oldTheta), 0], [0, self.dt]])
        speed = np.array([self.velocity, self.omega])
        new_attributes = state + np.matmul(rotation, speed)
        self.x = new_attributes[0]
        self.y = new_attributes[1]
        self.theta = new_attributes[2]


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
        #self.time += self.dt
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


def redrawGameWindow(win, robot, walls):
    win.fill((WHITE_2))
    for wall in walls:
        wall.draw(win)
    robot.draw(win)
    pygame.display.update()