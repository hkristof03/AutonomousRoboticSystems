import pygame, pygame.gfxdraw, math, random
from shapely.geometry import LineString
from shapely.geometry import Point as pnt

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



def CorrY(y):
    return (maxY - y)

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def distance(self, object):
        return math.sqrt((object.x - self.x)**2 + (object.y - self.y)**2)




class Dust( object):
    """
    Creates the dust for the whole map, comprised of the individual dust specks
    defined by the dust_speck class
    """
    def __init__(self, speckNumber, maxX, maxY, radius):
        self.specks = []
        self.speckNumber = speckNumber
        self.maxX = maxX
        self.maxY = maxY
        self.radius = radius
        for i in range(self.speckNumber):
            x = random.randint(self.radius, self.maxX - self.radius)
            y = random.randint(self.radius, self.maxY - self.radius)
            d = dust_speck(Point(x,y), self.radius)
            self.specks.append(d)
            d.draw()

    def draw(self):
        for speck in self.specks:
            speck.draw()






class dust_speck(object):
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
            self.angleRad =  math.radians(self.angleDeg)
        else:
            self.angleRad = math.atan((self.y2 - self.y1)/(self.x2-self.x1))
            self.angleDeg = math.degrees(self.angleRad)
        self.bound = LineString([(self.x1, self.y1), (self.x2, self.y2)])

    def draw(self):
        pygame.draw.line(win, (0, 0, 0), (self.x1,CorrY(self.y1)), (self.x2, CorrY(self.y2)), 5)

class Sensor(object):
    def __init__(self, startpoint, endpoint, width):
        self.x1 = startpoint.x
        self.y1 = startpoint.y
        self.x2 = endpoint.x
        self.y2 = endpoint.y
        self.width = width

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
        #Nik's attributes
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
        self.center = pnt(self.x,self.y)
        self.circle = self.center.buffer(self.radius).boundary
        self.angleDeg = 0
        self.collision = False
        self.collisionScore = 0
        self.velocityScore = 0
        self.fitnessScore = 0
        self.dustEaten = 0

    def create_adjust_sensors(self):
        self.sensor_range += self.radius
        for i in range(self.num_sensors):
            s = Sensor(Point(self.x, self.y), Point(self.x, self.y - self.sensor_range), self.sensor_width)
            angle = 360 / self.num_sensors * i
            s.set_sensor_direction(Point(self.x,self.y), angle, self.radius)
            self.sensors.append(s)

    def update_sensors(self, biaspoint, d_theta):
        for sen in self.sensors:
            sen.update_sensor_position(biaspoint)
            #last parameter the rotation degree
            sen.update_rotate_sensor_line(self.x, self.y, d_theta)

    def calculate_fitness(self):
        """
        Calculates the fitness of the robot.
        Each timestep it increases the score by the absolure value of the velocity, times a constant beta,
        and reduces the score if the object is colliding by alpha
        """
        alpha = -50 #constant to adjust weight of collisions
        beta = 100   #constant to adjust weight of velocity
        col = 0     #col is used to completely discount the velocity contribution if the object is colliding


        if self.collision:
            self.collisionScore +=1 #total dt that the robot has been colliding
            col = 1

        """

        self.velocityScore += (abs(self.velocity))* (1- col) #total positive score from the velocity
        self.fitnessScore = alpha * self.collisionScore + beta * self.velocityScore 
        """

        self.fitnessScore = alpha * self.collisionScore + beta  * self.dustEaten
        print(self.fitnessScore)



    def calculate_intersection(self, walls_):

        for sen in self.sensors:
            s2 = LineString([(sen.x1, sen.y1), (sen.x2, sen.y2)])

            for wall_ in walls:
                #FIX_AFTER
                #s1 = LineString([(wall_.x1, wall_.y1), (wall_.x2, wall_.y2)])
                ip = s2.intersection(wall_.bound)
                if ip:
                    sen.calculate_distance(ip)
                    break
                    # print("distance:", sen.distance)
                else:
                    sen.distance = int(self.sensor_range / 2)

    def imminent_collision(self, wall):
        
        
        self.center = pnt(self.x,self.y)
        self.circle = self.center.buffer(self.radius + 2).boundary
        collision = self.circle.intersection(wall.bound)
        return collision

    def eat_dust(self, Dust):
        """
        Checks if a dust_speck's center is inside the rumba. If so it gets
        eaten, and the dustEaten score increases, which is used later to
        calculate the fitness function.

        """
        for speck in Dust.specks:
            if (speck.x - self.x)**2 + (speck.y - self.y)**2 <= self.radius**2:
                Dust.specks.remove(speck)
                self.dustEaten += 1



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

        #COLLISION

        #doubleCollision = False
        #singleCollision = False
        otherWalls = walls.copy()

        self.collision = False

        for wall_ in walls:
            otherWalls.remove(wall_)
            col = robot.imminent_collision(wall_)       
            velSign = self.velocity/ (abs(self.velocity) + 0.000001) 


            

            if col:
                self.x -= self.velocity * math.cos(self.theta) * self.dt
                self.y -= self.velocity * math.sin(self.theta) * self.dt  

                self.x += 0.5 * self.velocity * math.cos(self.theta) * self.dt
                self.y += 0.5 * self.velocity * math.sin(self.theta) * self.dt 

                self.collision = True

                col1 = robot.imminent_collision(wall_) #NEED to just check for previous wall.




                if col1:

                    #singleCollision = True

                    self.x -= 0.5 * self.velocity * math.cos(self.theta) * self.dt
                    self.y -= 0.5 * self.velocity * math.sin(self.theta) * self.dt

                    self.x += self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (math.cos(math.radians(wall_.angleDeg )))
                    self.y += self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (math.sin(math.radians(wall_.angleDeg )))

                    

                    for otherWall in otherWalls:

                        col2 = robot.imminent_collision(otherWall)
                        if col2:
                            self.x -= self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (math.cos(math.radians(wall_.angleDeg )))
                            self.y -= self.velocity * (math.cos(self.theta - math.radians(wall_.angleDeg))) * (math.sin(math.radians(wall_.angleDeg )))                            
                            doubleCollision = True


                

        #END COLLISION


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
                   CorrY( math.floor(self.y) + math.floor(outerRad * math.sin(self.theta))))

        polygonPoints = (pos, lineEnd, pos)
        pygame.gfxdraw.filled_circle(win, pos[0], pos[1], outerRad, YELLOW)
        pygame.gfxdraw.aacircle(win, pos[0], pos[1], outerRad, BLACK)
        pygame.gfxdraw.aapolygon(win, polygonPoints, BLACK)

        for sen in self.sensors:
            pygame.draw.line(win, (255, 0, 0), (sen.x1, CorrY(sen.y1)), (sen.x2, CorrY(sen.y2)), sen.width)
            text_ = self.font.render(str(sen.distance), 1, BLUE)
            win.blit(text_, (sen.x1, CorrY(sen.y1)))
        #WHEELS
        textPosL = (self.x + (math.cos(self.theta + math.radians(90)) - 0.15) * textDistance,
                   CorrY(self.y + (math.sin(self.theta + math.radians(90))) * textDistance - math.sin(self.theta) * 5 - 5))
        textPosR = (self.x + (math.cos(self.theta + math.radians(-90)) - 0.15) * textDistance,
                   CorrY(self.y + (math.sin(self.theta + math.radians(-90))) * textDistance - math.sin(self.theta) * 5 - 5))
        textL = self.font.render(format(self.velL, '.2f'), 1, RED)
        win.blit(textL, textPosL)
        textR = self.font.render(format(self.velR, '.2f'), 1, RED)
        win.blit(textR, textPosR)


def redrawGameWindow():
    win.fill((WHITE_2))
    Dust.draw()
    for wall in walls:
        wall.draw()
    robot.draw()
    pygame.display.update()


window_width = 800
window_height = 600

pygame.init()

win = pygame.display.set_mode((window_width, window_height))

pygame.display.set_caption("BumbleBeeN'TheHood")



start_point = Point(200, 200)
robot = Robot(start_point, 40, 1, 12, 40, 10, 1)
robot.create_adjust_sensors()

Dust = Dust(70, maxX, maxY, 15)
#dust1 = dust_speck(Point(10,10), 15)

wall_right = Wall(Point(750, 50), Point(750, 550))
wall_left = Wall(Point(50, 50), Point(50, 550))
wall_top = Wall(Point(50, 50), Point(750, 50))
wall_bottom = Wall(Point(50, 550), Point(750, 550))
wall1 = Wall(Point(300, 0), Point(700, 692.8204))


walls = [wall_right, wall_left, wall_top, wall_bottom, wall1]

run = True
while run:

    pygame.time.delay(30)   #milliseconds delay
    bias_x = 0
    bias_y = 0
    d_theta = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    robot.key_input()
    robot.move()
    robot.eat_dust(Dust)

    bias_x = robot.x - robot.prev_x
    bias_y = robot.y - robot.prev_y
    d_theta = robot.angleDeg - robot.prev_theta

    robot.prev_theta = robot.angleDeg
    robot.prev_x = robot.x
    robot.prev_y = robot.y
    biasPoint = Point(bias_x, bias_y)
    
    robot.update_sensors(biasPoint, d_theta)
    robot.calculate_intersection(walls)
    robot.calculate_fitness()

    redrawGameWindow()




pygame.quit()