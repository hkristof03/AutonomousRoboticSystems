import Definitions as defs_
from Definitions import pygame, Point, Layout, Dust, Robot, GeneticAlgorithm, pd, run_GA, mean, redrawGameWindow, RecurrentNeuralNetwork


window_width = 800
window_height = 600

pygame.init()
win = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("BumbleBeeN'TheHood")
clock = pygame.time.Clock()
FPS = 30
start_point = defs_.Point(100, 100)
layout = 'double trapezoid'
walls = Layout(layout)
dust = Dust(150, window_width, window_height, 8, win)

generation = 0
number_of_individuals = 20
number_of_winners = 4
NN = RecurrentNeuralNetwork(12, 2, 6)
robots = []
for i in range(number_of_individuals):
    robots.append(Robot(start_point, 30, 1, 12, 80, 10, 1, NN))
    robots[i].create_adjust_sensors()

# Genetic Algorithm class instance
GA = GeneticAlgorithm()

# parents and children proportion
proportion = 0.25

pk = int(number_of_individuals * proportion)  # parents number
ck = int(number_of_individuals - pk)  # children number

vals_bf = []
vals_mf = []
vals_avg = []
file_name = 'Generations_data_test_4.json'
file_name_df = 'Generations_data_test_4.csv'

run = True
while run:

    values = []
    for robot in robots:

        dust.renew()
        dt = 0
        terminate = False
        while (dt < 2000) and (terminate == False):

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
            biasPoint = Point(bias_x, bias_y)

            robot.update_sensors(biasPoint, d_theta)
            robot.calculate_intersection(walls)

            robot.fitness_function()
            terminate = robot.terminate_check()
            dt += robot.dt

            redrawGameWindow(win, robot, walls, dust)
            clock.tick(240)

        robot.reposition(start_point)
        #print(robot.fitnessScore)
        values.append(robot.fitnessScore)

        dust.delete()

    val_bf = max(values)
    val_mf = min(values)
    val_avg = mean(values)
    vals_bf.append(val_bf)
    vals_mf.append(val_mf)
    vals_avg.append(val_avg)
    d = {'Best_fitness_score': vals_bf,
         'Minimum_fitness_score': vals_mf,
         'AVG_fitnessScore': vals_avg}
    print("max:", val_bf, "min:", val_mf, "avg:", val_avg)
    df = pd.DataFrame.from_dict(d)
    df.to_csv(file_name_df, index=False)
    robots = run_GA(robots, pk, ck, GA, file_name, generation)