from pso_definitions import *

path = 'D:\Maastricth University\Semester 2\Autonomous Robotic Systems\PSO assignment\Rosenbrock_test/'


x = np.linspace(x_begin*2, x_end*2, 100)
y = np.linspace(y_begin*2, y_end*2, 100)

X, Y = np.meshgrid(x, y)
fig = plt.figure(figsize=(20,20))

Z = np.array([function_rosenbrock((x, y)) for x, y in zip(np.ravel(X), np.ravel(Y))])  # calculating values of function
Z = Z.reshape(X.shape)

#initialisation of particles
nparticles = 20 # number of particles
dt = 1 # time step
particles = []
best_value = 1000

for i in range(nparticles):
    #particles.append(Particle(2 * x_end * np.random.random(2) - x_end, [0, 0]))  # case of random position, no velocity!
    #
    particles.append(Particle(x_end * np.random.random(2), [0, 0]))

c = [0.9, 2.0, 2.0]
number_of_iterations = 100

# main loop
for i in range(number_of_iterations):
    values = []
    ax = fig.add_subplot(111, projection='3d')

    for particle in particles:  # calculation loop
        particle.get_value(function_rosenbrock)
        values.append(particle.value)

    for particle in particles:  # update loop
        if min(values) < best_value:
            print('Iteration:', i, 'New minimum found: '+str(min(values)))
            best_value = min(values)

        particle.update_velocity(particles[values.index(min(values))].position, c)
        particle.update_position(dt)
        #This line is to visualize the particles position in each iteration.
        ax.scatter(particle.position[0], particle.position[1], particle.value, s=50, c='r')


    #For visualizing the surface and the particles position in each iteration.
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.08,color='blue', alpha=0.2, antialiased=True)
    fig.savefig(path + 'temp_rosenbrock_{}.png'.format(i), dpi=fig.dpi)
    plt.clf()

    if (c[0] > 0.4) and (i % 25 == 0) and (i > 0):
            c[0] -= 0.05
            print('value of c[0] - inertia coefficient is:', c[0])

    else:
        continue



image_folder = 'Rosenbrock'
video_name = 'Rosenbrock.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=natural_keys)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(path + video_name, 0, 3, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
