import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as sigmoid_activation_function
import random
from operator import attrgetter

settings = {}
# EVOLUTION SETTINGS
settings['pop_size'] = 50       # number of organisms
#settings['food_num'] = 100      # number of food particles
#settings['gens'] = 50           # number of generations
#settings['elitism'] = 0.20      # elitism (selection bias)
settings['mutate'] = 0.10       # mutation rate

# SIMULATION SETTINGS
#settings['gen_time'] = 100      # generation length         (seconds)
#settings['dt'] = 0.04           # simulation time step      (dt)
#settings['dr_max'] = 720        # max rotational speed      (degrees per second)
#settings['v_max'] = 0.5         # max velocity              (units per second)
#settings['dv_max'] =  0.25      # max acceleration (+/-)    (units per second^2)

#settings['x_min'] = 0        # arena western border
#settings['x_max'] = 800        # arena eastern border
#settings['y_min'] = 0        # arena southern border
#settings['y_max'] = 600        # arena northern border

#settings['plot'] = False        # plot final generation?

# ORGANISM NEURAL NET SETTINGS
settings['inodes'] = 12          # number of input nodes
settings['hnodes'] = 24          # number of hidden nodes
settings['onodes'] = 2          # number of output nodes


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

    def selBest(self, individuals, k, fit_attr="fitness"):
        """Select the *k* best individuals among the input *individuals*. The
        list returned contains references to the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list containing the k best individuals.
        """
        return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]

    def selRoulette(self, individuals, k, fit_attr="fitness"):
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

    def selTournament(self, individuals, k, tournsize, fit_attr="fitness"):
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



class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
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

        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc = mean, scale = sd)


    def create_chromosome(self):

        wih = self.weights_in_hidden.flatten()
        woh = self.weights_hidden_out.flatten()
        chromosome = np.concatenate((wih, woh), axis=None)
        return chromosome


    def update_weights(self, chromosome):

        self.weights_in_hidden = chromosome[:self.no_of_in_nodes * self.no_of_hidden_nodes].reshape(self.no_of_hidden_nodes,
                                                                                                    self.no_of_in_nodes)
        self.weights_hidden_out = chromosome[self.no_of_in_nodes * self.no_of_hidden_nodes:].reshape(self.no_of_out_nodes,
                                                                                                     self.no_of_hidden_nodes)


    def train(self):
        #bias node has to be added to the input vector!!!
        pass

    def run(self, input_vector):
        '''
        Running the network with an input vector input_vector, which can be a tuple,
        list or ndarray.
        '''
        #turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = sigmoid_activation_function(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = sigmoid_activation_function(output_vector)

        return output_vector


