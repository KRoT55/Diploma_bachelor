import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils.np_utils import to_categorical
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
X_train = X_train.reshape(5216, 150, 150, 3)
X_test = X_test.reshape(624, 150, 150, 3)


n_input = X_train # 28x28 = 784
n_hidden = 500 # hidden layer n neurons
n_classes = y_train # digits 0-9


class Neuron(object):

    def __init__(self, w_count, weights):

        self.w_count = w_count
        if weights is None:
            self.weights = np.random.rand(w_count)
        else:
            self.weights = weights

    def mutate(self, rate):
        for ind, val in enumerate(self.weights):
            random_value = np.random.random_sample()
            if abs(random_value) < rate:
                self.weights[ind] = np.random.uniform(-1.0, 1.0, 1)

    def crossover(self, other):
        offspring = np.empty(self.w_count)
        # 1-point crossover
        point = np.random.randint(0, len(self.weights) - 1)
        offspring[0:point] = self.weights[0:point]
        offspring[point:] = [0.5*sum(x) for x in zip(self.weights[point:], other.weights[point:])]
        return offspring

    def __repr__(self):
        return 'Neuron weights ' + str(self.weights)


class Layer(object):

    def __init__(self, w_count, n_count, neurons):

        # Create individuals
        self.neurons = []
        self.w_count= w_count
        self.n_count = n_count
        if neurons is None:
            for i in range(n_count):
                weights = np.random.rand(w_count)
                n = Neuron(w_count, weights)
                self.neurons.append(n)
        else:
            self.neurons.extend(neurons)

    def mutate(self, rate):
        for n in self.neurons:
            n.mutate(rate)

    def crossover(self, other):
        of_neurons =[]
        for ind, item in enumerate(self.neurons):
            neu_list = self.neurons[ind].crossover(other.neurons[ind])
            of_neu = Neuron(self.w_count, neu_list)
            of_neurons.append(of_neu)

        return of_neurons

class Net(object):

    def __init__(self, layers):

        if layers is None:
            self.layers = []
            self.layers.append(Layer(n_input , n_hidden, neurons=None))
            self.layers.append(Layer(n_hidden , n_classes, neurons=None))
        else:
            self.layers = layers

    def set_fitness(self, val):
        self.fitness = val

    def mutate(self, rate):
        for l in self.layers:
            l.mutate(rate)

    #def crossover(self, other):


class Population(object):

    def __init__(self, pop_size=10, mutate_rate=0.01):
        """
            Args
                pop_size: size of population
                fitness_goal: goal that population will be graded against
        """
        self.pop_size = pop_size
        self.mutate_rate = mutate_rate
        self.fitness_history = []
        self.done = False

        # Create individuals
        self.individuals = []
        for x in range(pop_size):
            self.individuals.append(Net(layers = None))

    def crossover(self):
        pair_size = np.uint8(self.pop_size/2)

        for i in range(0,pair_size,2):
            par1 = self.individuals[i]
            par2 = self.individuals[i+1]
            nlayers = []
            for ind, item in enumerate(par1.layers):
                print('par1 layer ', par1.layers[ind])
                new_layer = par1.layers[ind].crossover(par2.layers[ind])
                nlayers.append(Layer(par1.layers[ind].w_count, par1.layers[ind].n_count, new_layer))
            self.individuals.append(Net(layers=nlayers))

    def best_fitness(self):
        """
            Grade the generation by getting the average fitness of its individuals
        """
        max_fit = max(node.fitness for node in self.individuals)
        print("Best fitness: ", max_fit)
        print('fit ', [node.fitness for node in self.individuals])

    def selection(self):
        """
            Select the fittest individuals to be the parents of next generation (lower fitness it better in this case)
            Also select a some random non-fittest individuals to help get us out of local maximums
        """
        # Sort individuals by fitness (we use reversed because in this case higher fitness is better)
        print('len ', len(self.individuals))
        self.individuals = list(sorted(self.individuals, key=lambda x: x.fitness, reverse=True))
        #print('indivs ', self.individuals)

        # then cut
        self.individuals = self.individuals[:self.pop_size]

    def mutation(self):
        for i in self.individuals:
            i.mutate(self.mutate_rate)

    def evolve(self):
        # 1. Select fittest
        self.selection()
        # 2. Create children and new generation
        self.crossover()
        # 3. Reset parents and children
        self.mutation()

def mlp_config(n_input, n_hidden, n_classes, layer0, layer1):
    x = tf.placeholder("float", [None, n_input], name='x')
    y = tf.placeholder("float", [None, n_classes], name='y')
    lay0 = []
    for n in layer0.neurons:
        lay0.append(n.weights)

    lay1 = []
    for n in layer1.neurons:
        lay1.append(n.weights)

    data_np0 = np.asarray(lay0, np.float32)
    data_np1 = np.asarray(lay1, np.float32)

    weights = {
    #    'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
     #   'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        'h1': tf.Variable(tf.convert_to_tensor(data_np0.transpose(),dtype=tf.float32)),
        'out': tf.Variable(tf.convert_to_tensor(data_np1.transpose(),dtype=tf.float32))
    }
    #print('weights', weights)
    return x, y, weights

def mlp_model(x, y, weights):
    hidden = tf.nn.relu(tf.matmul(x, weights['h1']))
    logits = tf.matmul(hidden, weights['out'])
    pred   = tf.one_hot(tf.cast(tf.argmax(logits, 1), tf.int32), depth=10)
    return pred, logits

def get_accuracy(pred, y):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == "__main__":
    pop_size = 20
    mutate_rate = 0.2

    pop = Population(pop_size=pop_size, mutate_rate=mutate_rate)
    for i in pop.individuals:
        x, y, weights = mlp_config(n_input, n_hidden, n_classes, i.layers[0], i.layers[1])
        pred, logits = mlp_model(x, y, weights)
        accuracy = get_accuracy(pred, y)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        i.set_fitness(sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
        sess.close()

    SHOW_PLOT = False
    GENERATIONS = 5

    for gen in range(GENERATIONS):
        print('GENERATION ', gen)
        pop.best_fitness()
        pop.evolve()

        for i in range(pop_size, len(pop.individuals)):
            i = pop.individuals[i]
            x, y, weights = mlp_config(n_input, n_hidden, n_classes, i.layers[0], i.layers[1])
            pred, logits = mlp_model(x, y, weights)
            accuracy = get_accuracy(pred, y)
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            i.set_fitness(sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
            sess.close()

    # Plot fitness history
    if SHOW_PLOT:
        print("Showing fitness history graph")
        #matplotlib.use("MacOSX")
        plt.plot(np.arange(len(pop.fitness_history)), pop.fitness_history)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.title('Fitness - pop_size {} mutate_prob {} retain {} random_retain {}'.format(pop_size, mutate_prob, retain, random_retain))
        plt.show()