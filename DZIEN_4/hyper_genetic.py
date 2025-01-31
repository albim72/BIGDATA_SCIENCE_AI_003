import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import random

# Załaduj dane MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizacja
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def build_model(chromosome):
    num_layers, neurons, activation, learning_rate = chromosome
    activation_functions = ['relu', 'sigmoid', 'tanh']
    
    model = Sequential([Flatten(input_shape=(28, 28))])
    for i in range(num_layers):
        model.add(Dense(neurons[i], activation=activation_functions[activation]))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_fitness(chromosome, epochs=3):
    model = build_model(chromosome)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

def initialize_population(pop_size, max_layers, max_neurons, learning_rates):
    population = []
    for _ in range(pop_size):
        num_layers = random.randint(1, max_layers)
        neurons = [random.randint(10, max_neurons) for _ in range(num_layers)]
        activation = random.randint(0, 2)  # 0=relu, 1=sigmoid, 2=tanh
        learning_rate = random.choice(learning_rates)
        chromosome = (num_layers, neurons, activation, learning_rate)
        population.append(chromosome)
    return population

def select_parents(population, fitness):
    selected = random.choices(population, weights=fitness, k=2)
    return selected

def crossover(parent1, parent2):
    child1 = list(parent1)
    child2 = list(parent2)
    
    cross_point = random.randint(1, min(len(parent1[1]), len(parent2[1])) - 1)
    child1[1][:cross_point], child2[1][:cross_point] = child2[1][:cross_point], child1[1][:cross_point]
    child1[2], child2[2] = child2[2], child1[2]
    child1[3], child2[3] = child2[3], child1[3]
    
    return tuple(child1), tuple(child2)

def mutate(chromosome, max_neurons, learning_rates):
    mutation_rate = 0.1
    if random.random() < mutation_rate:
        chromosome = list(chromosome)
        layer_to_mutate = random.randint(0, chromosome[0] - 1)
        chromosome[1][layer_to_mutate] = random.randint(10, max_neurons)
        chromosome[3] = random.choice(learning_rates)
    return tuple(chromosome)

def genetic_algorithm(pop_size, generations, max_layers, max_neurons, learning_rates):
    population = initialize_population(pop_size, max_layers, max_neurons, learning_rates)
    best_solution = None
    best_fitness = 0
    
    for generation in range(generations):
        fitness = [evaluate_fitness(chromosome) for chromosome in population]
        
        for i, f in enumerate(fitness):
            if f > best_fitness:
                best_fitness = f
                best_solution = population[i]
        
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, max_neurons, learning_rates))
            new_population.append(mutate(child2, max_neurons, learning_rates))
        
        population = new_population
        print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}")
    
    return best_solution, best_fitness

# Parametry
pop_size = 10
generations = 5
max_layers = 5
max_neurons = 256
learning_rates = [0.001, 0.01, 0.1]

best_solution, best_fitness = genetic_algorithm(pop_size, generations, max_layers, max_neurons, learning_rates)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
