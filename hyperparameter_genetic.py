import environment
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def genetic(N, S, C, M):
    def graph_fitness(generation):
        x_axis = []
        y_axis = []

        for index, info in enumerate(generation_info):
            x_axis.append(index+1)
            y_axis.append(info["mean"])
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.plot(x_axis, y_axis, "x")

        x_axis = []
        y_axis = []
        for index, info in enumerate(generation_info):
            x_axis.append(index+1)
            y_axis.append(info["best"])
        plt.plot(x_axis, y_axis, "x")
        plt.savefig("graphs/generation"+str(generation+1)+".png")

    generation_info = [] # {"mean":, "best":}

    population = N
    generations = 10
    mutation_rate = M
    selection_size = C
    network_layers = S

    #Initialization - Create candidates
    class Candidate():
        def __init__(self, layers):
            self.weights = []
            self.bias = []
            for index, layer in enumerate(layers):
                if index > 0:
                    self.weights.append((np.random.uniform(-1, 1, (layer,layers[index-1]))))
                    self.bias.append(np.random.uniform(-1, 1, (layer,1)))
            self.fitness = 0
            self.probability = 0 #Probability of being selected in genetic selection, based on fitness.
            self.flat_weights = []
            self.flat_bias = []

    candidates = []
    for i in range(population):
        candidates.append(Candidate(network_layers))

    #Run generations
    for generation in range(generations):
        print("Generation:",generation+1)
        #Simulate 5 candidates at a time (to increase FPS/performance)
        for index, candidate in enumerate(candidates):
            if index % 10 == 0:
                testing_candidates = [candidate for candidate in candidates[index:index+10]]
                for i, candidate in enumerate(environment.simulate(testing_candidates)):
                    candidates[index+i].fitness = candidate.fitness

        #Selection - Pick 3 fittest candidates
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        print("Best fitness:", candidates[0].fitness)
        #Save fittesst cars brain
        np.save("brains/generation"+str(generation+1)+"-weights", candidates[0].weights)
        np.save("brains/generation"+str(generation+1)+"-bias", candidates[0].bias)
        with open("brains/generation"+str(generation+1)+"-metadata.txt", "w+") as metadata:
            metadata.write("fitness:"+str(candidates[0].fitness))
        mean_fitness = sum(candidate.fitness for candidate in candidates)/len(candidates)
        generation_info.append({"mean": mean_fitness, "best": candidates[0].fitness})
        graph_fitness(generation)
        selected = candidates[:2]
        #Genetic selection - chose 25% of population to breed
        print("Mean fitness:", sum(candidate.fitness for candidate in candidates)/len(candidates))
        #Genetic crossover
        child_candidates = []
        while len(child_candidates) < population:
            #Choose two random parents from selected population
            parents = np.random.choice(selected, 2, replace=False) #np.random.choice(candidates, 2, p=[candidate.probability for candidate in candidates], replace=False)
            children = [Candidate(network_layers) for n in range(2)]
            #Flatten weights and bias
            for index, parent in enumerate(parents):
                parent.flat_weights = []
                parent.flat_bias = []
                for layer in parent.weights:
                    children[index].flat_weights.append(layer.flatten())
                for bias in parent.bias:
                    children[index].flat_bias.append(bias.flatten())
            #Crossover parents' weights at random point
            for index, weight in enumerate(children[0].flat_weights):
                crossover_index = np.random.randint(0, len(children[0].flat_weights[index]))
                temp = children[1].flat_weights[index][:crossover_index].copy()
                children[1].flat_weights[index][:crossover_index], children[0].flat_weights[index][:crossover_index]  = children[0].flat_weights[index][:crossover_index], temp
            for child in children:
                child.weights = []
                for index, layer in enumerate(network_layers):
                    if index > 0:
                        child.weights.append(np.asmatrix(child.flat_weights[index-1]).reshape((layer,network_layers[index-1]))) #(np.random.uniform(-1, 1, (layer,layers[index-1])))
            #Crossover parents' bias at random point
            for index, weight in enumerate(children[0].flat_bias):
                crossover_index = np.random.randint(0, len(children[0].flat_bias[index]))
                temp = children[1].flat_bias[index][:crossover_index].copy()
                children[1].flat_bias[index][:crossover_index], children[0].flat_bias[index][:crossover_index]  = children[0].flat_bias[index][:crossover_index], temp
            for child in children:
                child.bias = []
                for index, layer in enumerate(network_layers):
                    if index > 0:
                        child.bias.append(np.asmatrix(child.flat_bias[index-1]).reshape(layer,1))
            #Mutate weights in children
            for child in children:
                if mutation_rate > 0:
                    for index, weight in enumerate(child.weights):
                        mask = np.random.choice(2,size=weight.shape,p=[1-mutation_rate, mutation_rate]).astype(np.bool) #,p=[1-prob, prob]
                        r = np.random.uniform(-1, 1, weight.shape)
                        child.weights[index][mask] = r[mask]
                    for index, bias in enumerate(child.bias):
                        mask = np.random.choice(2,size=bias.shape,p=[1-mutation_rate, mutation_rate]).astype(np.bool)
                        r = np.random.uniform(-1, 1, bias.shape)
                        child.bias[index][mask] = r[mask]
            for child in children:
                if len(child_candidates) < population:
                    child_candidates.append(child)
        candidates = []
        candidates = child_candidates.copy()
        if mutation_rate > 0.01:
            mutation_rate -= 0.01
    return mean_fitness
