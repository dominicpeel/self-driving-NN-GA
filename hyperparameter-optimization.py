"""
N = [10,25,50]
S = [(4,9,2), (4,5,5,2), (4,4,4,4,2)]
C = (0.1, 0.15, 0.3)
M = (0.05, 0.1, 0.25)

hyperparameters = []
for n in N:
    for s in S:
        for c in C:
            for m in M:
                hyperparameters.append([n,s,c,m])

with open("hyperparameters", "w+") as f:
    for hyperparameter in hyperparameters:
        f.write(str(hyperparameter)+"\n")
"""
import hyperparameter_genetic
best_hyperparameters = {
    "hyperparameters": [],
    "score": 0
}

with open("hyperparameters", "r") as hyperparameters:
    for p in hyperparameters:
        p = eval(p)
        mean_fitness = hyperparameter_genetic.genetic(p[0], p[1], p[2], p[3])
        with open("hyperparameters-fitness", "a") as fitness:
            fitness.write(str(p)+"~"+str(mean_fitness)+"\n")
        if mean_fitness > best_hyperparameters["score"]:
            best_hyperparameters["hyperparameters"] = list(p)
print(best_hyperparameters)
