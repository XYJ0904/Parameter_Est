import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import openseespy.opensees as ops
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, rosen, rosen_der
import pygad
import numpy.matlib

global count_overall
count_overall = 0
sample_ratio = 5

source_experiment = "GLBD8"
num_generations = 10 # Total Generation

source_parameter = source_experiment
Data = pd.read_csv('./Experiment/%s.txt' % source_experiment, header = None)
use_initial_population = False
use_initial_range = True

index_cons = 35980
Disp_exp = Data[4][0:index_cons].values
Force_exp = Data[1][0:index_cons].values/10**3
Disp_exp = Disp_exp[::sample_ratio]
Force_exp = Force_exp[::sample_ratio]

if use_initial_population:
    initial_population = np.loadtxt("./Extract_Para/Extract_para_%s.csv" % source_parameter, delimiter=",", dtype="float")
    sol_per_pop = initial_population.shape[0] # number of population in each generation
    num_genes = initial_population.shape[1] # number of genes (flexible parameters)
else:
    sol_per_pop = 100 # number of population in each generation
    num_genes = 80 # number of genes (flexible parameters)
    initial_population = None

num_parents_mating = 10 # selected parent population
parent_selection_type = "sss"
keep_parents = 10 # kept parent population
crossover_type = "single_point" # cross
mutation_type = "random" # mutation type is random mutation
mutation_percent_genes = 5 # mutation proportion

houzhui = "%s_%s_%s_%s_%s_%s_%s_%s" % (source_experiment, num_generations, num_parents_mating, keep_parents, sol_per_pop, num_genes, use_initial_population, use_initial_range)
print(houzhui)

if not os.path.exists(houzhui):
    os.mkdir(houzhui)

# Functin to calculate the energy
def energy(force, Disp):
    disp_diff = np.diff(Disp)
    force = np.array(force)
    int_force = np.abs(force[:-1] * disp_diff)
    energy = int_force.cumsum()
    return energy


def OS_solution(Disp, solution):

    # input the Disp and solution (parameters), and get the reaction force for output

    return force

# intital solution for GA, 40/80 parameters
# uniaxialMaterial('Pinching4', matTag, ePf1, ePd1, ePf2, ePd2, ePf3, ePd3, ePf4, ePd4, <eNf1, eNd1, eNf2, eNd2, eNf3, eNd3, eNf4, eNd4>, rDispP, rForceP, uForceP, <rDispN, rForceN, uForceN>, gK1, gK2, gK3, gK4, gKLim, gD1, gD2, gD3, gD4, gDLim, gF1, gF2, gF3, gF4, gFLim, gE, dmgType)
solution=[
    #floating point values defining force points on the positive response envelope
    #stress1 strain1 stress2 strain2 stress3 strain3 stress4 strain4
    25, 0.5, 37, 1, 45, 2.5, 15, 5,
    #floating point values defining force points on the negative response envelope
    #stress1 strain1 stress2 strain2 stress3 strain3 stress4 strain4
    -25, -0.5, -37, -1, -45, -2.5, -15, -5,
     # floating point value defining the ratio of the deformation at which reloading occurs to the maximum historic deformation demand
    0.5,
    # floating point value defining the ratio of the force at which reloading begins to force corresponding to the maximum historic deformation demand
    0.25,
    # floating point value defining the ratio of strength developed upon unloading from negative load to the maximum strength developed under monotonic loading
    0.05,
    # floating point value defining the ratio of the deformation at which reloading occurs to the minimum historic deformation demand
    0.5,
    # floating point value defining the ratio of the force at which reloading begins to force corresponding to the minimum historic deformation demand
    0.25,
    # floating point value defining the ratio of strength developed upon unloading from negative load to the minimum strength developed under monotonic loading
    0.05,
    # floating point values controlling cyclic degradation model for reloading stiffness degradation
    #1.0, 0.2, 0.3, 0.2, 0.9,
    0.1, 0.1, 0.1, 0.1, 0.1,
    # floating point values controlling cyclic degradation model for unloading stiffness degradation
    #0.5, 0.5, 2.0, 2.0, 0.5,
    0.1, 0.1, 0.1, 0.1, 0.1,
    # floating point values controlling cyclic degradation model for strength degradation
    #1.0, 0.5, 1.0, 1.0, 0.9,
    0.1, 0.1, 0.1, 0.1, 0.1,
    # floating point value used to define maximum energy dissipation under cyclic loading. Total energy dissipation capacity is defined as this factor multiplied by the energy dissipated under monotonic loading.
    10,
    -0.4,  10]


if use_initial_range:
    gene_space = []
    initial_range = np.loadtxt("./Extract_Para/Extract_para_%s.csv" % source_parameter, delimiter=",", dtype="float")
    for col in range(initial_range.shape[1]):
        sel_col = initial_range[:, col]
        min_value = np.min(sel_col)
        max_value = np.max(sel_col)
        gene_space.append({'low': min_value, 'high': max_value})

else:
    low_1 = 0.25
    up_1 = 5

    space = (np.array(solution), np.array(solution))
    gene_space = []
    for i in range(len(solution)):
        gene_space.append({'low': space[0][i]*low_1, 'high': space[1][i]*up_1})
    gene_space = gene_space + gene_space
# print(np.array(gene_space).shape)


def fitness_func(solution, solution_idx):
    Force_OS = OS_solution(Disp_exp, solution, "Optim")
    # Energy_exp=energy(Force_exp,Disp_exp)
    # Energy_OS=energy(Force_OS,Disp_exp[:-1])
    # loss= mean_squared_error(Energy_exp[:-1],Energy_OS)   #+ mean_squared_error(Force_exp[:-1],Force_OS)
    if len(Force_OS) != len(Force_exp):
        loss = 1e10
        fitness = 1.0 / (loss + 0.000001)
        return fitness
    else:
        loss = mean_squared_error(Force_exp, Force_OS) # + mean_squared_error(Energy_exp[:-1],Energy_OS)
        if np.isnan(loss):
            loss = 1e10
        fitness = 1.0 / (loss + 0.000001)
        return fitness


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       gene_type=[float, 2],
                       mutation_type=mutation_type,
                       initial_population=initial_population,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_percent_genes=mutation_percent_genes)
                       # save_best_solutions=True)
                       #gene_type=[float, 3])


# print(ga_instance.initial_population)

ga_instance.run()
ga_instance.plot_fitness() 

solution_best, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution_best))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
solution_Loss = (1.0 / solution_fitness) - 1e-6
np.savetxt("%s/%.2f_solution_best_%s.csv" % (houzhui, solution_Loss, houzhui), solution_best, delimiter=",", fmt="%s")

force_OS = OS_solution(Disp_exp, solution_best, "Best")
plt.title("Best Force-Disp %s" % houzhui)
plt.plot(Disp_exp, Force_exp)
plt.plot(Disp_exp, force_OS)
#plt.xlim((-5,5))
plt.grid('true')
plt.savefig("%s/Best_Force_Disp_%s.png" % (houzhui, houzhui), dpi=720)
np.savetxt("%s/Best_Force_Disp_%s.csv" % (houzhui, houzhui), np.vstack((Disp_exp, Force_exp, force_OS)).T, delimiter=",", fmt="%s")

plt.figure()
plt.title("Best Energy %s" % houzhui)
plt.plot(energy(Force_exp, Disp_exp))
plt.plot(energy(force_OS, Disp_exp))
plt.grid('true')
plt.savefig("%s/Best_Energy_%s.png" % (houzhui, houzhui), dpi=720)

# plt.show()
plt.close()
