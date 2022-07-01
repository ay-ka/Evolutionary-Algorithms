from torch import nn
import torch
import numpy as np
from collections import namedtuple
import pdb
import matplotlib.pyplot as plt
import argparse


class GrayWolf:
    
    def __init__(self, number_of_wolfs = 50, max_iteration = 100,
                 parameters_lower_bound = -10, parameters_upper_bound = 10,
                num_parameters = 5):
        
        """ genetic algorithm main class

        Args:
            
            parameters_lower_bound: decision parameters value can not go beyond this value
            
            parameters_upper_bound: decision parameters value can not go beyond this value
            
            crossover_percentage: number of crossover-created childtren in terms of percentage of whole population
            
            max_iteration: number of iteration (generation) to run algorithm
            
            number_of_wolfs: number of wolfs or number of population members
            
            num_parameters : number of decision variables

        Returns:
        
            no return --> initializing (create initial population)

        """
        
        #defining gray wolf settings
        
        self.number_of_wolfs = number_of_wolfs
        
        self.upper_bound = parameters_upper_bound
        
        self.lower_bound = parameters_lower_bound
        
        self.max_iteration = max_iteration
        
        self.wolf = namedtuple("gray_wolf", field_names=["position", "fitness"])
        
        self.alpha_wolf = self.wolf(0,0)
        
        self.betha_wolf = self.wolf(0,0)
        
        self.delta_wolf = self.wolf(0,0)
        
        self.wolf_pop = {"wolf_" + str(wolf_index) : self.wolf(None, None) for wolf_index in range(number_of_wolfs)}
        
        self.a = 2
        
        self.best_fitness = np.zeros([self.max_iteration])
        
        
        
        
        #defining network settings
        
        self.number_of_parameters = num_parameters
        
        

        
        # create initial population 
        
        self.CreateInitPop()
        
        
    def MainLoop(self):
        
        """ main loop of (EA --> graywolf)

        Args:
        
            no argument

        Returns:
        
            no return --> iterate through generation to reach a solution

        """
        
        for iteration in range(1, self.max_iteration):
            
            # update a
            
            self.a = self.a - (iteration * (2 / (self.max_iteration) ) )
            
            for wolf_index in range(self.number_of_wolfs):
            
                omega_new_to_alpha = self.UpdatePositions(wolf_index, Alpha = True)
                
                omega_new_to_betha = self.UpdatePositions(wolf_index, Betha = True)
                
                omega_new_to_delta = self.UpdatePositions(wolf_index, Delta = True)   
                
                # update position
                
                omega_new_position = (omega_new_to_alpha + omega_new_to_betha + omega_new_to_delta) / 6
                
                omega_new_position = np.clip(omega_new_position, a_min = self.lower_bound, a_max = self.upper_bound)
                
                omega_new_position = omega_new_position.tolist()[0]
                
                self.wolf_pop["wolf_" + str(wolf_index)] = self.wolf_pop["wolf_" + \
                                                                str(wolf_index)]._replace(position = omega_new_position)
                
                # update fitness
                
                omega_new_fitness = self.UpdateFitness(wolf_index)
                
                self.wolf_pop["wolf_" + str(wolf_index)] = self.wolf_pop["wolf_" + str(wolf_index)]._replace(
                                                                                fitness = omega_new_fitness)
                
                # update wolfs level
                
                self.UpdateWolfsLevel(wolf_index)
            
            self.best_fitness[iteration] = self.alpha_wolf.fitness
            
            print(f"for iteration {iteration} best fitness is {self.best_fitness[iteration]}")
            
        plt.plot(self.best_fitness)
        
        plt.show(block = True)
        
        plt.xlabel("iteration")
        
        plt.ylabel("best_cost")
            
        
          

    def CreateInitPop(self):
        
        """ create a new population at end of each generation

        Args:
        
            no argument
            
        Returns:
        
            no returns --> create new population

        """
            
        # crate wolf_population
        
        for index in range(0, self.number_of_wolfs): 
            
            # initialize parameters woth orthagonal method
            
            omega_wolf_position = torch.zeros([1, self.number_of_parameters])
            
            omega_wolf_position = nn.init.orthogonal_(omega_wolf_position)
            
            omega_wolf_position = omega_wolf_position.numpy()
            
            # clip values to range lower bound and upper bound
            
            omega_wolf_position = np.clip(omega_wolf_position, a_min = self.lower_bound, a_max = self.upper_bound)

            omega_wolf_position = omega_wolf_position.tolist()[0]
            
            # insert method to population
            
            wolf_index  = "wolf_" + str(index)
            
            self.wolf_pop[wolf_index] = self.wolf_pop[wolf_index]._replace(position = omega_wolf_position)
            
        self.CostFunction()
        
        alpha_wolf_index = "wolf_0"
        
        betha_wolf_index = "wolf_1"
        
        delta_wolf_index = "wolf_2"
        
        self.best_fitness[0] = self.wolf_pop[alpha_wolf_index].fitness
        
        self.alpha_wolf = self.alpha_wolf._replace(position = self.wolf_pop[alpha_wolf_index].position,
                                                    fitness = self.wolf_pop[alpha_wolf_index].fitness )
        
        
        self.betha_wolf = self.alpha_wolf._replace(position = self.wolf_pop[betha_wolf_index].position,
                                                    fitness = self.wolf_pop[betha_wolf_index].fitness )
        
        self.delta_wolf = self.alpha_wolf._replace(position = self.wolf_pop[delta_wolf_index].position,
                                                    fitness = self.wolf_pop[delta_wolf_index].fitness )
            
        
            
            
    def CostFunction(self, wolf_position = None):
        
        """ our problem cost function 

        Args:
        
           wolf_position: parameters of a wolf

        Returns:
        
            fitness: fitness of a olf (if parameters not passed calculate fitness of all wolfs and no return)

        """
        
        if wolf_position != None:
            
            fitness = self.Evaluate(wolf_position)
            
            return fitness
        
        else:
            
            for omega_wolf_index, omega_wolf in self.wolf_pop.items():
                
                fitness = self.Evaluate(omega_wolf.position)
                
                self.wolf_pop[omega_wolf_index] = self.wolf_pop[omega_wolf_index]._replace(fitness = fitness)
                
            self.SortPop()
    
    
    def Evaluate(self, wolf_position = None):
        
        """ evaluate one wolf by passing wolf parameters 

        Args:
        
           wolf_position: one wolf parameters

        Returns:
        
            fitness: fitness of targeted genome
        
        """
        
        fitness = sum(list(map(lambda x : x**2, wolf_position)))
        
        return fitness
    
    
    def SortPop(self):
        
                
        """ sort population with respect to their fitness

        Args:
        
            no argument

        Returns:
        
            no return --> sort population with respect to their fitness

        """
        
        fitneses = []

        for wolf_index, wolf in self.wolf_pop.items():

            fitneses.append((wolf.fitness, wolf.position, wolf_index))

        sorted_fitneses = list((sorted(fitneses)))

        for index, (wolf_fitness, wolf_position, past_wolf_index) in enumerate(sorted_fitneses):
            
            wolf_index = "wolf_" + str(index)

            self.wolf_pop[wolf_index] = self.wolf_pop[wolf_index]._replace(position = wolf_position, 
                                                                           fitness = wolf_fitness)
    
    
    def UpdateWolfsLevel(self, wolf_index):
        
                
        """ update levels of alpha and betha and delta wolf

        Args:
        
            wolf index : index of wolf in population that we want to make comparison between this wolf and alpha&betha&delta wolf

        Returns:
        
            no return --> update alpha & betha & delta wolf position and fitness

        """
        

        if self.wolf_pop["wolf_" + str(wolf_index)].fitness < self.delta_wolf.fitness:
            
            self.delta_wolf = self.delta_wolf._replace(position = self.wolf_pop["wolf_" + str(wolf_index)].position,
                                                      fitness = self.wolf_pop["wolf_" + str(wolf_index)].fitness)
            
            if self.wolf_pop["wolf_" + str(wolf_index)].fitness < self.betha_wolf.fitness:
                
                self.betha_wolf = self.betha_wolf._replace(position = self.wolf_pop["wolf_" + str(wolf_index)].position,
                                                      fitness = self.wolf_pop["wolf_" + str(wolf_index)].fitness)
                
                if self.wolf_pop["wolf_" + str(wolf_index)].fitness < self.alpha_wolf.fitness:
                
                    self.alpha_wolf = self.alpha_wolf._replace(position = self.wolf_pop["wolf_" + str(wolf_index)].position,
                                                      fitness = self.wolf_pop["wolf_" + str(wolf_index)].fitness)
        
        
    def UpdatePositions(self, wolf_index, Alpha = False, Betha = False, Delta = False):
        
        """ update positions of wolf in populations (evolutionary part)

        Args:
        
            wolf index : index of wolf in population that we want to upadte it's state
            
            Alpha : True menas update wolf position based of alpha wolf
            
            Betha : True menas update wolf position based of betha wolf
            
            Delta : True menas update wolf position based of delata wolf

        Returns:
        
            omega_new_position--> updated wolf (omega_wolf) position --> omega_wolf is wolf from population that we want update its position

        """
        
        if Alpha:
            
            targeted_wolf = self.alpha_wolf
            
        if Betha:
            
            targeted_wolf = self.betha_wolf
            
        if Delta:
            
            targeted_wolf = self.delta_wolf
        
        # update A and C

        A = self.a  * (2 * np.random.uniform(0, 1, size = (1, self.number_of_parameters)) - 1)

        C = 2 * np.random.uniform(0, 1, size = self.number_of_parameters) 

        # update D

        targeted_wolf_position = np.array(targeted_wolf.position).reshape(1, -1)

        omega_wolf_position  = np.array(self.wolf_pop["wolf_" + str(wolf_index)].position).reshape(1, -1)

        D = np.abs(C * targeted_wolf_position - omega_wolf_position)

        # update omega wolf position wrt to alpha wolf

        omega_new_position = targeted_wolf_position - ( A * D )

        return omega_new_position
        
        
    def UpdateFitness(self, wolf_index):
        
        """ update fitness of wolf in populations (evolutionary part)

        Args:
        
            wolf index : index of wolf in population that we want to calculate it's fitness

        Returns:
        
            fitness --> fitness of wolf

        """
        
        fitness = self.CostFunction(self.wolf_pop["wolf_" + str(wolf_index)].position)
        
        return fitness
    
    
    
if __name__ == "__main__":
    
    # get parameters from terminal
    
    GWO = argparse.ArgumentParser()
    
    
    GWO.add_argument("--parameters_lower_bound", default="-10", help='decision parameters value can not go beyond this value', type = int)
    
    GWO.add_argument("--parameters_upper_bound", default="10", help='decision parameters value can not go beyond this value', type = int)
        
    GWO.add_argument("--num_parameters", default="50",  help='number of decision values (number of genes in genome)', type = int)
        
    GWO.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
    
    GWO.add_argument("--max_iteration", default="500", help='max iteration for evolutionary method', type = int)
  
    
    
    args =  GWO.parse_args()
    

    graywolf = GrayWolf(number_of_wolfs = 50, max_iteration = 100, parameters_lower_bound = -10, parameters_upper_bound = 10, num_parameters = 5)

    graywolf.MainLoop()
    
