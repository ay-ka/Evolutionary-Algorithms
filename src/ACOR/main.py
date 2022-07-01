import argparse
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import pdb
import matplotlib
matplotlib.pyplot.show()


class ANT:
    
    def __init__(self, parameters_lower_bound = -10, num_parameters = 50,
                 zeta = 1, max_iteration = 1000, num_solution_archive = 20,
                 parameters_upper_bound = 10, num_child = 100, q = .5, num_ants = 20):
        
        """ ant clony algorithm main class

        Args:
            
            parameters_lower_bound: minimum value parameters (gens) can take inside ants (genome)
            
            parameters_upper_bound: maximum value parameters (gens) can take inside ants (genome)
            
            zeta: float constant defining exploration and exploitation rate --> more zeta means more exploration
            
            max_iteration: number of iteration (generation) to run algorithm
            
            num_solution_archive: number of ants allowed to be in solution archive population (n best members)
            
            num_child: number of children at each generation allowed to create
            
            q: selection pressure in calculating weights and subsuquently affect selecting each solution (refer to papers)
            
            num_ants: number of ants or number of population members
            
            num_parameters : number of decision variables

        Returns:
        
            no return --> initializing (create initial ant population and solution archive ANT population;
                                        also store current best ant)

        """
        
        # ACOR parameters
        
        self.num_ants  = num_ants
        
        self.max_iteration =  max_iteration
        
        self.upper_bound = parameters_upper_bound
        
        self.lower_bound = parameters_lower_bound
        
        self.num_solution_archive = num_solution_archive
        
        self.number_of_parameters = num_parameters
        
        self.num_child = num_child
        
        self.q = q  # selection pressure when calculating weights --> refer to papers
        
        self.zeta = zeta # importance of exploration in calculating standard deviation (then variance)
        
        # create populations
        
        self.ant = namedtuple("ANT", field_names = ["parameters", "fitness"])
        
        self.ants = {"ant_" + str(ant_index) : self.ant([], None) for ant_index in range(self.num_ants)}
        
        self.solution = namedtuple("Solution_Archive", field_names = ["parameters", "fitness"])
        
        self.solutions = {"solution_" + str(solution_index) : self.solution([], None) for solution_index in                                                                                   range(self.num_solution_archive)}
        
        self.child = namedtuple("Children", field_names = ["parameters", "fitness"])
        
        self.children = {"child_" + str(child_index) : self.child([], None) for child_index in range(self.num_child)}
        
        # define arrays of best fitness
        
        self.best_fitness = np.zeros([self.max_iteration])
        
        
        
        # create initial population 
        
        self.CreateInitPop()
        
        
        # create solution archive
        
        self.CreateSolutionArchive()
        
        # store best fitness of initial generation
        
        initial_generation_index = 0
        
        self.best_fitness[initial_generation_index] = self.ants["ant_0"].fitness # population is sorted inside CreateSolutionArchive()
    
    
    
    def MainLoop(self):
        
        """ main loop of (EA --> Ant Colony Optimization for Continues Problems) algorithm

        Args:
        
            no argument

        Returns:
        
            no return --> iterate through generation to reach a solution

        """
        
        weights = self.CalculateWeights()
            
        probs = self.CalculatesProbs(weights)
        
        for iteration in range(1, self.max_iteration):
            
            mu_array, sigma_array = self.CalculateDistributionParam()
            
            for child_index in range(self.num_child):
                
                #create new parameters
                
                for parameter_index in range(self.number_of_parameters):
                    
                    selected_solution_index = self.RolletWheel(probs)
                    
                    new_parameter = self.SampleNewParameter(selected_solution_index, parameter_index, mu_array,
                                                            sigma_array)
                    
                    self.children["child_" + str(child_index)].parameters.append(new_parameter)
                    
                # random picking activation function 
                
                child_parameters = np.clip(self.children["child_" + str(child_index)].parameters, 
                                         a_min = self.lower_bound, a_max = self.upper_bound)
                
                child_parameters = child_parameters.tolist()
                
                self.children["child_" + str(child_index)] = self.children["child_" + str(child_index)]._replace(
                                                                                        parameters = child_parameters)
                
                #calculate cost
                    
                child_fitness = self.CostFunction(self.children["child_" + str(child_index)].parameters) 
                
                
                self.children["child_" + str(child_index)] = self.children["child_" + str(child_index)]._replace(
                                                                                                fitness = child_fitness)

                
            self.CreateNewPop()
            
            self.CreateSolutionArchive()
            
            self.best_fitness[iteration] = self.ants["ant_0"].fitness # --> because in CreateSolutionArchive() <self.ants> is sorted
            
            self.children = {"child_" + str(child_index) : self.child([], None) for child_index in range(self.num_child)}
            
            self.zeta = self.zeta - ( 1 / (0.6 * self.max_iteration) )
            
            print(f"iteration {iteration} best_fitness is {self.best_fitness[iteration]}")
            

        plt.plot(self.best_fitness)
        
        plt.show(block = True)
        
        plt.xlabel("iteration")
        
        plt.ylabel("best_cost")
            
            
            
    def SampleNewParameter(self, solution_index, parameter_index, mu_array, sigma_array):
        
        """ with respect to sampled ant as parent, select new parameter from mean distribution created from mean and sigma
            of parent

        Args:
        
            solution_index: selected ants as parent index
            
            parameter_index: targeted parameter (weight) index to create new one
            
            mu_array: (numpy.array) (number of targeted ants x number of parameters) ecah parameter in each ant considered 
                      as mean 
            
            sigma_array: (numpy.array) (number of targeted ants x number of parameters) ecah parameter in each ant need
                        a sigma which is calculated by mean distance of ecah parameter from other parameters

        Returns:
        
            new_parameter: a float number sampled from mean distribution created by selected parent mean and sigma

        """
        
        new_parameter = mu_array[solution_index][parameter_index] + (sigma_array[solution_index][parameter_index] * np.random.randn())
        
        return new_parameter
        
    
    
    
    def CalculateWeights(self):
        
        """ calculate weights for targeted ants (solution archive ants in here) to get probabilties of each ant to 
            be selected as parent to children with rspect to ants fitness

        Args:
        
            no argument

        Returns:
        
            weights: weights calculated for targeted ants (solution archive ants in this problem) to get 
            probabilties of each ant to be selected as parent to children with rspect to ants fitness

        """
        
        l = np.arange(1, self.num_solution_archive + 1)
        
        first_term = (1 / (self.q * self.num_solution_archive) * np.sqrt( 2 * np.pi ) ) 
        
        second_term = np.exp( (-0.5) * ( np.square( (l-1) / (self.q * self.num_solution_archive) ) ) )
        
        weights = first_term *  second_term
        
        return weights
    
    
    def CalculatesProbs(self, weights):
        
        """ calculate probabilities with respect to targeted ants weight (solution archive ants in this problem)

        Args:
        
            weights: weights calculated for targeted ants (solution archive ants in here) to get probabilties of each ant to 
            be selected as parent to children with rspect to ants fitness

        Returns:
        
            probs: probabilty of selecting each ant as a potensila parent for creating next genaration
            with considering targeted ants weights 

        """
        
        probs = weights / sum(weights)
        
        return probs
    
    
    def CalculateDistributionParam(self):
        
        """ calculate means and variance for current population to cretae next generation population from 

        Args:
        
            no argument

        Returns:
        
            mu_array: (numpy.array) (number of targeted ants x number of parameters) ecah parameter in each ant considered 
                      as mean 
            
            sigma_array: (numpy.array) (number of targeted ants x number of parameters) ecah parameter in each ant need
                        a sigma which is calculated by mean distance of ecah parameter from other parameters

        """
        
        mu_array = np.array([solution.parameters for solution_index, solution in self.solutions.items()])
        
        sigma_array = np.zeros([self.num_solution_archive, self.number_of_parameters])
        
        for target_ant_index in range(self.num_solution_archive):
                
            sigma_array[target_ant_index] = np.sum(np.abs(mu_array[target_ant_index] - mu_array), axis = 0)
    
            sigma_array[target_ant_index] =  ((self.zeta) * (sigma_array[target_ant_index])) / (self.num_solution_archive - 1)
        
        return mu_array, sigma_array
            
    
    def CostFunction(self, ant_parameters = []):
        
        """ our problem cost function for ant colony algorithm

        Args:
        
            ant_parameters: list of float number which represent an ant (can be used to craete nueral network) -->
                            if is not passed calculate all ants fitness inside main ant population

        Returns:
        
            fitness: fitness from ant_parameters inputed as argument  (if ant_parameters not 
                     passed there is no return and just calculate all ants fitness inside main ant population)

        """
        
        if len(ant_parameters) != 0:
            
            fitness = self.Evaluate(ant_parameters)
            
            return fitness
        
        else:
            
            for ant_index, ant in self.ants.items():
                
                fitness = self.Evaluate(ant.parameters)
                
                self.ants[ant_index] = self.ants[ant_index]._replace(fitness = fitness)
        
        
    def Evaluate(self, ant_parameters = None):
        
        """ evaluate one ants by passing ant (parameters inside it as weights) to reinforcement algorithm

        Args:
        
            ant_network_model: nueral network created from spesific ant (parameters inside it) (pytorch model) --> 
                               if Not given, ant_network_model is set to Ant class own network which is initialized 
                               by spesific ant parameters

        Returns:
        
            fitness: fitness of targeted ant 

        """
        
        fitness = sum(list(map(lambda x : x**2, ant_parameters)))
        
        return fitness
        
        
        
    def SortPop(self):
        
        """ sort population with respect to their fitness

        Args:
        
            no argument

        Returns:
        
            no return --> sort population with respect to their fitness

        """
        
        fitneses = []

        for ant_index, ant in self.ants.items():

            fitneses.append((ant.fitness, ant.parameters, ant_index))

        sorted_fitneses = list((sorted(fitneses)))

        for index, (ant_fitness, ant_parameters, past_ant_index) in enumerate(sorted_fitneses):
            
            ant_index = "ant_" + str(index)

            self.ants[ant_index] = self.ants[ant_index]._replace(parameters = ant_parameters, 
                                                                fitness = ant_fitness)
            
            
    def CreateInitPop(self):
        
        """ create initial population and evaluate thier fitness

        Args:
        
            network_model: nueral network model (from reinforcement part --> actor or critic) (pytorch model)

        Returns:
        
            no return --> create initial population to start algorithm and evaluate all members in population

        """
            
        # crate wolf_population
        
        for index in range(0, self.num_ants): # (.... - 1) is for rl_wolf
            
            # initialize parameters woth uniform distribution
            
            ant_parameters = torch.zeros([self.number_of_parameters])
            
            #ant_parameters = nn.init.orthogonal_(ant_parameters)
            
            ant_parameters = nn.init.uniform_(ant_parameters)
            
            ant_parameters = ant_parameters.numpy()
            
            # clip values to range lower bound and upper bound
            
            ant_parameters = np.clip(ant_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
            
            # insert method to population
            
            ant_index  = "ant_" + str(index)
            
            self.ants[ant_index] = self.ants[ant_index]._replace(parameters = ant_parameters.tolist())
            
        self.CostFunction()
        
        
    def CreateSolutionArchive(self):
        
        """ create solution archive population from n best ant 

        Args:
        
            no argument

        Returns:
        
            no return --> sort population and create solution archive population from allowed n best ant

        """
        
        self.SortPop()
        
        for index in range(self.num_solution_archive):
            
            solution_index = "solution_" + str(index)
            
            ant_index = "ant_" + str(index)
            
            self.solutions[solution_index] = self.ants[ant_index] 
            
        
    
    def CreateNewPop(self):
        
        """ after doing needed operation in current generation on memebers, create new population for next generation

        Args:
        
            no argument

        Returns:
        
            no return --> create new population of ants with (children of this generation and 
                          best solution of previous genberation)

        """
        
        #unpack candidates of new solution archive to new dictionary
        
        candidates = {**self.solutions, **self.children}
        
        self.ants = {"ant_" + str(ant_index) : candidate for ant_index, (key, candidate) in enumerate(candidates.items())}
    
    def RolletWheel(self, probs):
        
        """ rolletwheel method for sampling from discrete distribution

        Args:
        
            probs: discrete distribution probabilioties
            
        Returns:
        
            selected_index: sampled instance from discrete distribution with respect to inputed probabilties

        """
        
        cumsum = 0
        
        cumsum_lst = []
        
        for target_index in range(1, len(probs) + 1):
            
            for index in range(0, target_index):
                
                cumsum = cumsum + probs[index]
                
            cumsum_lst.append(cumsum)
            
            cumsum = 0
            
        #select
            
        random_number = np.random.uniform(size = 1)
        
        list_of_valid = np.where(random_number < cumsum_lst)
        
        first_valid = 0
        
        selected_index = list_of_valid[first_valid][first_valid]
            
        return selected_index
        
        

if __name__ == "__main__":
    
    # get parameters from terminal
    
    ACOR = argparse.ArgumentParser()
    
    
    ACOR.add_argument("--parameters_lower_bound", default="-10", help='decision parameters value can not go beyond this value', type = int)
    
    ACOR.add_argument("--parameters_upper_bound", default="10", help='decision parameters value can not go beyond this value', type = int)
        
    ACOR.add_argument("--num_parameters", default="50",  help='number of decision values (number of genes in genome)', type = int)
    
    ACOR.add_argument("--zeta", default="1",  help='float constant defining exploration and exploitation rate ', type = float)
        
    ACOR.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
    
    ACOR.add_argument("--num_solution_archive", default="20",  help="number of best members of population to choose next generation from those (refer to acor paper)", type = int)
    
    ACOR.add_argument("--max_iteration", default="500", help='max iteration for evolutionary method', type = int)
        
    ACOR.add_argument("--num_child", default="100",  help='number of children created in each generation from best solutions (solution archive)', type = int)
    
    ACOR.add_argument("--q", default="0.5",  help="selection pressure in calculating weights and subsuquently affect selecting each solution (refer to papers)", type = float)
    
    
    args = ACOR.parse_args()
    

    ant_class = ANT(parameters_lower_bound = args.parameters_lower_bound, num_parameters = args.num_parameters,
                 zeta = args.zeta, max_iteration = args.max_iteration, num_solution_archive = args.num_solution_archive,
                 parameters_upper_bound = args.parameters_upper_bound, num_child = args.num_child, q = args.q, num_ants = args.pop_size)

    ant_class.MainLoop()