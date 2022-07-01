import numpy as np
import pdb
from collections import namedtuple
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class Genetic:
    
    def __init__(self, max_iteration = 1000, continues = True,
                 pop_size = 20, crossover_percentage = 0.8, mutation_percentage = 0.2,
                 tournament_selection_member = 3, mutation_rate = 0.5, parameters_lower_bound = -10,
                 parameters_upper_bound = 10, gamma = 0.1, mu = 0.01, betha = 8, num_parameters = 5):
        
        
        """ genetic algorithm main class

        Args:
            
            parameters_lower_bound: decision parameters value can not go beyond this value
            
            parameters_upper_bound: decision parameters value can not go beyond this value
            
            crossover_percentage: number of crossover-created childtren in terms of percentage of whole population
            
            max_iteration: number of iteration (generation) to run algorithm
            
            mutation_percentage: number of mutation-created childtren in terms of percentage of whole population
            
            tournament_selection_member: number of member set up group in tournemant selection method
            
            mutation_rate: probability of each gen in one genome to be mutated
            
            pop_size: number of genomes or number of population members
            
            num_parameters : number of decision variables
            
            gamma: constant used in UniformCrossoverContinues for generating new gens (decision parameter) through crossover whech define range that is allowed to parameter be
            
            mu: constant used in ContinuesMutation for generating new gens (decision parameter) through mutation whech acts a a standard deviation
            
            betha: selection pressure in rollet wheel method
            
            continues :if problem decision parameters lies in continues space or not (default is yes)

        Returns:
        
            no return --> initializing (create initial population)

        """
        
        
        self.max_iteration = max_iteration
        
        self.pop_size  = pop_size
        
        self.crossover_percentage = crossover_percentage
        
        self.mutation_percentage = mutation_percentage
        
        self.mutation_rate = mutation_rate
        
        self.tournament_selection_member = tournament_selection_member 
        
        self.lower_bound = parameters_lower_bound
        
        self.upper_bound = parameters_upper_bound
        
        self.gamma = gamma
        
        self.mu = mu
        
        self.continues = continues
        
        self.betha = betha
        
        self.genome = namedtuple("Genome", field_names = ["parameters", "fitness"] )
        
        self.genomes = {"genome_" + str(genome_index) : self.genome([], None) for genome_index in range(pop_size)}
                        
        self.crossover_child_number = int(2 * np.round((crossover_percentage * pop_size) / (2)))
                        
        self.mutation_child_number = int(np.round(mutation_percentage * pop_size))
        
        self.crossover_child = namedtuple("CrossoverChilds", field_names = ["parameters", "fitness"] )
        
        self.crossover_children = {"child_" + str(child_index) : self.crossover_child([], None) for child_index in \
                                                                           range(self.crossover_child_number)}
                        
        self.mutation_child = namedtuple("MutationChilds", field_names = ["parameters", "fitness"] )
        
        self.mutation_children = {"child_" + str(child_index) : self.mutation_child([], None) for child_index in \
                                                                           range(self.mutation_child_number)}               
                        
                        
        # defining network settings
        
        self.number_of_parameters = num_parameters
                        
        self.best_fitness = np.zeros([self.max_iteration])
        
        self.best_solutions = np.zeros([self.max_iteration, self.number_of_parameters])
                        
                        
                        
        # create initial population and sort
        
        self.CreateInitPop()
        
        
                        
                        
        # store best fitness of initial generation
        
        initial_generation_index = 0
        
        self.best_fitness[initial_generation_index] = self.genomes["genome_0"].fitness # population is sorted inside CreateInitPop()
        
        self.best_solutions[initial_generation_index] = self.genomes["genome_0"].parameters
                        
      
                        
    def MainLoop(self, continues = True):
        
        """ main loop of (EA --> genetic algorithm

        Args:
        
            no argument

        Returns:
        
            no return --> iterate through generation to reach a solution

        """
                        
        for iteration in range(1, self.max_iteration):
            
            print(iteration)
            
            #crossovers
            
            crossover_child_index = 0
            
            for each_crossover in range(int(self.crossover_child_number / 2)):
                
                
                #get parents
                
                parent_1_parameters, parent_2_parameters = self.GetParetntCrossover()
                
                #get child parameters
                
                child_1_parameters, child_2_parameters = self.Crossover(parent_1_parameters, parent_2_parameters, continues = continues)
                
                
                child_1_parameters = np.clip(child_1_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
                
                child_2_parameters = np.clip(child_2_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
                
                child_1_parameters = child_1_parameters.tolist()
                
                child_2_parameters = child_2_parameters.tolist()
                
                # fiil child population
                
                if continues:
                
                    self.crossover_children["child_" + str(crossover_child_index)] = self.crossover_children[
                                                                "child_" + str(crossover_child_index)]._replace(
                                                                        parameters = child_1_parameters)

                    self.crossover_children["child_" + str(crossover_child_index + 1)] = self.crossover_children[
                                                                "child_" + str(crossover_child_index + 1)]._replace(
                                                                        parameters = child_2_parameters)
                    
                else:
                    
                    self.crossover_children["child_" + str(crossover_child_index)] = self.crossover_children[
                                                                "child_" + str(crossover_child_index)]._replace(
                                                                parameters = child_1_parameters)

                    self.crossover_children["child_" + str(crossover_child_index + 1)] = self.crossover_children[
                                                                "child_" + str(crossover_child_index + 1)]._replace(
                                                                parameters = child_2_parameters)
                    
                #get child fitness
                
                child_1_fitness = self.CostFunction(parameters = child_1_parameters)
                
                child_2_fitness = self.CostFunction(parameters = child_2_parameters)
                                       
                self.crossover_children["child_" + str(crossover_child_index)] = self.crossover_children[
                                                                "child_" + str(crossover_child_index)]._replace(
                                                                fitness = child_1_fitness)
                
                self.crossover_children["child_" + str(crossover_child_index + 1)] = self.crossover_children[
                                                                "child_" + str(crossover_child_index + 1)]._replace(
                                                                fitness = child_2_fitness)
                
                #update child index
                
                crossover_child_index += 2
                
                
            #mutations
                
            mutation_child_index = 0
                
            for each_mutation in range(int(self.mutation_child_number)):
                
                selected_parent_parameters = self.GetParetntMutation(local_search = False)
                
                child_parameters = self.Mutation(selected_parent_parameters, continues = continues)
                
                child_parameters = np.clip(child_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
                
                child_parameters = child_parameters.tolist()
                
                # fiil child population
                
                if continues:
                
                    self.mutation_children["child_" + str(mutation_child_index)] = self.mutation_children[
                                                                        "child_" + str(mutation_child_index)]._replace(
                                                                        parameters = child_parameters)
                    
                else:
                    
                    self.mutation_children["child_" + str(mutation_child_index)] = self.mutation_children[
                                                                        "child_" + str(mutation_child_index)]._replace(
                                                                        parameters = child_parameters)
                    
                #get child fitness
                
                child_fitness = self.CostFunction(parameters = child_parameters)
                
                self.mutation_children["child_" + str(mutation_child_index)] = self.mutation_children[
                                                                        "child_" + str(mutation_child_index)]._replace(
                                                                        fitness = child_fitness)
                
                #update child index
                
                
                mutation_child_index += 1
                
                
            # merge all population (children and previous population)
                
            self.CreateNewPop()
            
            
            #store best fitness
            
            self.best_fitness[iteration] = self.genomes["genome_0"].fitness # population is sorted
            
            self.best_solutions[iteration] = self.genomes["genome_0"].parameters # population is sorted

            
            # print results
            
            print(f"iteration {iteration} best fitness is {self.best_fitness[iteration]}")
            
            
        plt.plot(self.best_fitness)
        
        plt.show(block = True)
        
        plt.xlabel("iteration")
        
        plt.ylabel("best costs")
        
        
    def CreateNewPop(self):
        
        """ create a new population at end of each generation

        Args:
        
            no argument
            
        Returns:
        
            no returns --> create new population

        """
        
        #unpack candidates of new solution archive to new dictionary
        
        candidates = {**self.genomes, **self.crossover_children, **self.mutation_children}
        
        self.genomes = {"genome_" + str(genome_index) : candidate for genome_index, (key, candidate) \
                                                                             in enumerate(candidates.items())}  
        #sort population
        
        self.SortPop()
        
        # make new pop
        
        key_indicies = list(self.genomes.keys())[0 : self.pop_size] 
        
        self.genomes = { "genome_" + str(genome_index) : genome for genome_index, genome in \
                                                        enumerate(map(self.genomes.get, key_indicies)) }
        
        
        
                
        
    def Crossover(self, parent_1_parameters, parent_2_parameters, single_point_prob = 1/3, double_point_prob = 1/3,
                  uniform_prob = 1/3, continues = True):
        
        """ main crossover class

        Args:
        
            parent_1_parameters: first parent parameters
            
            parent_2_parameters: second parent parameters
            
            single_point_prob: crossover method using only one point of division
            
            double_point_prob: crossover method using only two point of division
            
            uniform_prob: crossover method deciding on each gen seperately and using point number same as number of gens
            
            continues: if decision prameters are in continues space or not
            
        Returns:
        
            child_1_parameters: child_1 parameters created from parent parametrs mutation
            
            child_2_parameters: child_2 parameters created from parent parametrs mutation

        """
        
        crossovers = [self.OnePointCrossover, self.DoublePointCrossover, self.UniformCrossoverDiscrete]
        
        if continues:
            
            targeted_crossover = self.UniformCrossoverContinues
            
            child_1_parameters, child_2_parameters = targeted_crossover(parent_1_parameters, parent_2_parameters)
            
            
        else:
            
            probs = [single_point_prob, double_point_prob, uniform_prob]
            
            crossover_method_index = self.RolletWheel(probs)
            
            targeted_crossover = crossovers[crossover_method_index]
            
            child_1_parameters, child_2_parameters = targeted_crossover(parent_1_parameters, parent_2_parameters)
            
        return child_1_parameters, child_2_parameters
    
    
    def Mutation(self, parent_parameters, binary = False, discrete = False, continues = True):
        
        """ main mutation class

        Args:
        
            parent_parameters: paranet parameters
            
            binary: mutation method used for binary problems
            
            discrete: mutation method used for discrete problems
            
            continues: mutation method used for continues problems
        
            
        Returns:
        
            child_parameters: child parameters created from parent parametrs mutation

        """
        
        if binary:
            
            child_parameters = self.BinaryMutation(parent_parameters)
            
        elif discrete:
            
            child_parameters = self.DiscreteMutation(parent_parameters)
            
        elif continues:
            
            child_parameters = self.ContinuesMutation(parent_parameters)
            
        return child_parameters
    
            
    def BinaryMutation(self, parent_parameters):
        
        """ binary mutation for binary problems

        Args:
        
            parent_parameters : parent parameters 

        Returns:
        
            child_parameters --> child parameters created from mutation of parents parametrs

        """
        
        # how many of parametrs (gens in genome) we want to mutate
        
        mutation_vars_count = int( np.ceil( self.mutation_rate * self.number_of_parameters ) )
        
        mutation_vars_index = np.random.choice(np.arange(0, self.number_of_parameters),
                                               size = mutation_vars_count, replace = False)
        
        parent_parameters = np.array(parent_parameters)
        
        child_parameters = parent_parameters
        
        child_parameters[mutation_vars_index] = 1 - parent_parameters[mutation_vars_index] 
        
        child_parameters = child_parameters.tolist()
        
        return child_parameters
    
        
    def DiscreteMutation(self, parent_parameters):
        
        """  mutation for discrete problems

        Args:
        
            parent_parameters : parent parameters 

        Returns:
        
            child_parameters --> child parameters created from mutation of parents parametrs

        """
        
        mutation_vars_count = int( np.ceil( self.mutation_rate * self.number_of_parameters ) )
        
        mutation_vars_index = np.random.choice(np.arange(0, self.number_of_parameters),
                                               size = mutation_vars_count, replace = False) 
        for index in mutation_vars_index:
            
            previous_value = parent_parameters[index]
            
            valid_numbers_to_replace = np.arange(self.lower_bound, self.upper_bound + 1)
            
            remove_index = np.where(valid_numbers_to_replace == previous_value)[0][0]
            
            valid_numbers_to_replace = np.delete(valid_numbers_to_replace, remove_index)
            
            new_value = np.random.choice(valid_numbers_to_replace)
            
            parent_parameters[index] = new_value
            
        child_parameters = parent_parameters
        
        return child_parameters
    
    
        
    def ContinuesMutation(self, parent_parameters):
        
        """  mutation for continues problems

        Args:
        
            parent_parameters : parent parameters 

        Returns:
        
            child_parameters --> child parameters created from mutation of parents parametrs

        """
        
        mutation_vars_count = int( np.ceil( self.mutation_rate * self.number_of_parameters ) )
        
        mutation_vars_index = np.random.choice(np.arange(0, self.number_of_parameters),
                                               size = mutation_vars_count, replace = False) 
        
        parent_parameters = np.array(parent_parameters)
        
        child_parameters = parent_parameters
        
        distribution_sigma = self.mu * (self.upper_bound - self.lower_bound)
        
        child_parameters[mutation_vars_index] = parent_parameters[mutation_vars_index] + \
                                                                    distribution_sigma * np.random.randn(mutation_vars_count)
        
        child_parameters = np.clip(child_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
        
        child_parameters = child_parameters.tolist()
        
        return child_parameters
    
    
            
    def OnePointCrossover(self, parent_1_parameters, parent_2_parameters):
        
        """  single point crossover

        Args:
        
            parent_1_parameters: first parent parameters
            
            parent_2_parameters: second parent parameters 

        Returns:
        
            child_1_parameters: child_1 parameters created from crossover
            
            child_2_parameters: child_2 parameters created from crossover

        """
        
        cross = np.random.randint(1, self.number_of_parameters)
        
        child_1_parameters = parent_1_parameters[0 : cross] + parent_2_parameters[cross : ]
        
        child_2_parameters = parent_2_parameters[0 : cross] + parent_1_parameters[cross : ]
        
        return child_1_parameters, child_2_parameters
            
            
    def DoublePointCrossover(self, parent_1_parameters, parent_2_parameters):
        
        """  double point crossover

        Args:
        
            parent_1_parameters: first parent parameters
            
            parent_2_parameters: second parent parameters 

        Returns:
        
            child_1_parameters: child_1 parameters created from crossover
            
            child_2_parameters: child_2 parameters created from crossover

        """
        
        cross = np.random.choice(np.arange(1, self.number_of_parameters), size = 2, replace = False)
        
        cross_1 = cross[0]
        
        cross_2 = cross[1]
        
        child_1_parameters = parent_1_parameters[ : cross_1] + parent_2_parameters[cross_1 : cross_2] + \
                                        parent_1_parameters[cross_2 :]
        
        child_2_parameters = parent_2_parameters[ : cross_1] + parent_1_parameters[cross_1 : cross_2] + \
                                        parent_2_parameters[cross_2 :]
        
        return child_1_parameters, child_2_parameters
    
    
        
    def UniformCrossoverDiscrete(self, parent_1_parameters, parent_2_parameters):
        
        """  uniform crossover for discrete problems

        Args:
        
            parent_1_parameters: first parent parameters
            
            parent_2_parameters: second parent parameters 

        Returns:
        
            child_1_parameters: child_1 parameters created from crossover
            
            child_2_parameters: child_2 parameters created from crossover

        """
        
        mask = np.random.randint(0, 2, size = self.number_of_parameters)
        
        child_1_parameters =  mask * np.array(parent_1_parameters) + (1- mask) * np.array(parent_2_parameters)
        
        child_2_parameters =  mask * np.array(parent_2_parameters) + (1- mask) * np.array(parent_1_parameters)
        
        return child_1_parameters.tolist(), child_2_parameters.tolist()
    
        
    def UniformCrossoverContinues(self, parent_1_parameters, parent_2_parameters):
        
        """  uniform crossover for continues problems

        Args:
        
            parent_1_parameters: first parent parameters
            
            parent_2_parameters: second parent parameters 

        Returns:
        
            child_1_parameters: child_1 parameters created from crossover
            
            child_2_parameters: child_2 parameters created from crossover

        """
        
        mask = np.random.uniform(- self.gamma, 1 + self.gamma, size = self.number_of_parameters)
        
        child_1_parameters =  mask * np.array(parent_1_parameters) + (1- mask) * np.array(parent_2_parameters)
        
        child_2_parameters =  mask * np.array(parent_2_parameters) + (1- mask) * np.array(parent_1_parameters)
        
        child_1_parameters = np.clip(child_1_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
        
        child_2_parameters = np.clip(child_2_parameters, a_min = self.lower_bound, a_max = self.upper_bound)
        
        return child_1_parameters.tolist(), child_2_parameters.tolist()
    
    
    def GetParetntCrossover(self, random_selection_prob = 4/9, tornament_selection_prob = 4/9, fitness_selection_prob  = 1/9):
        
        """  method for selecting parents for crossover

        Args:
        
            random_selection_prob : probability to use --> randmomly selecting parents for crossover
            
            tornament_selection_prob : probability to use --> tournemant selection method for selecting parents for crossover
            
            fitness_selection_prob : probability to use --> selection parents based on fitness method

        Returns:
        
            parent_1_parameters: parent_1 parameters selected for crossover
            
            parent_2_parameters: parent_2 parameters selected for crossover

        """
        
        candidate_methods = [self.RandomSelection, self.TornamentSelection, self.Fitnessselection ]
        
        probs = [random_selection_prob, tornament_selection_prob, fitness_selection_prob]
        
        selection_method_index = self.RolletWheel(probs)
        
        target_method = candidate_methods[selection_method_index]
        
        parent_1_index, parent_2_index = target_method()
        
        parent_1_parameters = self.genomes["genome_" + str(parent_1_index)].parameters
        
        parent_2_parameters = self.genomes["genome_" + str(parent_2_index)].parameters
        
        return parent_1_parameters, parent_2_parameters
    
    
    def GetParetntMutation(self, local_search = False, random = False):
        
        """  method for selecting parents for crossover

        Args:
        
            random : if true randomly select parent
            
            local_search : if true select best genome as a parent for mutation which this contribute to local search if not select worst genome which is exploration 

        Returns:
        
            selected_parnet_parameters: parent parameters created for mutation
            

        """
        
        if random:
            
            selected_genome_index = np.random.randint(0, self.pop_size)
            
            selected_parnet_parameters = self.genomes[selected_genome_index].parameters
            
        else:
        
            if local_search:

                best_genome_index = list(self.genomes.keys())[0] # population is sorted after each iteration

                selected_parnet_parameters = self.genomes[best_genome_index].parameters

            else:

                worst_genome_index = list(self.genomes.keys())[-1] # population is sorted after each iteration

                selected_parnet_parameters = self.genomes[worst_genome_index].parameters

        return selected_parnet_parameters

    
    def RandomSelection(self):
        
        """  randmomly selecting parents for crossover

        Args:
        
            no argument

        Returns:
        
            parent_1_index: selected parent_1 index in population
            
            parent_2_index: selected parent_2 index in population
            
        """
        
        parent_1_index = np.random.randint(0, self.pop_size)
        
        parent_2_index = np.random.randint(0, self.pop_size)
        
        return parent_1_index, parent_2_index
    
    
        
    def TornamentSelection(self):
        
        """  selection parents based of tournemant selection method

        Args:
        
            no argument

        Returns:
        
            selected_parents_index : selected parent indicies in population
            
        """
        
        selected_parents_index = []
        
        for iteration in range(2): # we need 2 selected parent
         
            candidates = np.random.randint(0, self.pop_size, size = self.tournament_selection_member)

            candidates_fitness = [self.genomes["genome_" + str(candidate_index)].fitness for candidate_index \
                                                                                              in candidates]

            index = np.argmax(candidates_fitness)

            parent_index = candidates[index]
        
            selected_parents_index.append(parent_index)
        
        return selected_parents_index
    
    
        
    def Fitnessselection(self):
        
                
        """  selection parents based of fitness criteria

        Args:
        
            no argument

        Returns:
        
            selected_parents_index : selected parent index in population
            
        """
        
        fitness_array = np.array([genome.fitness for genome_index, genome in self.genomes.items()])
        
        max_fitness_index = np.argmax(fitness_array)
        
        weights = np.exp( - 1 * self.betha * ( (1 / fitness_array) * fitness_array[max_fitness_index] ) )  
        
        probs = weights / np.sum(weights)
        
        selected_parents_index = []
        
        for iteration in range(2): # we need 2 selected parent
            
            parent_index = self.RolletWheel(probs)
            
            selected_parents_index.append(parent_index)
            
        return selected_parents_index
    
    
            
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

    
    def CostFunction(self, parameters = None):
        
        """ our problem cost function 

        Args:
        
            parameters: parameters of a genome

        Returns:
        
            fitness: fitness of genome (if parameters not passed calculate fitness of all genome and no return)

        """
        
        if parameters != None:
            
            fitness = self.Evaluate(parameters)
            
            return fitness
        
        else:
            
            for genome_index, genome in self.genomes.items():
                
                fitness = self.Evaluate(genome.parameters)
                
                self.genomes[genome_index] = self.genomes[genome_index]._replace(fitness = fitness)
                        
            self.SortPop()  
           
                        
    def CreateInitPop(self):
        
        """ create initial population and evaluate thier fitness

        Args:
        
            no argument
            
        Returns:
        
            no return --> create initial population to start algorithm and evaluate all members in population

        """
            
        # crate wolf_population
        
        for index in range(0, self.pop_size): 
            
            # initialize parameters woth orthagonal method
            
            target_genome_params = torch.zeros([1, self.number_of_parameters])
            
            target_genome_params = nn.init.uniform_(target_genome_params)
            
            target_genome_params = target_genome_params.numpy()
            
            # clip values to range lower bound and upper bound
            
            target_genome_params = np.clip(target_genome_params, a_min = self.lower_bound, a_max = self.upper_bound)

            target_genome_params = target_genome_params.tolist()[0]
            
            # insert method to population
            
            genome_index  = "genome_" + str(index)
            
            self.genomes[genome_index] = self.genomes[genome_index]._replace(parameters = target_genome_params)
            
        self.CostFunction()
                        
                        
    def Evaluate(self, genome_parameters):
        
        """ evaluate one genome by passing genome parameters 

        Args:
        
           genome_parameters: one genome parameters

        Returns:
        
            fitness: fitness of targeted genome
        
        """
        
        fitness = sum(list(map(lambda x : x**2, genome_parameters)))
        
        return fitness
                        
                        
    def SortPop(self):
        
        """ sort population with respect to their fitness

        Args:
        
            no argument

        Returns:
        
            no return --> sort population with respect to their fitness

        """
        
        fitneses = []
        
        #pdb.set_trace()

        for genome_index, genome in self.genomes.items():

            fitneses.append((genome.fitness, genome.parameters, genome_index))

        sorted_fitneses = list((sorted(fitneses)))

        for index, (genome_fitness, genome_parameters, past_genome_index) in enumerate(sorted_fitneses):
            
            genome_index = "genome_" + str(index)

            self.genomes[genome_index] = self.genomes[genome_index]._replace(parameters = genome_parameters, 
                                                                fitness = genome_fitness)
            
            
if __name__ == "__main__":
    
    # get parameters from terminal
    
    Genetic_parser = argparse.ArgumentParser()
    
    
    Genetic_parser.add_argument("--parameters_lower_bound", default="-10", help='decision parameters value can not go beyond this value', type = float)
    
    Genetic_parser.add_argument("--parameters_upper_bound", default="10", help='decision parameters value can not go beyond this value', type = float)
        
    Genetic_parser.add_argument("--num_parameters", default="5",  help='number of decision values (number of genes in genome)', type = int)
    
    Genetic_parser.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
    
    Genetic_parser.add_argument("--max_iteration", default="1000", help='max iteration for evolutionary method', type = int)
    
    Genetic_parser.add_argument("--crossover_percentage", default=".8",  help='number of crossover-created childtren in terms of percentage of whole population', type = float)
        
    Genetic_parser.add_argument("--mutation_percentage", default=".2",  help='number of mutation-created childtren in terms of percentage of whole population', type = float)
    
    Genetic_parser.add_argument("--tournament_selection_member", default="3",  help="number of member set up group in tournemant selection method", type = int)
    
    Genetic_parser.add_argument("--mutation_rate", default="0.5", help='probability of each gen in one genome to be mutated', type = float)
        
    Genetic_parser.add_argument("--gamma", default="0.1",  help='constant used in UniformCrossoverContinues for generating new gens (decision parameter) through crossover whech define range that is allowed to parameter be', type = float)
    
    Genetic_parser.add_argument("--mu", default="0.01",  help="constant used in ContinuesMutation for generating new gens (decision parameter) through mutation whech acts a a standard deviation", type = float)
    
    Genetic_parser.add_argument("--betha", default="8", help='selection pressure in rollet wheel method', type = float)
    
    Genetic_parser.add_argument("--continues", action="store_false", help='if problem decision parameters lies in continues space or not (default is yes)')
    
    
    args = Genetic_parser.parse_args()

    genetic = Genetic(max_iteration = args.max_iteration, pop_size = args.pop_size, crossover_percentage = args.crossover_percentage,
                 mutation_percentage = args.mutation_percentage, tournament_selection_member = args.tournament_selection_member,
                 mutation_rate = args.mutation_rate, parameters_lower_bound = args.parameters_lower_bound,
                 parameters_upper_bound = args.parameters_upper_bound, gamma = args.gamma, mu = args.mu, betha = args.betha, num_parameters = args.num_parameters)

    genetic.MainLoop(args.continues)

