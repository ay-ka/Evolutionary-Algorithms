from torch import nn
import torch
import numpy as np
from collections import namedtuple
import pdb
import math
import matplotlib.pyplot as plt
import argparse



class PSO:
    
    def __init__(self, max_iteration = 500, num_parameters = 5,
                pop_size = 20, parameters_lower_bound = -10, parameters_upper_bound = 10, alpha = 0.1,
                inertia_damp = 0.99, phi_1 = 2.05, phi_2 = 2.05):
        
        """ pso algorithm main class

        Args:

            parameters_lower_bound: minimum value parameters (gens) can take inside ants (genome)
            
            parameters_upper_bound: maximum value parameters (gens) can take inside ants (genome)
            
            num_parameters : number of gens each genome have (decision variables)
            
            pop_size: number of particles or number of population members
            
            phi_1 : parameter for calculation of velocities inertia as well as personal learning and global learnbing weights
            
            phi_2 : parameter for calculation of velocities inertia as well as personal learning and global learnbing weights
            
            max_iteration: number of iteration (generation) to run algorithm
            
            alpha: constant contributing in calculating all particles maximum and minimum velocities
            
            inertia_damp: decaying factor for inertia weight through generations

        Returns:
        
            no return --> initializing (create initial particle population also store current best particle)

        """
        
        self.max_iteration = max_iteration
        
        self.pop_size  = pop_size 
        
        self.alpha = alpha
        
        self.lower_bound = parameters_lower_bound
        
        self.upper_bound = parameters_upper_bound
        
        self.velocity_upper_bound = self.alpha * (self.upper_bound - self.lower_bound)
        
        self.velocity_lower_bound = -1 * self.velocity_upper_bound
        
        self.phi_one = phi_1 # there is paper specifing this value is best wrt to dynamic relations 
        # --> (used in calculating inertia and learning_rates)
        
        self.phi_two = phi_2 # there is paper specifing this value is best wrt to dynamic relations
        # --> (used in calculating inertia and learning_rates)
        
        self.phi = self.phi_one + self.phi_two
        
        self.chi = (2 / (self.phi - 2 * np.sqrt(self.phi ** 2 - 4 * self.phi)))
        
        self.inertia_weight = self.chi
        
        self.personal_learning = self.chi * self.phi_one
        
        self.global_learning = self.chi * self.phi_two
        
        self.inertia_damp = inertia_damp
        
        self.particle = namedtuple("Particles", field_names = ["parameters", "fitness", "velocity",
                                                             "best_fitness", "best_parameters"] )
        
        self.particles = {"particle_" + str(particle_index) : self.particle([], None, [], (-1 * math.inf), [])                                                                 for particle_index in range(self.pop_size)}  
        
        self.best_particle_constructor = namedtuple("Best_Particles", field_names = ["parameters", "fitness"] )
        
        self.best_particle = self.best_particle_constructor([], (-1 * math.inf))
        
        self.best_costs = np.zeros(self.max_iteration)
        
        self.nfe_counter = 0
        
        self.NFE = np.zeros(self.max_iteration)
        
        
        self.number_of_parameters = num_parameters
        
        
        
        # create initial population and sort
        
        self.CreateInitPop()
        
        
        
    def MainLoop(self):
        
        """ main loop of (EA --> PSO for Continues Problems) algorithm

        Args:
        
            no argument

        Returns:
        
            no return --> iterate through generation to reach a solution

        """
        
        for iteration in range(1, self.max_iteration):
            
            
            for particle_index, particle in self.particles.items():
                
                new_parameters, new_fitness, new_velocity = self.UpdateParameters(particle)
                
                self.particles[particle_index] = self.particles[particle_index]._replace(fitness = new_fitness,
                                                                                        parameters = new_parameters,
                                                                                        velocity = new_velocity)
                
                if self.particles[particle_index].fitness < self.particles[particle_index].best_fitness:
                    
                    self.particles[particle_index] = self.particles[particle_index]._replace(
                                                                                    best_fitness = new_fitness,
                                                                                    best_parameters = new_parameters,
                                                                                        )
                    
                    if self.particles[particle_index].best_fitness < self.best_particle.fitness:
                        
                        self.best_particle = self.best_particle._replace(
                                                        fitness = self.particles[particle_index].best_fitness,
                                                        parameters = self.particles[particle_index].best_parameters)
                        
            self.best_costs[iteration] = self.best_particle.fitness
            
            self.NFE[iteration] = self.nfe_counter - self.pop_size
            
            self.inertia_weight = self.inertia_weight * self.inertia_damp
            
            print(f"iteration {iteration}, NFE is {self.NFE[iteration]}, fitness is {self.best_costs[iteration]}")
                
        plt.plot(self.best_costs)
        
        plt.show(block = True)
        
        plt.xlabel("iteration")
        
        plt.ylabel("cost")
    

    def UpdateParameters(self, particle):
        
        """ update position, velocity and fitness of spesific particle

        Args:
        
            particle: (named_tuple) containing poisition, velocity and that particle's memory of best position and 
                        best_fitness experienced

        Returns:
        
            new_parameters: new position of particle updated wrt to equation of paper 
            
            new_velocity: new velocity of particle updated wrt to equation of paper
            
            new_fitness: new fitness of particle updated

        """
        
        first_term =  self.inertia_weight * np.array(particle.velocity) 


        second_term = self.personal_learning * np.random.uniform(size = self.number_of_parameters) *\
                                                             (np.array(particle.best_parameters) - np.array(particle.parameters))

        third_term = self.global_learning * np.random.uniform(size = self.number_of_parameters) *\
                                                             (np.array(self.best_particle.parameters) - np.array(particle.parameters))

        new_velocity = first_term + second_term + third_term
                
        new_velocity = np.clip(new_velocity, a_min = self.velocity_lower_bound,
                                                     a_max = self.velocity_upper_bound)

        new_parameters = np.array(particle.parameters) + new_velocity
        
        #velocity mirror effect
        
        valid_indicies = (new_parameters < self.lower_bound) + (new_parameters > self.upper_bound)
        
        new_velocity[valid_indicies] = -1 * new_velocity[valid_indicies]
        
        new_parameters = np.clip(new_parameters, a_min = self.lower_bound, a_max = self.upper_bound)

        new_parameters = new_parameters.tolist()

        new_velocity = new_velocity.tolist()

        new_fitness = self.CostFunction(parameters = new_parameters)
        
        return new_parameters, new_fitness, new_velocity    
    


    
    
    def CostFunction(self, parameters = []):
        
        """ our problem cost function for PSO algorithm

        Args:
        
            pso_parameters: list of float number which represent an particle (can be used to craete nueral network) -->
                            if is not passed calculate all particles fitness inside main particles population

        Returns:
        
            fitness: fitness of nueral network creatd from particle_parameters inputed as argument 
            (if particle_parameters not passed, there is no return and just calculate all particles fitness
            inside main particles population)

        """
        
        if len(parameters) != 0:
            
            fitness = self.Evaluate(parameters)
            
            self.nfe_counter += 1
            
            return fitness
        
        else:
            
            for particle_index, particle in self.particles.items():
                
                fitness = self.Evaluate(particle.parameters)
                
                self.particles[particle_index] = self.particles[particle_index]._replace(fitness = fitness)
                
                self.particles[particle_index] = self.particles[particle_index]._replace(best_fitness = fitness)
                
                if self.particles[particle_index].best_fitness > self.best_particle.fitness:
                    
                    self.best_particle = self.best_particle._replace(fitness = 
                                                                     self.particles[particle_index].best_fitness,
                                                                     parameters = 
                                                                     self.particles[particle_index].best_parameters)
                    
                    
                        
            self.SortPop() 
            
            self.nfe_counter += self.pop_size

            
            
                            
    def CreateInitPop(self):
        
        """ create initial population and evaluate thier fitness

        Args:
        
            network_model: nueral network model (from reinforcement part --> actor or critic) (pytorch model)

        Returns:
        
            no return --> create initial population to start algorithm and evaluate all members in population

        """
            
        # crate wolf_population
        
        for index in range(0, self.pop_size): 
            
            # initialize parameters woth orthagonal method
            
            target_particle_params = torch.zeros([1, self.number_of_parameters])
            
            target_particle_params = nn.init.uniform_(target_particle_params)
            
            target_particle_params = target_particle_params.numpy()
            
            # clip values to range lower bound and upper bound
            
            target_particle_params = np.clip(target_particle_params, a_min = self.lower_bound, a_max = self.upper_bound)

            target_particle_params = target_particle_params.tolist()[0]
             
            # insert method to population
            
            particle_index  = "particle_" + str(index)
            
            self.particles[particle_index] = self.particles[particle_index]._replace(parameters = target_particle_params)
            
            self.particles[particle_index] = self.particles[particle_index]._replace(velocity =
                                                                        np.zeros(self.number_of_parameters).tolist())
            
            self.particles[particle_index] = self.particles[particle_index]._replace(best_parameters = target_particle_params)
            
        self.CostFunction()
        
        # store best cost 
        
        initial_iteration_index = 0
        
        self.best_costs[initial_iteration_index] = self.best_particle.fitness
        
        
        
    def Evaluate(self, particle_parametrs):
        
        """ evaluate one particle by passing particle (parameters inside it as weights) to reinforcement algorithm

        Args:
        
            particle_network_model: nueral network created from spesific particle (parameters inside it)
                                    (pytorch model) --> if Not given, particle_network_model is set to PSO class own 
                                    network which is initialized by spesific particle parameters

        Returns:
        
            fitness: fitness of targeted particle

        """
        fitness = sum(list(map(lambda x : x**2, particle_parametrs)))
        
        return fitness
                        
                        
    def SortPop(self):
        
        """ sort population with respect to their fitness

        Args:
        
            no argument

        Returns:
        
            no return --> sort population with respect to their fitness

        """
        
        fitneses = []

        for particle_index, particle in self.particles.items():

            fitneses.append((particle.fitness, particle.parameters, particle.velocity, particle.best_fitness, 
                             particle.best_parameters, particle_index))

        sorted_fitneses = list((sorted(fitneses)))

        for index, (particle_fitness, particle_parameters, particle_velocity, particle_best_fitness, 
                    particle_best_parameters, particle_past_index) in enumerate(sorted_fitneses):
            
            particle_index = "particle_" + str(index)

            self.particles[particle_index] = self.particles[particle_index]._replace(parameters = particle_parameters, 
                                                                fitness = particle_fitness, velocity = particle_velocity,
                                                                best_fitness = particle_best_fitness, 
                                                                best_parameters = particle_best_parameters                     )
            

        

if __name__ == "__main__":
    
    # get parameters from terminal
    
    PSO_Parser = argparse.ArgumentParser()
    
    
    PSO_Parser.add_argument("--parameters_lower_bound", default="-10", help='decision parameters value can not go beyond this value', type = int)
    
    PSO_Parser.add_argument("--parameters_upper_bound", default="10", help='decision parameters value can not go beyond this value', type = int)
        
    PSO_Parser.add_argument("--num_parameters", default="5",  help='number of decision values (number of genes in genome)', type = int)
    
    PSO_Parser.add_argument("--alpha", default="0.1",  help='constant contributing in calculating all particles maximum and minimum velocities', type = float)
        
    PSO_Parser.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
    
    PSO_Parser.add_argument("--phi_1", default="2.05",  help="constant for calculation of velocities inertia as well as personal learning and global learnbing weights for updating a partckle position", type = float)
        
    PSO_Parser.add_argument("--phi_2", default="2.05",  help='constant for calculation of velocities inertia as well as personal learning and global learnbing weights for updating a partckle position', type = float)
    
    PSO_Parser.add_argument("--max_iteration", default="500",  help="number of best members of population to choose next generation from those (refer to acor paper)", type = int)
    
    PSO_Parser.add_argument("--inertia_damp", default="0.99",  help="decaying factor for velocity inertia in partickle", type = float)
    
    
    args = PSO_Parser.parse_args()
    

    pso = PSO(parameters_lower_bound = args.parameters_lower_bound, num_parameters = args.num_parameters,
                 alpha = args.alpha, max_iteration = args.max_iteration, inertia_damp = args.inertia_damp,
                 parameters_upper_bound = args.parameters_upper_bound, phi_1 = args.phi_1, phi_2 = args.phi_2, pop_size = args.pop_size)

    pso.MainLoop()

