import numpy as np
from collections import namedtuple
import pdb
import math
import matplotlib.pyplot as plt
import copy
import argparse


class Ant:
    
    def __init__(self, evaporation_rate, alpha, betha, benchmark,
                     max_iteration = 600, number_of_ants = 40, Q = 1):
        
        """ ant clony algorithm main class

        Args:
            
            evaporation_rate: rate in which phromones are evaporated
            
            alpha: weight given to ants experience and useful in calculating probability of choosing next city from current one (refer paper)
            
            betha: 'weight given to our knowledge of problem and useful in calculating probability of choosing next city from current one (refer paper)'
            
            max_iteration: number of iteration (generation) to run algorithm
            
            benchmark: problem (benchmark) we want to apply ACO on that
            
            number_of_ants: number of ants or number of population members
            
            Q : constant useful in calculating ants initial phromones

        Returns:
        
            no return --> initializing (create initial ant population and solution archive ANT population;
                                        also store current best ant)

        """
        
        self.TSP = benchmark
        
        self.max_iteration = max_iteration
        
        self.number_of_ants = number_of_ants
        
        self.evaporation_rate = evaporation_rate
        
        self.num_var = self.TSP.num_cities
        
        self.alpha = alpha
        
        self.betha = betha
        
        self.Q = Q
        
        self.best_costs = np.zeros([self.max_iteration])
        
        self.init_tau = 5 * self.Q / (self.num_var * np.mean(self.TSP.distances))
        
        self.ant = namedtuple("Ant", field_names = ["path", "cost"])
        
        self.eta = []
        
        for city_index in range(self.TSP.distances.shape[0]):
            
            self.eta.append(list(map(self.getEta, self.TSP.distances[city_index].tolist())))
        
        self.eta = np.array(self.eta)
        
        self.tau = self.init_tau * np.ones([self.num_var, self.num_var])
        
        self.ants = {"ant_" + str(ant_index) : self.ant([], None) for ant_index in range(self.num_var)}
        
        self.best_ant = self.ant([], math.inf)
        
    
    def MainLoop(self):
        
        """ main loop of (EA --> Ant Colony Optimization for discrete Problems) algorithm

        Args:
        
            no argument

        Returns:
        
            no return --> iterate through generation to reach a solution

        """
        
        for iteration in range(self.max_iteration):
            
            for ant_index in self.ants:
                
                start_city = np.random.randint(1, self.num_var + 1)
                
                self.ants[ant_index].path.append(start_city)
                
                for loop_index in range(2, self.num_var + 1):
                
                    current_city = self.ants[ant_index].path[-1]
                    
                    probs = (self.tau[current_city - 1] ** self.alpha) * (self.eta[current_city - 1] ** self.betha)
                    
                    visited_cities = self.ants[ant_index].path
                    
                    visited_cities = [vis_city_index - one for vis_city_index, one in zip(visited_cities, 
                                                                                    [1] * len(visited_cities))]
                    
                    probs[visited_cities] = 0
                    
                    norm_probs = probs / sum(probs)
                    
                    next_city_index = self.RolletWheel(norm_probs)
                    
                    self.ants[ant_index].path.append(next_city_index)
                    
                    
                Cost = self.TSP.CostFunction(self.ants[ant_index].path)  
                
                self.ants[ant_index] = self.ants[ant_index]._replace(cost = Cost)
                
                if self.ants[ant_index].cost < self.best_ant.cost:
                    
                    self.best_ant = self.best_ant._replace(cost = self.ants[ant_index].cost,
                                                           path = self.ants[ant_index].path)
                    
            self.UpdateTue()
            
            self.best_costs[iteration] = self.best_ant.cost
            
            print(f"iteration {iteration} --> best cost is : {self.best_ant.cost}")
            
            #reset ants
            
            self.ants = {"ant_" + str(ant_index) : self.ant([], None) for ant_index in range(self.num_var)}
            
        # plot results
            
        plt.plot(self.best_costs)
        
        plt.show(block = True)
        
        plt.xlabel("iteration")
        
        plt.ylabel("best_cost")
        
        
    def getEta(self, distance):
        
        """ class for calculating eta parameter which will be used in calculation of probability of selecting next dcision parameters

        Args:
        
            distance : distance from one decision parameter to other ( in TSP distance between cities)

        Returns:
        
            no return --> reprirocal of input 

        """
        
        if distance == 0:
            
            return 0
        
        else:
            
            return 1 / distance             
            
       
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
        
        selected_city_index = list_of_valid[first_valid][first_valid]
        
        selected_index = selected_city_index + 1
            
        return selected_index
    
    
    def UpdateTue(self):
        
        """ class for calculating eta parameter which will be used in calculation of probability of selecting next dcision parameters

        Args:
        
            no argument

        Returns:
        
            no return --> update phromones of ants

        """
        
        for ant_index in self.ants:
            
            target_path = copy.copy(self.ants[ant_index].path)
            
            # add first city to last index to complete path
            
            target_path.append(target_path[0])
            
            for first_city_index, second_city_index in zip(target_path[0:], target_path[1:]):
                
                first_city_index = first_city_index - 1
                
                second_city_index = second_city_index - 1
                
                self.tau[first_city_index][second_city_index] = self.tau[first_city_index][second_city_index] + \
                                                                (self.Q / self.ants[ant_index].cost)
                
        self.tau * (1 - self.evaporation_rate)
        
        
class TSP:
    
    def __init__(self, num_cities, cities_x_lower_bound, cities_x_upper_bound, cities_y_lower_bound,
                            cities_y_upper_bound):
        
        """ TSP problem main class

        Args:
            
            num_cities: number of cities to create
            
            cities_x_lower_bound: lower bound for cities x position
            
            cities_x_upper_bound: upper bound for cities x position
            
            cities_y_lower_bound: lower bound for cities y position
            
            cities_y_upper_bound: upper bound for cities x position

        Returns:
        
            no return --> initializing TSP

        """
        
        self.num_cities = num_cities
        
        self.cities_x_lower_bound = cities_x_lower_bound
        
        self.cities_x_upper_bound = cities_x_upper_bound
        
        self.cities_y_lower_bound = cities_y_lower_bound
        
        self.cities_y_upper_bound = cities_y_upper_bound
        
        self.distances = np.zeros([self.num_cities, self.num_cities])
        
        self.cities_x = None
        
        self.cities_y = None
    
    
    def CreateCities(self):
        
        """ create cities for TSP

        Args:
            
            no argument

        Returns:
        
            no return --> initializing cities and their position

        """
        
        self.cities_x = np.random.randint(self.cities_x_lower_bound, self.cities_x_upper_bound, size = (self.num_cities))
    
        self.cities_y = np.random.randint(self.cities_y_lower_bound, self.cities_y_upper_bound, size = (self.num_cities))
        
        
    def CalculateDistance(self, first_city_index = None, second_city_index = None):
        
        """ calculate distance between cities

        Args:
            
            first_city_index : first city index
            
            second_city_index : second city index

        Returns:
        
            distance --> distance between two city 

        """
        
        if first_city_index == None and second_city_index == None:
        
            for city_index in range(self.num_cities - 1):

                for linked_city_index in range(city_index + 1, self.num_cities):

                    x_distance = np.square((self.cities_x[city_index] - self.cities_x[linked_city_index]))

                    y_distance = np.square((self.cities_y[city_index] - self.cities_y[linked_city_index]))

                    distance = np.sqrt(x_distance + y_distance)

                    self.distances[city_index][linked_city_index] = distance

                    self.distances[linked_city_index][city_index] = distance
                    
        else:
            
            x_distance = np.square((self.cities_x[first_city_index] - self.cities_x[second_city_index]))

            y_distance = np.square((self.cities_y[first_city_index] - self.cities_y[second_city_index]))

            distance = np.sqrt(x_distance + y_distance)
            
            return distance
                
    def Initialize(self):
        
        """ initilize TSP (create cities and calculate distance)

        Args:
            
            no argument

        Returns:
        
            no return --> create cities and calculate distance between them

        """
        
        self.CreateCities()
        
        self.CalculateDistance()
        
        
    def CostFunction(self, path):
        
        """ our problem cost function 

        Args:
        
            path: list of all visited city for one possible solution

        Returns:
        
            fitness: fitness of one possible solution

        """
        
        path_copy = copy.copy(path)
        
        path_copy.append(path_copy[0])
        
        distances = [self.distances[first_city_index - 1][second_city_index - 1] for first_city_index, \
                                     second_city_index in zip(path[0 : len(path) - 1], path[1 : len(path)])]
        
        fitness = sum(distances)
        
        return fitness
    
    
if __name__ == "__main__":
    
    ACO = argparse.ArgumentParser()
    
    ACO.add_argument("--evaporation_rate", default="0.5", help='rate in which phromones are evaporated', type = float)
    
    ACO.add_argument("--alpha", default="1", help='weight given to ants experience and useful in calculating probability of choosing next city from current one (refer paper)', type = float)
        
    ACO.add_argument("--betha", default="1",  help='weight given to our knowledge of problem and useful in calculating probability of choosing next city from current one (refer paper)', type = float)
        
    ACO.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
    
    ACO.add_argument("--max_iteration", default="600", help='max iteration for evolutionary method', type = int)
    
    ACO.add_argument("--Q", default="1", help='constant useful in calculating ants initial phromones', type = float)
    
    ACO.add_argument("--problem_definition", default="TSP", help='problem (benchmark) we want to apply ACO on that (default is TSP)', type = str)
    
    
    args = ACO.parse_args()
    
    if args.problem_definition == "TSP":
        
        num_cities = int(input("please input number of cities: "))
        
        cities_x_lower_bound = int(input("please input lower bound for cities x position: "))
        
        cities_x_upper_bound = int(input("please input upper bound for cities x position: "))
        
        cities_y_lower_bound = int(input("please input lower bound for cities y position:"))
        
        cities_y_upper_bound = int(input("please input upper bound for cities y position:"))
        
        benchmark = TSP(num_cities = num_cities, cities_x_lower_bound = cities_x_lower_bound, cities_x_upper_bound = cities_x_upper_bound,
                        cities_y_lower_bound = cities_y_lower_bound, cities_y_upper_bound = cities_y_upper_bound)
        
        benchmark.Initialize()
        
    else:
        
        #create a class for your problem definition and add here
        
        raise NotImplementedError
    

    ant = Ant(evaporation_rate = args.evaporation_rate, alpha = args.alpha, betha = args.betha, max_iteration = args.max_iteration, 
                                    Q = args.Q, number_of_ants = args.pop_size, benchmark = benchmark)
    

    ant.MainLoop()
