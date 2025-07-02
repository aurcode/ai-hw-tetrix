import random
import numpy as np

from src.agent.heuristic_ai_player import HeuristicAIPlayer

class EvolutionaryOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1, mutation_strength=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            weights = {
                'lines_cleared': random.uniform(10, 100),
                'aggregate_height': random.uniform(-10, -1),
                'holes': random.uniform(-20, -5),
                'bumpiness': random.uniform(-10, -1),
                'game_over_penalty': -1000.0 
            }
            population.append(HeuristicAIPlayer(weights=weights))
        return population

    def _select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # All individuals have fitness 0, select randomly
            return random.choices(self.population, k=2)

        selection_probs = [score / total_fitness for score in fitness_scores]
        return random.choices(self.population, weights=selection_probs, k=2)

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.weights.copy(), parent2.weights.copy()

        child1_weights, child2_weights = {}, {}
        for key in parent1.weights:
            if random.random() < 0.5:
                child1_weights[key] = parent1.weights[key]
                child2_weights[key] = parent2.weights[key]
            else:
                child1_weights[key] = parent2.weights[key]
                child2_weights[key] = parent1.weights[key]
        return child1_weights, child2_weights

    def _mutate(self, weights):
        mutated_weights = weights.copy()
        for key in mutated_weights:
            if key != 'game_over_penalty' and random.random() < self.mutation_rate:
                change = random.uniform(-self.mutation_strength, self.mutation_strength)
                mutated_weights[key] *= (1 + change)
        return mutated_weights

    def evolve(self, fitness_scores):
        new_population = []
        
        # Elitism: Keep the best individual
        best_individual_idx = np.argmax(fitness_scores)
        best_individual = self.population[best_individual_idx]
        new_population.append(HeuristicAIPlayer(weights=best_individual.weights.copy()))

        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents(fitness_scores)
            
            child1_weights, child2_weights = self._crossover(parent1, parent2)
            
            mutated_child1_weights = self._mutate(child1_weights)
            mutated_child2_weights = self._mutate(child2_weights)

            new_population.append(HeuristicAIPlayer(weights=mutated_child1_weights))
            if len(new_population) < self.population_size:
                new_population.append(HeuristicAIPlayer(weights=mutated_child2_weights))
        
        self.population = new_population
