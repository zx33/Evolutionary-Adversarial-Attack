import os
import random
import numpy as np

class EA_Util:
    def __init__(self, gen_size, pop_size=30, eval_func=None, max_gen=50, early_stop=0):
        self.gen_size = gen_size
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.early_stop = early_stop
        
        if eval_func == None:
            raise Exception("Undefined Evaluation Function")
        else:
            self.eval_func = eval_func
        
        self._init_pop()
        
    def _init_pop(self):
        population = []
        for _ in range(self.pop_size):
            individual = [1] * self.gen_size
            p = np.random.uniform(size=self.gen_size)
            for x in range(self.gen_size):
                individual[x] = 1 if p[x] <= 0.9 else 0
            population.append(individual)
        self.population = population
        self.fitness = [-1] * self.pop_size

    def _mutation(self, individual):
        new_chrom = individual.copy()
        t = random.sample(range(self.gen_size), 5)
        for s in t:
            new_chrom[s] = 1 - new_chrom[s]
        return new_chrom
    
    def _crossover(self, a, b):
        x = a.copy()
        for i in range(self.gen_size):
            p = random.random()
            if p > 0.5:
                x[i] = b[i]
        return x
    
    def _eval_pop(self):
        for x in range(self.pop_size):
            fit = self.eval_func(self.population[x])
            self.fitness[x] = fit
    
    def _reproduct(self):
        best = []
        fitsort = np.argsort(self.fitness)
        for x in fitsort[-5:]:
            best.append(x)
        for i in range(self.pop_size//2):
            if i in best:
                continue
            self.population[i] = self._mutation(self.population[random.choice(best)])
        for i in range(self.pop_size//2, self.pop_size):
            if i in best:
                continue
            t = random.sample(best, 2)
            self.population[i] = self._crossover(self.population[t[0]], self.population[t[1]])
    
    def evolution(self):
        self._eval_pop()
        print('Init Pop')
        best_fit = round(max(self.fitness), 4)
        temp_fit = self.fitness.copy()
        for i in range(self.pop_size):
            temp_fit[i] = round(temp_fit[i], 4)
        print('  Best Fitness: %.4f' % (best_fit))
        print('  Pop Fitness:', temp_fit)
        for gen in range(1, self.max_gen+1):
            print('%d evolution' % (gen))
            self._reproduct()
            self._eval_pop()
            best_fit = round(max(self.fitness), 4)
            temp_fit = self.fitness.copy()
            for i in range(self.pop_size):
                temp_fit[i] = round(temp_fit[i], 4)
            print('  Best Fitness: %.4f' % (best_fit))
            print('  Pop Fitness:', temp_fit)
        index = self.fitness.index(max(self.fitness))
        return index