import random
import math


# Definirea funcției de fitness (polinom de gradul 2)
def fitness_function(x):
    return -x ** 2 + x + 2


# Codificarea cromozomului
def encode_chromosome(x, precision):
    return format(int((x + 1) * 10 ** precision), '0' + str(precision) + 'b')


# Decodificarea cromozomului
def decode_chromosome(chromosome, precision):
    return (int(chromosome, 2) / 10 ** precision) - 1


# Inițializarea populației
def initialize_population(population_size, precision):
    population = []
    for _ in range(population_size):
        x = random.uniform(-1, 2)
        chromosome = encode_chromosome(x, precision)
        population.append(chromosome)
    return population


# Evaluarea populației
def evaluate_population(population, precision):
    evaluated_population = []
    for chromosome in population:
        x = decode_chromosome(chromosome, precision)
        fitness = fitness_function(x)
        evaluated_population.append((chromosome, x, fitness))
    return evaluated_population


# Selectia parintilor
def select_parents(evaluated_population):
    total_fitness = sum(fitness for _, _, fitness in evaluated_population)
    probabilities = [fitness / total_fitness for _, _, fitness in evaluated_population]
    return random.choices(evaluated_population, probabilities, k=2)


# Crossover
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    return parent1, parent2


# Mutation
def mutate(chromosome, mutation_rate):
    mutated_chromosome = ''
    for gene in chromosome:
        if random.random() < mutation_rate:
            mutated_chromosome += '0' if gene == '1' else '1'
        else:
            mutated_chromosome += gene
    return mutated_chromosome


# Algoritmul genetic
def genetic_algorithm(population_size, precision, crossover_rate, mutation_rate, num_generations):
    population = initialize_population(population_size, precision)
    best_solution = None

    with open("Evolutie.txt", "w") as file:
        for generation in range(num_generations):
            evaluated_population = evaluate_population(population, precision)
            best_solution = max(evaluated_population, key=lambda x: x[2])

            file.write(f"Populatia initiala\n")
            for idx, (chromosome, x, fitness) in enumerate(evaluated_population, start=1):
                file.write(f"   {idx}: {chromosome} x= {x:.6f} f={fitness}\n")

            # Probabilități de selecție
            file.write("\nProbabilitati selectie\n")
            for idx, (chromosome, _, _) in enumerate(evaluated_population, start=1):
                probability = chromosome.count('1') / len(chromosome)
                file.write(f"cromozom {idx} probabilitate {probability}\n")

            # Intervale probabilități de selecție
            probabilities_cumulative = [0] * (population_size + 1)
            for i in range(population_size):
                probabilities_cumulative[i + 1] = probabilities_cumulative[i] + (
                            evaluated_population[i][0].count('1') / len(evaluated_population[i][0]))
            file.write("\nIntervale probabilitati selectie\n")
            for p in probabilities_cumulative:
                file.write(f"{p:.15f} ")
            file.write("\n")

            # Reproducere
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = select_parents(evaluated_population)
                child1, child2 = crossover(parent1[0], parent2[0], crossover_rate)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                new_population.extend([child1, child2])

            population = new_population[:population_size]

            #Dupa recombinare
            file.write("\nDupa selectie:\n")
            for idx, (chromosome, x, fitness) in enumerate(evaluated_population, start=1):
                file.write(f"   {idx}: {chromosome} x= {x:.6f} f={fitness}\n")

            # Probabilitate de incrucisare
            file.write(f"\nProbabilitatea de incrucisare {crossover_rate}\n")
            for idx, chromosome in enumerate(population, start=1):
                u = chromosome.count('1') / len(chromosome)
                file.write(f"{idx}: {chromosome} u={u}\n")

            # Recombinare
            file.write("\nRecombinare dintre cromozomi:\n")
            for _ in range(population_size // 2):
                parent1, parent2 = select_parents(evaluated_population)
                child1, child2 = crossover(parent1[0], parent2[0], crossover_rate)
                file.write(
                    f"{parent1[0]} {parent2[0]} punct  {random.randint(0, min(len(parent1[0]), len(parent2[0])))}\n")
                file.write(f"Rezultat    {child1} {child2}\n")

            # După recombinare
            file.write("\nDupa recombinare:\n")
            evaluated_population = evaluate_population(population, precision)
            for idx, (chromosome, x, fitness) in enumerate(evaluated_population, start=1):
                file.write(f"   {idx}: {chromosome} x= {x:.6f} f={fitness}\n")

            # Mutare
            file.write(f"\nProbabilitate de mutatie pentru fiecare gena {mutation_rate}\n")
            mutated_indices = set()
            for _ in range(population_size):
                idx = random.randint(0, population_size - 1)
                if idx not in mutated_indices:
                    mutated_indices.add(idx)
                    file.write(f"{idx + 1}\n")

            # Dupa mutatie
            file.write("\nDupa mutatie:\n")
            for idx, (chromosome, x, fitness) in enumerate(evaluated_population, start=1):
                file.write(f"   {idx}: {chromosome} x= {x:.6f} f={fitness}\n")

            # Evo max
            file.write("\nEvolutia maximului\n")
            for _ in range(num_generations):
                file.write(f"{best_solution[2]}\n")


#Input param
population_size = 20
precision = 6
crossover_rate = 0.25
mutation_rate = 0.01
num_generations = 50


genetic_algorithm(population_size, precision, crossover_rate, mutation_rate, num_generations)
