import numpy as np
import symengine as se
from symengine import Symbol, Basic
import copy
import random
from deap import base, creator, tools, algorithms
import pandas as pd
from hobotan import *
import re
import functools
import time
from collections import defaultdict
import multiprocessing as mp
import os
import json
from datetime import datetime


# === Excel Output Functions ===
def classify_variable(var_name):
    """Classify variable type based on variable name"""
    var_name = str(var_name).lower()
    if 't_' in var_name or '_t' in var_name:
        return 'Structure_Selection'
    elif 'md_' in var_name or '_md' in var_name:
        return 'MD_Component'
    elif 'x_' in var_name or '_x' in var_name:
        return 'Composition'
    elif 'u_' in var_name:
        return 'Separator_Selection'
    elif 'p_' in var_name or '_p' in var_name:
        return 'Pressure'
    elif var_name.startswith('q'):
        return 'Binary_Variable'
    else:
        return 'Other'


def create_excel_report_deap(results, sym_list, C_expressions, filename=None):
    """Create DEAP optimization results Excel report"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deap_optimization_results_{timestamp}.xlsx"

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            # 1. Results Summary
            summary_data = []
            for i, (individual, fitness) in enumerate(results):
                selected_vars = sum(individual)
                summary_data.append({
                    'Solution_ID': i,
                    'Fitness_Value': fitness.values[0] if hasattr(fitness, 'values') else fitness,
                    'Selected_Variables': selected_vars,
                    'Selection_Rate': f"{(selected_vars / len(sym_list) * 100):.2f}%"
                })

            results_df = pd.DataFrame(summary_data)
            results_df.to_excel(writer, sheet_name='Optimization_Results', index=False)

            # 2. Best Solution Details
            if results:
                best_individual, best_fitness = results[0]

                var_details = []
                for i, (sym, value) in enumerate(zip(sym_list, best_individual)):
                    var_details.append({
                        'Variable_ID': i,
                        'Symbol': str(sym),
                        'Value': int(value),
                        'Variable_Type': classify_variable(str(sym)),
                        'Selected': 'Yes' if value == 1 else 'No'
                    })

                best_solution_df = pd.DataFrame(var_details)
                best_solution_df.to_excel(writer, sheet_name='Best_Solution_Details', index=False)

                # 3. Statistics
                fitness_values = [fitness.values[0] if hasattr(fitness, 'values') else fitness
                                  for _, fitness in results]
                selected_counts = [sum(individual) for individual, _ in results]

                stats_data = {
                    'Metric': [
                        'Total_Solutions', 'Best_Fitness', 'Worst_Fitness', 'Mean_Fitness',
                        'Std_Fitness', 'Best_Selected_Vars', 'Mean_Selected_Vars', 'Total_Variables'
                    ],
                    'Value': [
                        len(results),
                        min(fitness_values),
                        max(fitness_values),
                        np.mean(fitness_values),
                        np.std(fitness_values) if len(fitness_values) > 1 else 0,
                        min(selected_counts),
                        np.mean(selected_counts),
                        len(sym_list)
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)

                # 4. Calculate C element values
                print("Calculating C element values for best solution...")
                try:
                    substitution_dict = {}
                    for i, sym in enumerate(sym_list):
                        if i < len(best_individual):
                            substitution_dict[sym] = best_individual[i]
                        else:
                            substitution_dict[sym] = 0

                    c_values = []
                    c_data = []

                    for i, c_expr in enumerate(C_expressions):
                        try:
                            c_value = float(c_expr.subs(substitution_dict))
                            c_values.append(c_value)
                            c_data.append({
                                'Element_ID': f'C[{i}]',
                                'Value': c_value,
                                'Absolute_Value': abs(c_value),
                                'Non_Zero': 'Yes' if abs(c_value) > 1e-10 else 'No'
                            })
                        except Exception as e:
                            c_values.append(None)
                            c_data.append({
                                'Element_ID': f'C[{i}]',
                                'Value': 'Error',
                                'Absolute_Value': 0,
                                'Non_Zero': 'No',
                                'Error': str(e)[:100]
                            })

                    c_elements_df = pd.DataFrame(c_data)
                    c_elements_df_sorted = c_elements_df.sort_values('Absolute_Value', ascending=False)
                    c_elements_df_sorted.to_excel(writer, sheet_name='C_Elements_Values', index=False)

                    print(
                        f"C element calculation completed: {len([v for v in c_values if v is not None])}/{len(C_expressions)} successful")

                except Exception as c_error:
                    print(f"C element calculation error: {c_error}")

        print(f"Results saved to Excel file: {filename}")
        return filename

    except Exception as e:
        print(f"Excel output error: {e}")
        return None


# === Memory-efficient Square Calculation ===
COEFFICIENT_THRESHOLD = 1e-8


def normalize_binary_term(term):
    """Normalize binary variable term"""
    if term == '1':
        return '1'
    vars_set = set(term.split('*'))
    return '*'.join(sorted(vars_set))


def process_var_range(args):
    """Process variable range (worker function)"""
    worker_id, start_idx, end_idx, variables, coefficients = args
    result_dict = defaultdict(float)

    processed_pairs = 0
    skipped_small = 0

    # 1. Square terms (for binary variables, x^2 = x)
    for i in range(start_idx, end_idx):
        if i >= len(variables):
            break

        var_i = variables[i]
        coef_i = coefficients[i]
        squared_coef = coef_i * coef_i

        if abs(squared_coef) > COEFFICIENT_THRESHOLD:
            result_dict[var_i] += squared_coef
        else:
            skipped_small += 1

    # 2. Cross terms
    for i in range(start_idx, end_idx):
        if i >= len(variables):
            break

        var_i = variables[i]
        coef_i = coefficients[i]

        if abs(coef_i) <= COEFFICIENT_THRESHOLD:
            continue

        for j in range(i + 1, len(variables)):
            var_j = variables[j]
            coef_j = coefficients[j]

            if abs(coef_j) <= COEFFICIENT_THRESHOLD:
                continue

            cross_term_coef = 2.0 * coef_i * coef_j

            if abs(cross_term_coef) <= COEFFICIENT_THRESHOLD:
                skipped_small += 1
                continue

            processed_pairs += 1

            if var_i == '1':
                cross_term = var_j
            elif var_j == '1':
                cross_term = var_i
            else:
                combined = var_i + '*' + var_j
                cross_term = normalize_binary_term(combined)

            result_dict[cross_term] += cross_term_coef

    final_result = {}
    for term, coef in result_dict.items():
        if abs(coef) > COEFFICIENT_THRESHOLD:
            final_result[term] = coef

    return final_result


def memory_efficient_square(expr, num_processes=32, threshold=1e-8):
    """Memory-efficient square calculation"""
    global COEFFICIENT_THRESHOLD
    COEFFICIENT_THRESHOLD = threshold

    variables, coefficients = extract_symengine_expression(expr)
    num_vars = len(variables)

    if num_vars <= 100:
        result_dict = process_var_range((0, 0, num_vars, variables, coefficients))
        return polynomial_to_symengine(result_dict)

    chunks = split_variable_range(num_vars, num_processes)

    pool = mp.Pool(processes=min(len(chunks), num_processes))
    try:
        args = [(i, chunks[i][0], chunks[i][1], variables, coefficients) for i in range(len(chunks))]
        results = pool.map(process_var_range, args)
    finally:
        pool.close()
        pool.join()

    final_result = defaultdict(float)
    for result_dict in results:
        for term, coef in result_dict.items():
            final_result[term] += coef

    cleaned_result = {}
    for term, coef in final_result.items():
        if abs(coef) > threshold:
            cleaned_result[term] = coef

    return polynomial_to_symengine(cleaned_result)


def split_variable_range(num_vars, num_chunks, max_memory_gb=4):
    """Split variable range"""
    chunk_size = max(1, num_vars // num_chunks)
    chunks = []

    for i in range(0, num_vars, chunk_size):
        end_idx = min(i + chunk_size, num_vars)
        chunks.append((i, end_idx))

    return chunks


def polynomial_to_symengine(result_dict):
    """Convert dictionary result to symengine expression"""
    if not result_dict:
        return se.Integer(0)

    symbol_cache = {}

    def get_symbol(var_name):
        if var_name not in symbol_cache:
            symbol_cache[var_name] = se.Symbol(var_name)
        return symbol_cache[var_name]

    terms = []
    for term_str, coef in result_dict.items():
        if abs(coef) < 1e-12:
            continue

        if term_str == '1':
            if abs(coef - round(coef)) < 1e-10:
                terms.append(se.Integer(round(coef)))
            else:
                terms.append(se.Float(coef))
        else:
            if abs(coef - round(coef)) < 1e-10:
                coef_part = se.Integer(round(coef))
            else:
                coef_part = se.Float(coef)

            var_names = term_str.split('*')
            var_part = se.Integer(1)
            for var_name in var_names:
                var_part *= get_symbol(var_name)

            terms.append(coef_part * var_part)

    if not terms:
        return se.Integer(0)
    elif len(terms) == 1:
        return terms[0]
    else:
        return sum(terms)


def extract_symengine_expression(expr):
    """Extract variables and coefficients list from symengine expression"""
    variables = []
    coefficients = []

    if expr.is_Add:
        for term in expr.args:
            vars_list, coefs_list = extract_symengine_expression(term)
            for var, coef in zip(vars_list, coefs_list):
                if var in variables:
                    idx = variables.index(var)
                    coefficients[idx] += coef
                else:
                    variables.append(var)
                    coefficients.append(coef)
        return variables, coefficients

    coef = 1.0
    var_parts = set()

    if expr.is_Mul:
        for factor in expr.args:
            if factor.is_Number:
                coef *= float(factor)
            elif factor.is_Symbol:
                var_parts.add(str(factor))
            elif factor.is_Pow and factor.args[0].is_Symbol:
                base, exp = factor.args
                var_parts.add(str(base))
            else:
                sub_vars, sub_coefs = extract_symengine_expression(factor)
                for i, sub_var in enumerate(sub_vars):
                    if sub_var == '1':
                        coef *= sub_coefs[i]
                    else:
                        for part in sub_var.split('*'):
                            var_parts.add(part)
                        coef *= sub_coefs[i]
    elif expr.is_Symbol:
        var_parts.add(str(expr))
    elif expr.is_Number:
        return ['1'], [float(expr)]
    elif expr.is_Pow:
        base, exp = expr.args
        if base.is_Symbol:
            var_parts.add(str(base))
        else:
            sub_vars, sub_coefs = extract_symengine_expression(base)
            for sub_var in sub_vars:
                if sub_var == '1':
                    coef *= sub_coefs[0] ** float(exp)
                else:
                    for part in sub_var.split('*'):
                        var_parts.add(part)
                    coef *= sub_coefs[0] ** float(exp)

    var_list = sorted(var_parts)
    var_str = '*'.join(var_list) if var_list else '1'

    return [var_str], [coef]


# === Required Functions from Original Code ===
def efficient_expand(expr):
    """Efficient expansion"""
    st = time.time()
    expanded = cached_expand(expr)
    ed = time.time()
    return expanded


class Binary(Basic):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def _eval_power(self, other):
        return self


@functools.lru_cache(maxsize=1024)
def cached_expand(expr):
    """Cached expansion"""
    return se.expand(expr)


def simplify_binary_expression(expr):
    """Simplify binary variable exponents"""
    expr_str = str(expr)
    pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\*\*[1-9][0-9]*'
    simplified_str = re.sub(pattern, r'\1', expr_str)
    result = se.sympify(simplified_str)
    return result


def get_cofactor_matrix(mad):
    """Calculate cofactor matrix"""
    n = mad.shape[0]
    cofactor_list = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            minor_list = [[0 for _ in range(n - 1)] for _ in range(n - 1)]
            row_count = 0
            for r in range(n):
                if r != i:
                    col_count = 0
                    for c in range(n):
                        if c != j:
                            minor_list[row_count][col_count] = mad[r, c]
                            col_count += 1
                    row_count += 1

            minor = se.Matrix(minor_list)

            count = 0
            for k in range(len(minor[0, :])):
                for l in range(len(minor[:, 0])):
                    if not isinstance(minor[k, l], se.Integer):
                        count += 1
            sy = se.symbols("sy0:" + str(count))
            symdict = {}
            count = 0
            for k in range(len(minor[0, :])):
                for l in range(len(minor[:, 0])):
                    if not isinstance(minor[k, l], se.Integer):
                        symdict[sy[count]] = minor[k, l]
                        minor[k, l] = sy[count]
                        count += 1
                    else:
                        continue

            cofactor = ((-1) ** (i + j)) * minor.det()
            cofactor = efficient_expand(cofactor)
            cofactor_list[j][i] = cofactor.subs(symdict)
            cofactor_list[j][i] = efficient_expand(cofactor_list[j][i])

    return se.Matrix(cofactor_list)


def efficient_block_det(A, B, C, D):
    """Efficiently calculate determinant of block matrix"""
    if isinstance(A, se.Matrix):
        A_diag = [A[i, i] for i in range(A.shape[0])]
        B_diag = [B[i, i] for i in range(B.shape[0])]
        C_diag = [C[i, i] for i in range(C.shape[0])]
        D_diag = [D[i, i] for i in range(D.shape[0])]
    else:
        A_diag, B_diag, C_diag, D_diag = A, B, C, D

    det = se.Integer(1)
    for i in range(len(A_diag)):
        block_det = A_diag[i] * D_diag[i] - B_diag[i] * C_diag[i]
        det = det * block_det

    return det


# === DEAP Evaluation Function ===
def evaluate_objective(individual, C_expressions, sym_list):
    """Evaluate objective function value for individual"""
    try:
        # Create substitution dictionary for variables
        substitution_dict = {}
        for i, sym in enumerate(sym_list):
            if i < len(individual):
                substitution_dict[sym] = individual[i]
            else:
                substitution_dict[sym] = 0

        # Calculate objective function value
        total_objective = 0
        for i, expr in enumerate(C_expressions):
            try:
                value = float(expr.subs(substitution_dict))
                total_objective += value
            except Exception as e:
                # Large penalty for calculation errors
                print(f"C[{i}] calculation error: {e}")
                total_objective += 1e10

        return (total_objective,)

    except Exception as e:
        print(f"Evaluation function error: {e}")
        return (1e10,)


# === Main Execution (Using Original Code) ===
def main():
    print("Starting optimization problem setup...")

    # Execute original code
    from func_218 import symb

    total_time = time.time()

    cmpnum = 2


    Species = ["A", "B", "C"]
    num = 2
    MDcmp = cmpnum
    cmp = cmpnum
    local, sym, lsum, moji_list = symb(num, Species, MDcmp, cmp)

    t_list = lsum[0]
    MD_list = lsum[1]
    X_list = lsum[2]
    globals().update(local)

    M = [se.zeros(len(Species))] * num

    B = [se.Matrix([[0.6, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]), se.Matrix([[0.1, 0, 0], [0, 0.6, 0], [0, 0, 0.1]]),
         se.Matrix([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.6]])]

    for n in range(num):
        count = 0
        for s in sym[3 * n:3 + 3 * n]:
            M[n] += s * B[count]
            count += 1
    EYE = se.eye(len(Species))

    P = [0] * num
    z = len(Species) * num
    count = 0
    for i in range(num):
        P[i] = M[0] * sym[z + 2 * num * i] + (EYE - M[0]) * sym[z + 2 * num * i + 1]
        for n in range(num - 1):
            pp = P[i].row_join(
                M[n + 1] * sym[z + 2 * num * i + 2 * n + 2] + (EYE - M[n + 1]) * sym[z + 2 * num * i + 2 * n + 3])
            P[i] = pp
            count += 1

    MP = P[0]
    for i in range(num - 1):
        MP = MP.col_join(P[i + 1])
    MP = efficient_expand(MP)

    E = se.eye(MP.shape[0])
    E3 = se.eye(3)

    print("Starting cofactor matrix calculation...")
    start_time = time.time()
    Mad = get_cofactor_matrix(E - MP)
    end_time = time.time()
    print(f"Cofactor matrix calculation completed: {end_time - start_time:.2f} seconds")

    print("Simplifying each element of cofactor matrix...")
    start_time = time.time()
    for i in range(len(Mad[0, :])):
        for j in range(len(Mad[:, 0])):
            Mad[i, j] = simplify_binary_expression(Mad[i, j])
    end_time = time.time()
    print(f"Cofactor matrix simplification completed: {end_time - start_time:.2f} seconds")

    print("Calculating determinant...")
    start_time = time.time()
    Mdet = efficient_block_det(efficient_expand(E3 - P[0][0:3, 0:3]), efficient_expand(P[0][0:3, 3:6]),
                               efficient_expand(P[1][0:3, 0:3]), efficient_expand(E3 - P[1][0:3, 3:6]))
    end_time = time.time()
    Mdet = efficient_expand(Mdet)
    Mdet = simplify_binary_expression(Mdet)
    print(f"Determinant calculation completed: {end_time - start_time:.2f} seconds")

    MDfin = 1
    MDET = 0
    for i in range(MDcmp):
        MDET += MDfin / (2 ** (MDcmp) - 1) * 2 ** (i) * MD_list[i]

    print("Calculating inverse matrix...")
    start_time = time.time()
    Minv = Mad * MDET
    for i in range(len(Minv[0, :])):
        for j in range(len(Minv[:, 0])):
            Minv[i, j] = efficient_expand(Minv[i, j])
    end_time = time.time()
    print(f"Inverse matrix calculation completed: {end_time - start_time:.2f} seconds")

    F = se.zeros(len(Species), 3 * (num + 1))
    F_list = [100] * len(Species)
    for i in range(len(Species)):
        F[i, 0] = F_list[i]

    FF = F[:, 0].col_join(se.zeros(3 * (num - 1), 1))

    print("Calculating flows...")
    start_time = time.time()
    FT = Minv * FF
    FT = efficient_expand(FT)
    for n in range(num):
        F[:, 1 + 3 * n] = FT[3 * n: 3 * n + len(Species), 0]
        F[:, 1 + 3 * n + 1] = M[n] * F[:, 1 + 3 * n]
        F[:, 1 + 3 * n + 2] = (EYE - M[n]) * F[:, 1 + 3 * n]
    end_time = time.time()
    print(f"Flow calculation completed: {end_time - start_time:.2f} seconds")

    F = efficient_expand(F)

    C = []
    Vl = [100, 50, 10]
    X = []
    for n in range(num):
        X.append(sum(F[:, 3 * n + 2]))

    # Xfin = [0.02, 0.02]
    Xfin = [0.010, 0.010]

    print("Calculating expressions...")
    start_time = time.time()
    for n in range(num):
        X_sum = 0.005
        for i in range(cmp):
            X_sum += Xfin[n] / (2 ** (cmp) - 1) * 2 ** (i) * X_list[i + cmp * n]
        Fsubs = []
        for i in range(3):
            st = time.time()
            print(f"Starting square calculation for F[{i}, {3 * n + 2}]...")
            F0 = simplify_binary_expression(F[i, 3 * n + 2])
            F0 = memory_efficient_square(F0, num_processes=32, threshold=1e-8)
            print(f"Parallel calculation time: {time.time() - st:.3f} seconds")
            Fsubs.append(F0)

        D = -(sym[-2 * (num - n) - MDcmp - num * cmp] * (
                Vl[0] * sym[3 * n] * Fsubs[0] + Vl[1] * sym[3 * n + 1] * Fsubs[1] + Vl[2] * sym[
            3 * n + 2] * Fsubs[2])) * X_sum
        C.append(efficient_expand(D))
    end_time = time.time()
    print(f"Expression calculation completed: {end_time - start_time:.2f} seconds")

    # Constraints
    print("Calculating constraint conditions...")
    start_time = time.time()
    for i in range(int(2 * num)):
        D = ((sum(t_list[i::2 * num]) - 1) ** 2)
        C.append(efficient_expand(D))

    D = efficient_expand(Mdet * MDET - 1)
    D = simplify_binary_expression(D)
    D = memory_efficient_square(D, num_processes=32, threshold=1e-8)
    D = simplify_binary_expression(D)
    C.append(D)

    for n in range(num):
        X_sum = 0.005
        for i in range(cmp):
            X_sum += Xfin[n] / (2 ** (cmp) - 1) * 2 ** (i) * X_list[i + cmp * n]
        CSD = sym[-2 * (num - n) - MDcmp - num * cmp] * (X_sum * X[n] - 1)
        if int(n) == int(1):
            CSD = CSD * (t_24 + t_34 - t_24 * t_34)
        CSD = efficient_expand(CSD)
        CSD = simplify_binary_expression(CSD)
        D = memory_efficient_square(CSD, num_processes=32, threshold=1e-8)
        C.append(D)
    end_time = time.time()
    print(f"Constraint condition calculation completed: {end_time - start_time:.2f} seconds")

    print("Separator selection constraints")
    C.append(efficient_expand((u_IA + u_IB + u_IC - 1) ** 2))
    C.append(efficient_expand((u_IIA + u_IIB + u_IIC - 1) ** 2))

    for i in range(len(C) - num):
        C[i + num] = C[i + num] * 10 ** 6

    print(f"Number of objective function terms: {len(C)}")
    print(f"Number of variables: {len(sym)}")

    # === DEAP Setup and Execution ===
    print("Setting up DEAP genetic algorithm...")

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, len(sym))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_objective, C_expressions=C, sym_list=sym)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Parameters
    POPULATION_SIZE = 100
    GENERATIONS = 500
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.2

    print("Running genetic algorithm...")
    population = toolbox.population(n=POPULATION_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Execute algorithm
    optimization_start_time = time.time()

    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=GENERATIONS,
        stats=stats,
        verbose=True
    )

    optimization_end_time = time.time()
    print(f"Optimization completed: {optimization_end_time - optimization_start_time:.2f} seconds")

    # === Result Processing ===
    print("Processing results...")

    # Get best solution
    fits = [ind.fitness.values[0] for ind in population]
    best_idx = fits.index(min(fits))
    best_individual = population[best_idx]
    best_fitness = best_individual.fitness

    print(f"Best fitness: {best_fitness.values[0]}")
    print(f"Number of selected variables: {sum(best_individual)}")

    # Sort results by fitness
    evaluated_population = [(ind, ind.fitness) for ind in population]
    evaluated_population.sort(key=lambda x: x[1].values[0])

    # Excel output
    print("Saving results to Excel...")
    filename = create_excel_report_deap(evaluated_population, sym, C)

    if filename:
        print(f"Optimization completed! Results saved to {filename}")
    else:
        print("Error occurred while saving Excel file.")

    # Display best solution details
    print("\n=== Best Solution Details ===")
    print(f"Fitness value: {best_fitness.values[0]:.6f}")
    print(f"Selected variables: {sum(best_individual)}/{len(sym)}")
    print("Selected variables (first 20):")
    selected_count = 0
    for i, (value, symbol) in enumerate(zip(best_individual, sym)):
        if value == 1:
            print(f"  {str(symbol)}: {value}")
            selected_count += 1
            if selected_count >= 20:
                print(f"  ... (and {sum(best_individual) - 20} more variables are selected)")
                break

    # Display C element values
    print("\n=== C Element Values (first 10) ===")
    substitution_dict = {}
    for i, sym_var in enumerate(sym):
        if i < len(best_individual):
            substitution_dict[sym_var] = best_individual[i]
        else:
            substitution_dict[sym_var] = 0

    for i, c_expr in enumerate(C[:10]):
        try:
            c_value = float(c_expr.subs(substitution_dict))
            print(f"C[{i}]: {c_value:.6f}")
        except Exception as e:
            print(f"C[{i}]: Calculation error - {str(e)[:50]}")

    if len(C) > 10:
        print(f"... (and {len(C) - 10} more C elements)")

    total_end_time = time.time()
    print(f"\nTotal execution time: {total_end_time - total_time:.2f} seconds")


if __name__ == "__main__":
    main()
