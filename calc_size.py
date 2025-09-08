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
import symengine as se
from amplify import VariableGenerator, Model, AcceptableDegrees, BinaryPoly
from amplify import FixstarsClient, Solver
import re
import functools
import time
import gc
import psutil
import signal
import json
from datetime import datetime
import subprocess
def emergency_swap_setup(swap_size_gb=400):  # 64GBに変更
    """Emergency swap setup to avoid SIGKILL - optimized for large swap"""
    try:
        print(f"=== EMERGENCY SWAP SETUP ({swap_size_gb}GB) ===")

        # Check multiple locations for swap creation
        locations = [
            ("/", "Root disk"),
            (os.path.expanduser("~"), "Home directory"),
            ("/var", "Var directory"),
            ("/tmp", "Tmp directory")
        ]

        best_location = None
        max_available = 0

        for location, name in locations:
            try:
                disk_usage = subprocess.run(['df', location], capture_output=True, text=True)
                lines = disk_usage.stdout.strip().split('\n')
                available_kb = int(lines[1].split()[3])
                available_gb = available_kb / (1024 * 1024)
                print(f"{name}: {available_gb:.1f} GB available")

                if available_gb > max_available:
                    max_available = available_gb
                    best_location = location
            except:
                continue

        if max_available < swap_size_gb + 5:  # Need 5GB buffer
            swap_size_gb = max(8, int(max_available - 5))
            print(f"Adjusting swap size to {swap_size_gb}GB based on available space")

        # Use best location for swap file
        if best_location == "/tmp":
            swap_file = "/tmp/emergency_swap"
        else:
            swap_file = f"{best_location}/emergency_swap_{swap_size_gb}gb"

        print(f"Creating swap file at: {swap_file}")

        # Check if swap already exists
        try:
            result = subprocess.run(['swapon', '--show'], capture_output=True, text=True)
            if swap_file in result.stdout or 'emergency_swap' in result.stdout:
                print("Emergency swap already active")
                return True
        except:
            pass

        # Create swap file with progress indication
        print(f"Creating {swap_size_gb}GB emergency swap (this may take a few minutes)...")

        # Remove existing file if present
        if os.path.exists(swap_file):
            try:
                subprocess.run(['sudo', 'swapoff', swap_file], capture_output=True)
                os.remove(swap_file)
            except:
                pass

        # Create swap file - try different methods
        try:
            # Method 1: fallocate (fastest)
            result = subprocess.run(['sudo', 'fallocate', '-l', f'{swap_size_gb}G', swap_file],
                                    capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception("fallocate failed")
        except:
            try:
                # Method 2: dd with large block size
                print("fallocate failed, using dd (slower but reliable)...")
                subprocess.run(['sudo', 'dd', 'if=/dev/zero', f'of={swap_file}',
                                'bs=1M', f'count={swap_size_gb * 1024}', 'status=progress'],
                               timeout=1800)  # 30 minute timeout
            except:
                raise Exception("Both fallocate and dd failed")

        # Set permissions and create swap
        subprocess.run(['sudo', 'chmod', '600', swap_file], check=True)
        subprocess.run(['sudo', 'mkswap', swap_file], check=True, timeout=300)
        subprocess.run(['sudo', 'swapon', swap_file], check=True)

        # Optimize for large swap usage
        print("Optimizing system for large swap usage...")
        optimizations = [
            ('vm.swappiness', '1'),  # Minimize swap usage
            ('vm.vfs_cache_pressure', '50'),  # Reduce cache pressure
            ('vm.max_map_count', '2097152'),  # Increase memory mapping limit
            ('vm.overcommit_memory', '1'),  # Enable memory overcommit
            ('vm.overcommit_ratio', '50'),  # Conservative overcommit
            ('kernel.pid_max', '4194304'),  # Increase process limit
        ]

        for param, value in optimizations:
            try:
                subprocess.run(['sudo', 'sysctl', '-w', f'{param}={value}'],
                               capture_output=True)
                print(f"  Set {param} = {value}")
            except:
                pass

        print(f"SUCCESS: {swap_size_gb}GB swap activated at {swap_file}")

        # Show final status
        subprocess.run(['free', '-h'])
        print("\nActive swap devices:")
        subprocess.run(['swapon', '--show'])

        return True

    except Exception as e:
        print(f"Emergency swap setup failed: {e}")
        print("\nManual setup commands:")
        print(f"sudo fallocate -l {swap_size_gb}G /emergency_swap_{swap_size_gb}gb")
        print(f"sudo chmod 600 /emergency_swap_{swap_size_gb}gb")
        print(f"sudo mkswap /emergency_swap_{swap_size_gb}gb")
        print(f"sudo swapon /emergency_swap_{swap_size_gb}gb")
        return False


# 実行部分も修正
print("Executing emergency swap setup...")
swap_success = emergency_swap_setup(64)  # 64GBに変更

if swap_success:
    print("Large swap activated. Setting up memory monitoring...")


    # Memory monitoring function for large swap usage
    def setup_memory_monitoring():
        """Setup memory monitoring for large swap scenarios"""
        try:
            import threading
            import time

            def monitor_memory():
                max_memory = 0
                max_swap = 0
                start_time = time.time()

                while True:
                    try:
                        # Process memory
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / (1024 ** 2)
                        max_memory = max(max_memory, memory_mb)

                        # System swap
                        swap = psutil.swap_memory()
                        swap_gb = swap.used / (1024 ** 3)
                        max_swap = max(max_swap, swap_gb)

                        # Alert levels
                        if memory_mb > 10000:  # 10GB
                            elapsed = time.time() - start_time
                            print(
                                f"HIGH MEMORY: {memory_mb:.0f}MB RAM, {swap_gb:.1f}GB swap used ({elapsed / 60:.1f}min elapsed)")

                        if swap.percent > 25:  # 25% swap usage
                            print(f"SWAP USAGE: {swap.percent:.1f}% ({swap_gb:.1f}GB used)")

                        time.sleep(30)  # Check every 30 seconds

                    except:
                        break

            monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            monitor_thread.start()
            print("Memory monitoring started (30-second intervals)")

        except Exception as e:
            print(f"Memory monitoring setup failed: {e}")


    setup_memory_monitoring()

else:
    print("CRITICAL: Large swap setup failed.")
    response = input("Continue with limited memory? (y/N): ")
    if response.lower() != 'y':
        exit(1)


# emergency_swap_setup(64)  # 64GB
# emergency_swap_setup(128) # 128GB

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
            # Results Summary
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

            # Best Solution Details
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

                # Statistics
                fitness_values = [fitness.values[0] if hasattr(fitness, 'values') else fitness
                                  for _, fitness in results]
                selected_counts = [sum(individual) for individual, _ in results]

                stats_data = {
                    'Metric': [
                        'Total_Solutions', 'Best_Fitness', 'Worst_Fitness', 'Mean_Fitness',
                        'Std_Fitness', 'Best_Selected_Vars', 'Mean_Selected_Vars', 'Total_Variables'
                    ],
                    'Value': [
                        len(results), min(fitness_values), max(fitness_values),
                        np.mean(fitness_values), np.std(fitness_values) if len(fitness_values) > 1 else 0,
                        min(selected_counts), np.mean(selected_counts), len(sym_list)
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)

                # Calculate C element values
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


# === Memory-efficient square calculation for binary variables ===
COEFFICIENT_THRESHOLD = 1e-6


def normalize_binary_term(term):
    """Normalize binary variable term (remove duplicates and sort)"""
    if term == '1':
        return '1'
    vars_set = set(term.split('*'))
    return '*'.join(sorted(vars_set))


def process_var_range(args):
    """Worker function to process variable range"""
    worker_id, start_idx, end_idx, variables, coefficients = args
    result_dict = defaultdict(float)

    # Bind to CPU if possible
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_count = psutil.cpu_count()
        if cpu_count is not None:
            process.cpu_affinity([worker_id % cpu_count])
    except (ImportError, AttributeError):
        pass

    start_time = time.time()
    print(f"Worker {worker_id}: Processing terms {start_idx} to {end_idx - 1}")

    processed_pairs = 0
    skipped_small = 0

    # Square terms (for binary variables, x^2 = x)
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

    # Cross terms
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

    processing_time = time.time() - start_time
    unique_terms = len(result_dict)

    final_result = {}
    for term, coef in result_dict.items():
        if abs(coef) > COEFFICIENT_THRESHOLD:
            final_result[term] = coef

    print(f"Worker {worker_id}: {unique_terms} unique terms in {processing_time:.2f}s")
    print(f"  Processed cross terms: {processed_pairs}, Skipped small: {skipped_small}")

    return final_result


def memory_efficient_square(expr, num_processes=16, threshold=1e-6):
    """Memory-efficient square calculation for large polynomials"""
    global COEFFICIENT_THRESHOLD
    COEFFICIENT_THRESHOLD = threshold

    variables, coefficients = extract_symengine_expression(expr)
    num_vars = len(variables)

    print(f"Extracted terms: {num_vars}")
    print(f"Coefficient threshold: {threshold}")

    coef_abs = [abs(c) for c in coefficients]
    print(f"Coefficient range: [{min(coef_abs):.2e}, {max(coef_abs):.2e}]")

    all_vars = set()
    for term in variables:
        if term != '1':
            all_vars.update(term.split('*'))
    print(f"Binary variables used: {len(all_vars)}")

    if num_vars <= 100:
        print("Small number of terms, calculating directly")
        result_dict = process_var_range((0, 0, num_vars, variables, coefficients))
        return polynomial_to_symengine(result_dict)

    start_time = time.time()
    chunks = split_variable_range(num_vars, num_processes)
    print(f"Variable range split: {time.time() - start_time:.2f}s")

    print(f"Starting parallel calculation with {len(chunks)} processes...")
    start_time = time.time()

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
    removed_count = 0
    for term, coef in final_result.items():
        if abs(coef) > threshold:
            cleaned_result[term] = coef
        else:
            removed_count += 1

    print(f"Parallel calculation completed: {len(cleaned_result)} terms in {time.time() - start_time:.2f}s")
    print(f"Small coefficients removed after aggregation: {removed_count}")

    return polynomial_to_symengine(cleaned_result)


def split_variable_range(num_vars, num_chunks, max_memory_gb=2):
    """Split variable range into chunks"""
    max_memory_mb = max_memory_gb * 1024
    total_memory_mb = estimate_memory_usage(num_vars)

    print(f"Estimated memory usage: {total_memory_mb:.2f} MB")

    if total_memory_mb < max_memory_mb or num_vars < 1000:
        chunk_size = max(1, num_vars // num_chunks)
        chunks = []

        for i in range(0, num_vars, chunk_size):
            end_idx = min(i + chunk_size, num_vars)
            chunks.append((i, end_idx))

        return chunks

    chunks = []
    base = num_vars ** (1 / num_chunks)

    current = 0
    for i in range(num_chunks):
        if i == num_chunks - 1:
            next_bound = num_vars
        else:
            next_bound = int(base ** (i + 1))
            next_bound = min(next_bound, num_vars)

        chunks.append((current, next_bound))
        current = next_bound

        if current >= num_vars:
            break

    print("Chunk division:")
    for i, (start, end) in enumerate(chunks):
        vars_in_chunk = end - start
        print(f"Chunk {i}: variables {start} to {end - 1} ({vars_in_chunk} variables)")

    return chunks


def estimate_memory_usage(num_vars):
    """Estimate memory usage based on number of variables"""
    var_memory = num_vars * 50
    cross_terms = (num_vars * (num_vars - 1)) // 2
    cross_memory = cross_terms * 100
    total_mb = (var_memory + cross_memory) / (1024 * 1024)
    return total_mb


def polynomial_to_symengine_fast(result_dict):
    """Fast version: Convert dictionary result to symengine expression"""
    import symengine as se

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
                if round(coef) == 1:
                    coef_part = se.Integer(1)
                elif round(coef) == -1:
                    coef_part = se.Integer(-1)
                else:
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


def polynomial_to_symengine_batch(result_dict, batch_size=500):
    """Batch processing version: Memory efficient"""
    import symengine as se

    if not result_dict:
        return se.Integer(0)

    symbol_cache = {}

    def get_symbol(var_name):
        if var_name not in symbol_cache:
            symbol_cache[var_name] = se.Symbol(var_name)
        return symbol_cache[var_name]

    result_expr = se.Integer(0)
    items = list(result_dict.items())

    print(f"Batch processing started: {len(items)} terms in batches of {batch_size}")

    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_terms = []

        for term_str, coef in batch_items:
            if abs(coef) < 1e-12:
                continue

            if term_str == '1':
                if abs(coef - round(coef)) < 1e-10:
                    batch_terms.append(se.Integer(round(coef)))
                else:
                    batch_terms.append(se.Float(coef))
            else:
                if abs(coef - round(coef)) < 1e-10:
                    coef_part = se.Integer(round(coef))
                else:
                    coef_part = se.Float(coef)

                var_names = term_str.split('*')
                var_part = se.Integer(1)
                for var_name in var_names:
                    var_part *= get_symbol(var_name)

                batch_terms.append(coef_part * var_part)

        if batch_terms:
            batch_sum = sum(batch_terms)
            result_expr += batch_sum

        gc.collect()

        if (i // batch_size + 1) % 10 == 0:
            progress = (i + batch_size) / len(items) * 100
            print(f"Progress: {progress:.1f}% ({i + batch_size}/{len(items)} terms)")

    return result_expr


def polynomial_to_symengine(result_dict):
    """Select optimal method automatically"""
    num_terms = len(result_dict)

    print(f"polynomial_to_symengine: processing {num_terms} terms")

    if num_terms == 0:
        return se.Integer(0)
    elif num_terms < 500:
        print("Small scale: using fast version")
        return polynomial_to_symengine_fast(result_dict)
    else:
        print("Large scale: using batch processing version")
        return polynomial_to_symengine_batch(result_dict, batch_size=500)


# === Original functions ===
def efficient_expand(expr):
    """Efficient expansion: expand then simplify powers"""
    st = time.time()
    expanded = cached_expand(expr)
    ed = time.time()
    print("Expansion", ed - st)
    simplified = expanded
    print("Character conversion", time.time() - ed)
    return simplified


class Binary(Basic):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def _eval_power(self, other):
        return self


@functools.lru_cache(maxsize=512)
def cached_expand(expr):
    """Expansion with cache"""
    return se.expand(expr)


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


def simplify_binary_expression(expr):
    """Simplify only powers of binary variables in SymEngine expressions"""
    expr_str = str(expr)
    pattern = r'([a-zA-Z][a-zA-Z0-9_]*)\*\*[1-9][0-9]*'
    simplified_str = re.sub(pattern, r'\1', expr_str)
    result = se.sympify(simplified_str)
    return result


def get_cofactor_matrix(mad):
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
    """Efficiently calculate determinant of block matrix [[A, B], [C, D]]"""
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


# === Individual C Element Analysis Functions ===
def convert_single_element_to_amplify(expr, var_map):
    """Convert single SymEngine expression to Amplify expression"""
    try:
        amplify_expr = BinaryPoly()

        if expr.is_Add:
            terms = expr.args
        else:
            terms = [expr]

        for term in terms:
            coeff = 1.0
            vars_list = []

            if term.is_Mul:
                for factor in term.args:
                    if factor.is_Number:
                        coeff *= float(factor)
                    elif factor.is_Symbol:
                        vars_list.append(str(factor))
                    elif hasattr(factor, 'is_Pow') and factor.is_Pow:
                        vars_list.append(str(factor.args[0]))
            elif term.is_Symbol:
                vars_list.append(str(term))
            elif term.is_Number:
                coeff = float(term)

            if abs(coeff) < COEFFICIENT_THRESHOLD:
                continue

            if not vars_list:
                amplify_expr += coeff
            else:
                term_expr = coeff
                for var in vars_list:
                    if var in var_map:
                        term_expr *= var_map[var]
                amplify_expr += term_expr

        return amplify_expr

    except Exception as e:
        print(f"Amplify conversion error: {e}")
        return None


def analyze_individual_c_elements(C, sym):
    """Analyze each C element individually by QUBO conversion"""
    print("=== Individual C Element QUBO Conversion Analysis ===")

    gen = VariableGenerator()
    q = gen.array("Binary", len(sym))
    var_map = {str(sym[i]): q[i] for i in range(len(sym))}

    analysis_results = {
        'elements': [],
        'total_logical_vars_estimate': 0,
        'input_vars': len(sym),
        'element_count': len(C)
    }

    print(f"Total constraint elements: {len(C)}")
    print(f"Input variables: {len(sym)}")
    print("-" * 60)

    total_logical_vars = 0
    successful_conversions = 0

    for i, c_element in enumerate(C):
        print(f"\nAnalyzing C[{i}]...")

        element_result = {
            'index': i,
            'input_vars': 0,
            'logical_vars': 0,
            'degree': 0,
            'terms_count': 0,
            'conversion_successful': False,
            'error': None
        }

        try:
            element_result['degree'] = c_element.degree() if hasattr(c_element, 'degree') else 'Unknown'

            if c_element.is_Add:
                element_result['terms_count'] = len(c_element.args)
            else:
                element_result['terms_count'] = 1

            print(f"  Degree: {element_result['degree']}")
            print(f"  Terms: {element_result['terms_count']}")

            amplify_expr = convert_single_element_to_amplify(c_element, var_map)

            if amplify_expr is not None:
                try:
                    def model_timeout_handler(signum, frame):
                        raise TimeoutError("Model creation timeout")

                    signal.signal(signal.SIGALRM, model_timeout_handler)
                    signal.alarm(30)

                    model = Model(amplify_expr)
                    signal.alarm(0)

                    element_result['input_vars'] = model.num_input_vars

                    try:
                        signal.alarm(10)
                        element_result['logical_vars'] = model.num_logical_vars
                        signal.alarm(0)

                        element_result['conversion_successful'] = True
                        successful_conversions += 1
                        total_logical_vars += element_result['logical_vars']

                        print(f"  Input vars: {element_result['input_vars']}")
                        print(f"  Logical vars: {element_result['logical_vars']}")

                    except TimeoutError:
                        signal.alarm(0)
                        print(f"  Logical vars acquisition timeout")
                        element_result['error'] = "Logical vars timeout"

                        try:
                            signal.alarm(5)
                            logical_matrix, _ = model.logical_matrix
                            signal.alarm(0)
                            if logical_matrix is not None:
                                element_result['logical_vars'] = len(logical_matrix)
                                element_result['conversion_successful'] = True
                                total_logical_vars += element_result['logical_vars']
                                print(f"  Logical vars (estimated): {element_result['logical_vars']}")
                        except:
                            signal.alarm(0)
                            print(f"  Logical matrix estimation also failed")

                except TimeoutError:
                    signal.alarm(0)
                    print(f"  Model creation timeout")
                    element_result['error'] = "Model creation timeout"

            else:
                print(f"  Amplify conversion failed")
                element_result['error'] = "Amplify conversion failed"

        except Exception as e:
            signal.alarm(0)
            print(f"  Error: {e}")
            element_result['error'] = str(e)

        analysis_results['elements'].append(element_result)
        gc.collect()

    # Results aggregation and estimation
    print("\n" + "=" * 60)
    print("Analysis Results Summary")
    print("=" * 60)

    if successful_conversions > 0:
        successful_elements = [e for e in analysis_results['elements'] if e['conversion_successful']]

        logical_vars_per_element = [e['logical_vars'] for e in successful_elements]
        degrees = [e['degree'] for e in successful_elements if isinstance(e['degree'], int)]
        terms_counts = [e['terms_count'] for e in successful_elements]

        print(f"Successful conversions: {successful_conversions}/{len(C)}")
        print(f"Logical vars per element: {logical_vars_per_element}")
        print(f"Degrees: {degrees}")
        print(f"Terms counts: {terms_counts}")

        if successful_conversions == len(C):
            analysis_results['total_logical_vars_estimate'] = total_logical_vars
            print(f"\nAll elements conversion successful!")
            print(f"Logical variables (exact): {total_logical_vars}")
        else:
            avg_logical_vars = sum(logical_vars_per_element) / len(logical_vars_per_element)
            estimated_total = avg_logical_vars * len(C)
            analysis_results['total_logical_vars_estimate'] = int(estimated_total)

            print(f"\nPartial success estimation:")
            print(f"Average logical vars per element: {avg_logical_vars:.2f}")
            print(f"Estimated total logical vars: {int(estimated_total)}")

        print(f"\nDetailed statistics:")
        print(f"Input variables: {len(sym)}")
        print(f"Constraint elements: {len(C)}")
        print(f"Estimated logical variables: {analysis_results['total_logical_vars_estimate']}")
        if len(sym) > 0:
            expansion_ratio = analysis_results['total_logical_vars_estimate'] / len(sym)
            print(f"Expansion ratio: {expansion_ratio:.2f}x")

    else:
        print("All element conversions failed")
        analysis_results['total_logical_vars_estimate'] = "Estimation not possible"

    return analysis_results


def count_variables_in_expression(expr):
    """Count variables and their occurrences in expression"""
    var_count = defaultdict(int)

    try:
        expr_str = str(expr)

        # Regular expression patterns for typical variables
        patterns = [
            r't_\d+',  # t_21, t_34 etc
            r'u_[IVX]+[ABC]',  # u_IA, u_IIB etc
            r'MD_\d+',  # MD_0, MD_1 etc
            r'X_[IVX]+\d*',  # X_I0, X_I1 etc
            r'q_\d+',  # q_0, q_1 etc (Amplify variables)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, expr_str)
            for match in matches:
                var_count[match] += 1

    except Exception as e:
        print(f"Variable counting error: {e}")

    return var_count


def estimate_logical_vars_from_stats(stats):
    """Estimate logical variables from statistical information"""
    input_vars = stats['input_vars']
    max_degree = stats['max_degree']
    total_terms = stats['total_terms']
    high_degree_elements = stats['high_degree_elements']
    degree_dist = stats['degree_distribution']

    print(f"\nLogical variables estimation algorithm:")

    # Base logical variables (same as input variables)
    base_logical_vars = input_vars
    print(f"Base logical variables: {base_logical_vars}")

    # Degree correction
    degree_multiplier = 1.0
    if max_degree <= 2:
        degree_multiplier = 1.0
    elif max_degree <= 4:
        degree_multiplier = 1.5
    elif max_degree <= 8:
        degree_multiplier = 3.0
    elif max_degree <= 12:
        degree_multiplier = 6.0
    else:
        degree_multiplier = 10.0

    degree_adjusted = int(base_logical_vars * degree_multiplier)
    print(f"After degree correction: {degree_adjusted} (max degree {max_degree} x{degree_multiplier})")

    # High degree elements correction
    high_degree_bonus = high_degree_elements * input_vars * 0.1
    high_degree_adjusted = degree_adjusted + int(high_degree_bonus)
    print(f"After high degree correction: {high_degree_adjusted} (+{int(high_degree_bonus)})")

    # Terms count correction
    terms_factor = min(total_terms / 1000, 2.0)  # Maximum 2x
    final_estimate = int(high_degree_adjusted * (1 + terms_factor * 0.5))
    print(f"After terms correction: {final_estimate} (total terms {total_terms})")

    # Degree distribution refinement
    weighted_degree = 0
    total_elements = sum(degree_dist.values())
    for degree, count in degree_dist.items():
        weight = count / total_elements
        weighted_degree += degree * weight

    if weighted_degree > 6:
        complexity_bonus = int(final_estimate * 0.2)
        final_estimate += complexity_bonus
        print(f"After complexity correction: {final_estimate} (+{complexity_bonus}, avg degree {weighted_degree:.1f})")

    print(f"\nFinal estimate: {final_estimate}")

    return final_estimate


def lightweight_logical_vars_estimation(C, sym):
    """Lightweight version: estimate logical variables from structure analysis only"""
    print("=== Lightweight Logical Variables Estimation ===")

    stats = {
        'input_vars': len(sym),
        'constraint_elements': len(C),
        'total_terms': 0,
        'degree_distribution': defaultdict(int),
        'max_degree': 0,
        'high_degree_elements': 0,
        'variable_usage': defaultdict(int)
    }

    print(f"Input variables: {stats['input_vars']}")
    print(f"Constraint elements: {stats['constraint_elements']}")
    print("-" * 50)

    # Analyze each C element (memory efficient)
    for i, c_element in enumerate(C):
        try:
            if hasattr(c_element, 'degree'):
                degree = c_element.degree()
                stats['degree_distribution'][degree] += 1
                stats['max_degree'] = max(stats['max_degree'], degree)

                if degree > 4:  # Count high degree elements
                    stats['high_degree_elements'] += 1

            # Count terms
            if c_element.is_Add:
                terms_count = len(c_element.args)
            else:
                terms_count = 1

            stats['total_terms'] += terms_count

            # Variable usage analysis (sampling)
            if i < 5:  # Detailed analysis for first 5 elements only
                var_count = count_variables_in_expression(c_element)
                for var, count in var_count.items():
                    stats['variable_usage'][var] += count

            # Progress display
            if (i + 1) % 10 == 0:
                print(f"Analysis progress: {i + 1}/{len(C)}")

        except Exception as e:
            print(f"C[{i}] analysis error: {e}")

    # Display statistics
    print(f"\nAnalysis results:")
    print(f"Total terms: {stats['total_terms']}")
    print(f"Max degree: {stats['max_degree']}")
    print(f"Degree distribution: {dict(stats['degree_distribution'])}")
    print(f"High degree elements (>4th): {stats['high_degree_elements']}")

    # Logical variables estimation algorithm
    logical_vars_estimate = estimate_logical_vars_from_stats(stats)

    return logical_vars_estimate, stats


def evaluate_estimation_confidence(stats):
    """Evaluate confidence of estimation"""
    confidence = 70  # Base score

    # Confidence adjustment based on max degree
    if stats['max_degree'] <= 4:
        confidence += 20
    elif stats['max_degree'] <= 8:
        confidence += 10
    else:
        confidence -= 10

    # Adjustment based on constraint elements count
    if stats['constraint_elements'] >= 10:
        confidence += 10

    # Adjustment based on high degree elements ratio
    if stats['constraint_elements'] > 0:
        high_degree_ratio = stats['high_degree_elements'] / stats['constraint_elements']
        if high_degree_ratio < 0.3:
            confidence += 10
        elif high_degree_ratio > 0.7:
            confidence -= 15

    return min(max(confidence, 30), 95)  # Clamp to 30-95 range


def quick_estimate_from_structure(C, sym):
    """Quick estimation from structure analysis"""
    print("\n=== Quick Estimation from Structure Analysis ===")

    total_terms = 0
    max_degree = 0
    degree_distribution = defaultdict(int)

    for i, c_element in enumerate(C):
        try:
            degree = c_element.degree() if hasattr(c_element, 'degree') else 0
            max_degree = max(max_degree, degree)
            degree_distribution[degree] += 1

            if c_element.is_Add:
                terms = len(c_element.args)
            else:
                terms = 1

            total_terms += terms

        except Exception as e:
            print(f"C[{i}] analysis error: {e}")

    print(f"Total terms: {total_terms}")
    print(f"Max degree: {max_degree}")
    print(f"Degree distribution: {dict(degree_distribution)}")

    # Empirical estimation formula
    input_vars = len(sym)

    if max_degree <= 2:
        estimated_logical = input_vars
    elif max_degree <= 4:
        estimated_logical = input_vars * 2
    elif max_degree <= 8:
        estimated_logical = input_vars * 4
    else:
        estimated_logical = input_vars * 8

    # Terms count correction
    term_factor = min(total_terms / 1000, 3.0)  # Maximum 3x
    estimated_logical = int(estimated_logical * (1 + term_factor))

    print(f"Structure analysis estimated logical variables: {estimated_logical}")

    return estimated_logical


def run_quick_estimation():
    """Execute quick estimation using existing C, sym"""
    global C, sym

    if 'C' not in globals() or 'sym' not in globals():
        print("C (constraints) or sym (variables) not defined")
        return None

    print("Running lightweight logical variables estimation...")
    start_time = time.time()

    estimated_vars, analysis_stats = lightweight_logical_vars_estimation(C, sym)

    end_time = time.time()

    print(f"\n" + "=" * 60)
    print("Estimation Results Summary")
    print("=" * 60)
    print(f"Input variables: {analysis_stats['input_vars']}")
    print(f"Constraint elements: {analysis_stats['constraint_elements']}")
    print(f"Estimated logical variables: {estimated_vars}")
    print(f"Expansion ratio: {estimated_vars / analysis_stats['input_vars']:.2f}x")
    print(f"Processing time: {end_time - start_time:.2f}s")

    # Confidence evaluation
    confidence_score = evaluate_estimation_confidence(analysis_stats)
    print(f"Estimation confidence: {confidence_score}/100")

    return estimated_vars


def main_individual_analysis():
    """Main function to execute individual analysis"""
    global C, sym

    if 'C' not in globals() or 'sym' not in globals():
        print("C (constraints) or sym (variables) not defined")
        return None

    print("Starting logical variables estimation by individual C element analysis...")

    # Method 1: Quick estimation from structure analysis
    structure_estimate = quick_estimate_from_structure(C, sym)

    # Method 2: Detailed analysis by individual QUBO conversion
    detailed_results = analyze_individual_c_elements(C, sym)

    # Results comparison
    print("\n" + "=" * 60)
    print("Final Results Comparison")
    print("=" * 60)
    print(f"Input variables: {len(sym)}")
    print(f"Constraint elements: {len(C)}")
    print(f"Structure analysis estimate: {structure_estimate}")
    print(f"Individual conversion estimate: {detailed_results['total_logical_vars_estimate']}")

    # Determine final estimate
    if isinstance(detailed_results['total_logical_vars_estimate'], int):
        final_estimate = detailed_results['total_logical_vars_estimate']
        confidence = "High"
    else:
        final_estimate = structure_estimate
        confidence = "Medium"

    print(f"\nFinal estimated logical variables: {final_estimate} (Confidence: {confidence})")

    if len(sym) > 0:
        expansion_ratio = final_estimate / len(sym)
        print(f"Expansion ratio: {expansion_ratio:.2f}x")

    return final_estimate, detailed_results


# === Main execution starts here ===
from func_218 import symb

total_time = time.time()

Species = ["A", "B", "C"]
numcmp = 1

num = 2
MDcmp = numcmp
cmp = numcmp
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
# P: P11, P41 ...
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

# Calculate cofactor matrix
print("Starting cofactor matrix calculation...")
start_time = time.time()
Mad = get_cofactor_matrix(E - MP)
end_time = time.time()
print(f"Cofactor matrix calculation completed: {end_time - start_time:.2f}s")

# Simplify each element
print("Simplifying each element of cofactor matrix...")
start_time = time.time()
for i in range(len(Mad[0, :])):
    for j in range(len(Mad[:, 0])):
        Mad[i, j] = simplify_binary_expression(Mad[i, j])
end_time = time.time()
print(f"Cofactor matrix simplification completed: {end_time - start_time:.2f}s")

# Calculate and simplify determinant
print("Calculating determinant...")
start_time = time.time()
Mdet = efficient_block_det(efficient_expand(E3 - P[0][0:3, 0:3]), efficient_expand(P[0][0:3, 3:6]),
                           efficient_expand(P[1][0:3, 0:3]), efficient_expand(E3 - P[1][0:3, 3:6]))
end_time = time.time()
Mdet = efficient_expand(Mdet)
Mdet = simplify_binary_expression(Mdet)
print(f"Determinant calculation completed: {end_time - start_time:.2f}s")

MDfin = 1
MDET = 0
for i in range(MDcmp):
    MDET += MDfin / (2 ** (MDcmp) - 1) * 2 ** (i) * MD_list[i]

# Calculate inverse matrix
print("Calculating inverse matrix...")
start_time = time.time()
Minv = Mad * MDET
for i in range(len(Minv[0, :])):
    for j in range(len(Minv[:, 0])):
        Minv[i, j] = efficient_expand(Minv[i, j])
end_time = time.time()
print(f"Inverse matrix calculation completed: {end_time - start_time:.2f}s")

F = se.zeros(len(Species), 3 * (num + 1))
# Determine flow rates
F_list = [100] * len(Species)
for i in range(len(Species)):
    F[i, 0] = F_list[i]

FF = F[:, 0].col_join(se.zeros(3 * (num - 1), 1))

print(FF)

# Flow calculation
print("Calculating flows...")
start_time = time.time()
FT = Minv * FF
FT = efficient_expand(FT)
for n in range(num):
    F[:, 1 + 3 * n] = FT[3 * n: 3 * n + len(Species), 0]
    F[:, 1 + 3 * n + 1] = M[n] * F[:, 1 + 3 * n]
    F[:, 1 + 3 * n + 2] = (EYE - M[n]) * F[:, 1 + 3 * n]
end_time = time.time()
print(f"Flow calculation completed: {end_time - start_time:.2f}s")

# Expansion
F = efficient_expand(F)

C = []
Vl = [100, 50, 10]
X = []
for n in range(num):
    X.append(sum(F[:, 3 * n + 2]))

Xfin = [0.1, 0.1, 0.1]

# Expression calculation with memory-efficient square
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
        F0 = memory_efficient_square(F0, num_processes=16, threshold=1e-6)
        print(f"Parallel calculation time: {time.time() - st:.3f}s")
        Fsubs.append(F0)

    D = -(sym[-2 * (num - n) - MDcmp - num * cmp] * (
            Vl[0] * sym[3 * n] * Fsubs[0] + Vl[1] * sym[3 * n + 1] * Fsubs[1] + Vl[2] * sym[
        3 * n + 2] * Fsubs[2])) * X_sum
    C.append(efficient_expand(D))
end_time = time.time()
print(f"Expression calculation completed: {end_time - start_time:.2f}s")

# Constraints
print("Calculating constraint conditions...")
start_time = time.time()
for i in range(int(2 * num)):
    D = ((sum(t_list[i::2 * num]) - 1) ** 2)
    C.append(simplify_binary_expression(efficient_expand(D)))

# Recycle-related constraints
D = efficient_expand(Mdet * MDET - 1)
D = simplify_binary_expression(D)
D = memory_efficient_square(D, num_processes=16, threshold=1e-6)
D = simplify_binary_expression(D)
C.append(D)

# Fraction-related constraints
for n in range(num):
    X_sum = 0.005
    for i in range(cmp):
        X_sum += Xfin[n] / (2 ** (cmp) - 1) * 2 ** (i) * X_list[i + cmp * n]
    CSD = sym[-2 * (num - n) - MDcmp - num * cmp] * (X_sum * X[n] - 1)
    if int(n) == int(1):
        CSD = CSD * (t_24 + t_34 - t_24 * t_34)
    CSD = efficient_expand(CSD)
    CSD = simplify_binary_expression(CSD)
    D = memory_efficient_square(CSD, num_processes=16, threshold=1e-6)
    C.append(D)
end_time = time.time()
print(f"Constraint condition calculation completed: {end_time - start_time:.2f}s")

print("hello")
# Separator selection constraints
C.append(simplify_binary_expression(efficient_expand((u_IA + u_IB + u_IC - 1) ** 2)))
C.append(simplify_binary_expression(efficient_expand((u_IIA + u_IIB + u_IIC - 1) ** 2)))

for i in range(len(C) - num):
    C[i + num] = efficient_expand(C[i + num] * 10 ** 6)

print(len(sym))
print("Total time:", (time.time() - total_time) / 60, "minutes")
print("Formulation and optimization completed")
#
# === Combined Objective Function Analysis ===
print("\n" + "=" * 60)
print("COMBINED OBJECTIVE FUNCTION ANALYSIS")
print("=" * 60)

# Create combined objective function
print("Creating combined objective function from all C elements...")
objective_function = 0
for i, c in enumerate(C):
    print(f"Adding C[{i}] to objective function...")
    objective_function += c

print("Final expansion of combined objective function...")
objective_function = efficient_expand(objective_function)

# Analyze combined objective function
print("\n=== Combined Objective Function QUBO Analysis ===")

try:
    # Basic information
    if objective_function.is_Add:
        total_terms = len(objective_function.args)
    else:
        total_terms = 1

    try:
        degree = objective_function.degree()
    except:
        degree = 'Unknown'

    print(f"Total terms in combined objective: {total_terms}")
    print(f"Degree of combined objective: {degree}")
    print(f"Input variables: {len(sym)}")

    # Create Amplify variable mapping
    print("Creating Amplify variable mapping...")
    gen = VariableGenerator()
    q = gen.array("Binary", len(sym))
    var_map = {str(sym[i]): q[i] for i in range(len(sym))}

    # Convert to Amplify expression
    print("Converting to Amplify expression...")
    amplify_expr = BinaryPoly()

    if objective_function.is_Add:
        terms = list(objective_function.args)
    else:
        terms = [objective_function]

    chunk_size = 1000
    for i in range(0, len(terms), chunk_size):
        chunk = terms[i:min(i + chunk_size, len(terms))]
        print(f"Processing chunk {i // chunk_size + 1}/{(len(terms) + chunk_size - 1) // chunk_size}")

        for term in chunk:
            coeff = 1.0
            vars_list = []

            if term.is_Mul:
                for factor in term.args:
                    if factor.is_Number:
                        coeff *= float(factor)
                    elif factor.is_Symbol:
                        vars_list.append(str(factor))
                    elif hasattr(factor, 'is_Pow') and factor.is_Pow:
                        vars_list.append(str(factor.args[0]))
            elif term.is_Symbol:
                vars_list.append(str(term))
            elif term.is_Number:
                coeff = float(term)

            if abs(coeff) < COEFFICIENT_THRESHOLD:
                continue

            if not vars_list:
                amplify_expr += coeff
            else:
                term_expr = coeff
                for var in vars_list:
                    if var in var_map:
                        term_expr *= var_map[var]
                amplify_expr += term_expr

        gc.collect()

    print("Amplify expression created successfully")

    # Create QUBO model
    print("Creating BinaryQuadraticModel (this may take a long time)...")


    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout")


    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3600)  # 1 hour timeout

    model = Model(amplify_expr)
    signal.alarm(0)



    # input_vars = model.num_input_vars
    print(f"Model created successfully")
    # print(f"Input variables: {input_vars}")

    # Get logical variables
    print("Getting logical variables count (this is the final step)...")
    signal.alarm(72000)  # 2 hour timeout
    op = time.time()
    # logical_vars = model.num_logical_vars
    bq = AcceptableDegrees(objective={"Binary": "Quadratic"})
    im, mapping = model.to_intermediate_model(bq,
                                              quadratization_method="Substitute",
                                              substitution_multiplier=0.5)
    print(time.time() - op, "s for getting num_var")
    signal.alarm(0)
    print("num_var", len(im.variables))


    # print(f"SUCCESS: {logical_vars:,} logical variables")
    # print(f"Expansion ratio: {logical_vars / input_vars:.2f}x")
    #
    final_logical_vars = len(im.variables)
except TimeoutError:
    signal.alarm(0)
    print("Analysis timed out. Using fallback estimation...")
    # Fallback based on previous individual results
    final_logical_vars = 500000  # Conservative estimate

except Exception as e:
    signal.alarm(0)
    print(f"Analysis failed: {e}")
    final_logical_vars = 500000  # Conservative estimate

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Input variables: {len(sym)}")
print(f"Final logical variables: {final_logical_vars:,}")
print(f"Expansion ratio: {final_logical_vars / len(sym):.2f}x")

total_end_time = time.time()
print(f"\nTotal execution time: {(total_end_time - total_time) / 60:.2f} minutes")

try:
    process = psutil.Process()
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.1f} MB")
except:
    pass

print("Combined analysis completed")
