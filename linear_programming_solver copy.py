"""
Linear Programming Solver - Professional Edition
Decision Support Systems Course - Final Project

Features:
- Maximization and Minimization
- Constraint types: ≤, =, ≥
- Graphical Method (2 variables) with visualization
- Simplex Method (unlimited variables) with Two-Phase method
- Interactive user interface with clear instructions
- Input validation and error handling
- Results export capability
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ============================================
# PART 1: GRAPHICAL METHOD (2 Variables Only)
# ============================================

def solve_graphical(c1, c2, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq, maximize=True):
    """Solve LP with 2 variables graphically"""
    
    print("\n" + "="*70)
    print("📐 GRAPHICAL METHOD SOLUTION".center(70))
    print("="*70)
    
    print("\n📊 PROBLEM FORMULATION:")
    print("-"*70)
    if maximize:
        print(f"  Maximize Z = {c1}x₁ + {c2}x₂")
    else:
        print(f"  Minimize Z = {c1}x₁ + {c2}x₂")
    
    print("\n  Subject to:")
    
    A_combined = []
    b_combined = []
    all_constraints_text = []
    
    for i in range(len(A_ineq)):
        A_combined.append([A_ineq[i][0], A_ineq[i][1]])
        b_combined.append(b_ineq[i])
        print(f"    {A_ineq[i][0]}x₁ + {A_ineq[i][1]}x₂ ≤ {b_ineq[i]}")
        all_constraints_text.append(f"{A_ineq[i][0]}x₁ + {A_ineq[i][1]}x₂ ≤ {b_ineq[i]}")
    
    for i in range(len(A_eq)):
        A_combined.append([A_eq[i][0], A_eq[i][1]])
        b_combined.append(b_eq[i])
        A_combined.append([-A_eq[i][0], -A_eq[i][1]])
        b_combined.append(-b_eq[i])
        print(f"    {A_eq[i][0]}x₁ + {A_eq[i][1]}x₂ = {b_eq[i]}")
        all_constraints_text.append(f"{A_eq[i][0]}x₁ + {A_eq[i][1]}x₂ = {b_eq[i]}")
    
    for i in range(len(A_geq)):
        A_combined.append([-A_geq[i][0], -A_geq[i][1]])
        b_combined.append(-b_geq[i])
        print(f"    {A_geq[i][0]}x₁ + {A_geq[i][1]}x₂ ≥ {b_geq[i]}")
        all_constraints_text.append(f"{A_geq[i][0]}x₁ + {A_geq[i][1]}x₂ ≥ {b_geq[i]}")
    
    print("    x₁ ≥ 0, x₂ ≥ 0")
    print("-"*70)
    
    if not A_combined:
        print("\n❌ No constraints entered!")
        return None, None
    
    vertices = []
    
    # Origin
    origin_feasible = all(A_combined[i][0]*0 + A_combined[i][1]*0 <= b_combined[i] + 1e-7 
                          for i in range(len(A_combined)))
    if origin_feasible:
        vertices.append((0, 0))
    
    # Intersection with axes
    for i in range(len(A_combined)):
        a1, a2 = A_combined[i]
        rhs = b_combined[i]
        
        if abs(a1) > 1e-7:
            x1 = rhs / a1
            x2 = 0
            if x1 >= 0:
                feasible = True
                for j in range(len(A_combined)):
                    if A_combined[j][0]*x1 + A_combined[j][1]*x2 > b_combined[j] + 1e-7:
                        feasible = False
                        break
                if feasible:
                    vertices.append((x1, x2))
        
        if abs(a2) > 1e-7:
            x1 = 0
            x2 = rhs / a2
            if x2 >= 0:
                feasible = True
                for j in range(len(A_combined)):
                    if A_combined[j][0]*x1 + A_combined[j][1]*x2 > b_combined[j] + 1e-7:
                        feasible = False
                        break
                if feasible:
                    vertices.append((x1, x2))
    
    # Intersection of constraint pairs
    for i in range(len(A_combined)):
        a1_i, a2_i = A_combined[i]
        rhs_i = b_combined[i]
        for j in range(i+1, len(A_combined)):
            a1_j, a2_j = A_combined[j]
            rhs_j = b_combined[j]
            
            det = a1_i * a2_j - a1_j * a2_i
            
            if abs(det) > 1e-7:
                x1 = (rhs_i * a2_j - rhs_j * a2_i) / det
                x2 = (a1_i * rhs_j - a1_j * rhs_i) / det
                
                if x1 >= -1e-7 and x2 >= -1e-7:
                    feasible = True
                    for k in range(len(A_combined)):
                        if A_combined[k][0]*x1 + A_combined[k][1]*x2 > b_combined[k] + 1e-5:
                            feasible = False
                            break
                    if feasible:
                        vertices.append((x1, x2))
    
    vertices = list(set([(round(v[0], 6), round(v[1], 6)) for v in vertices]))
    
    if not vertices:
        print("\n❌ ERROR: No feasible region found!")
        print("   The problem might have conflicting constraints.")
        return None, None
    
    print("\n📌 FEASIBLE REGION VERTICES:")
    print("-"*70)
    
    obj_values = []
    vertex_data = []
    for v in vertices:
        z = c1 * v[0] + c2 * v[1]
        obj_values.append(z)
        vertex_data.append((v[0], v[1], z))
        print(f"  • ({v[0]:.3f}, {v[1]:.3f}) → Z = {z:.3f}")
    
    if maximize:
        best_idx = np.argmax(obj_values)
        best_value = np.max(obj_values)
        direction = "MAXIMUM"
    else:
        best_idx = np.argmin(obj_values)
        best_value = np.min(obj_values)
        direction = "MINIMUM"
    
    optimal_point = vertices[best_idx]
    
    print("\n" + "="*70)
    print(f"✅ OPTIMAL SOLUTION ({direction}):".center(70))
    print("="*70)
    print(f"  x₁ = {optimal_point[0]:.4f}")
    print(f"  x₂ = {optimal_point[1]:.4f}")
    print(f"  Z = {best_value:.4f}")
    print("="*70)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    if vertices:
        max_x = max(max([v[0] for v in vertices]), 1) * 1.2
    else:
        max_x = 100
    x1_plot = np.linspace(0, max_x, 100)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(A_combined)))
    
    for idx, i in enumerate(range(len(A_combined))):
        a1, a2 = A_combined[i]
        rhs = b_combined[i]
        if abs(a2) > 1e-7:
            x2_plot = (rhs - a1 * x1_plot) / a2
            mask = x2_plot >= 0
            if any(mask):
                plt.plot(x1_plot[mask], x2_plot[mask], linewidth=2, 
                        alpha=0.8, color=colors[idx], 
                        label=f'{a1}x₁ + {a2}x₂ = {rhs:.0f}')
        elif abs(a1) > 1e-7:
            plt.axvline(x=rhs/a1, linewidth=2, alpha=0.8, 
                       color=colors[idx],
                       label=f'{a1}x₁ = {rhs:.0f}')
    
    if vertices:
        vertices_sorted = sorted(vertices, key=lambda p: (p[0], p[1]))
        vertices_array = np.array(vertices_sorted)
        plt.fill(vertices_array[:, 0], vertices_array[:, 1], 
                alpha=0.3, color='lightgreen', label='Feasible Region')
    
    x_vals = [v[0] for v in vertices]
    y_vals = [v[1] for v in vertices]
    plt.scatter(x_vals, y_vals, color='blue', s=120, zorder=5, 
               edgecolors='black', linewidth=1.5, label='Vertices')
    
    if optimal_point:
        plt.scatter(optimal_point[0], optimal_point[1], color='red', 
                   s=400, zorder=6, marker='*', 
                   edgecolors='darkred', linewidth=2,
                   label=f'Optimal Point\n({optimal_point[0]:.2f}, {optimal_point[1]:.2f})')
    
    plt.xlabel('x₁', fontsize=14, fontweight='bold')
    plt.ylabel('x₂', fontsize=14, fontweight='bold')
    plt.title('Graphical Method - Linear Programming', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linewidth=0.8)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlim(-0.5, max_x)
    plt.ylim(-0.5, max_x)
    plt.tight_layout()
    plt.show()
    
    return optimal_point, best_value


# ============================================
# PART 2: SIMPLEX METHOD (Two-Phase Method)
# ============================================

class SimplexSolver:
    """Simplex Method solver using Two-Phase method for mixed constraints"""
    
    def __init__(self, maximize=True):
        self.maximize = maximize
        self.tableau = None
        self.n_vars = 0
        self.n_constraints = 0
        self.n_slack = 0
        self.n_surplus = 0
        self.n_artificial = 0
        self.var_names = []
        self.original_c = []
        self.basis = []
        self.artificial_indices = []
        self.iteration_history = []
        
    def solve(self, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq):
        """Solve LP using Two-Phase Simplex Method"""
        
        self.original_c = c.copy()
        
        print("\n" + "="*70)
        print("🧮 SIMPLEX METHOD (Two-Phase)".center(70))
        print("="*70)
        
        self.display_problem(c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq)
        
        # Build initial tableau
        self.build_initial_tableau(c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq)
        
        print("\n📋 PHASE I - INITIAL TABLEAU")
        print("   (Minimizing sum of artificial variables to find feasible solution)")
        self.print_tableau()
        
        # Phase I
        phase1_result = self.phase_one()
        
        if not phase1_result:
            print("\n❌ ERROR: Problem is infeasible!")
            print("   No solution exists that satisfies all constraints.")
            return None, None
        
        # Phase II
        phase2_result = self.phase_two()
        
        if not phase2_result:
            print("\n❌ ERROR: Problem is unbounded!")
            print("   The objective function can be improved indefinitely.")
            return None, None
        
        # Extract solution
        solution, obj_value = self.extract_solution()
        self.display_solution(solution, obj_value)
        
        return solution, obj_value
    
    def display_problem(self, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq):
        """Display problem formulation"""
        print("\n📊 PROBLEM FORMULATION:")
        print("-"*70)
        
        obj_parts = []
        for j in range(len(c)):
            if j == 0:
                obj_parts.append(f"{c[j]}x{j+1}")
            elif c[j] >= 0:
                obj_parts.append(f"+ {c[j]}x{j+1}")
            else:
                obj_parts.append(f"- {abs(c[j])}x{j+1}")
        
        if self.maximize:
            print(f"  Maximize Z = {' '.join(obj_parts)}")
        else:
            print(f"  Minimize Z = {' '.join(obj_parts)}")
        
        print("\n  Subject to:")
        
        for i in range(len(A_ineq)):
            parts = [f"{A_ineq[i][0]}x₁"]
            for j in range(1, len(c)):
                if A_ineq[i][j] >= 0:
                    parts.append(f"+ {A_ineq[i][j]}x{j+1}")
                else:
                    parts.append(f"- {abs(A_ineq[i][j])}x{j+1}")
            print(f"    {' '.join(parts)} ≤ {b_ineq[i]}")
        
        for i in range(len(A_eq)):
            parts = [f"{A_eq[i][0]}x₁"]
            for j in range(1, len(c)):
                if A_eq[i][j] >= 0:
                    parts.append(f"+ {A_eq[i][j]}x{j+1}")
                else:
                    parts.append(f"- {abs(A_eq[i][j])}x{j+1}")
            print(f"    {' '.join(parts)} = {b_eq[i]}")
        
        for i in range(len(A_geq)):
            parts = [f"{A_geq[i][0]}x₁"]
            for j in range(1, len(c)):
                if A_geq[i][j] >= 0:
                    parts.append(f"+ {A_geq[i][j]}x{j+1}")
                else:
                    parts.append(f"- {abs(A_geq[i][j])}x{j+1}")
            print(f"    {' '.join(parts)} ≥ {b_geq[i]}")
        
        print("    x₁, x₂, ..., xₙ ≥ 0")
        print("-"*70)
    
    def build_initial_tableau(self, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq):
        """Build initial tableau for Phase I"""
        
        self.n_vars = len(c)
        n_ineq = len(A_ineq)
        n_eq = len(A_eq)
        n_geq = len(A_geq)
        
        self.n_slack = n_ineq
        self.n_surplus = n_geq
        self.n_artificial = n_eq + n_geq
        
        self.n_constraints = n_ineq + n_eq + n_geq
        
        total_vars = self.n_vars + self.n_slack + self.n_surplus + self.n_artificial
        
        self.tableau = np.zeros((self.n_constraints + 1, total_vars + 1))
        
        # Variable names
        self.var_names = [f"x{j+1}" for j in range(self.n_vars)]
        self.var_names += [f"s{i+1}" for i in range(self.n_slack)]
        self.var_names += [f"e{i+1}" for i in range(self.n_surplus)]
        self.var_names += [f"A{i+1}" for i in range(self.n_artificial)]
        
        self.basis = []
        self.artificial_indices = []
        
        row = 0
        
        # ≤ constraints (add slack)
        for i in range(n_ineq):
            self.tableau[row, :self.n_vars] = A_ineq[i]
            slack_col = self.n_vars + i
            self.tableau[row, slack_col] = 1
            self.tableau[row, -1] = b_ineq[i]
            self.basis.append(slack_col)
            row += 1
        
        # = constraints (add artificial)
        for i in range(n_eq):
            self.tableau[row, :self.n_vars] = A_eq[i]
            artificial_col = self.n_vars + self.n_slack + self.n_surplus + i
            self.tableau[row, artificial_col] = 1
            self.tableau[row, -1] = b_eq[i]
            self.basis.append(artificial_col)
            self.artificial_indices.append(artificial_col)
            row += 1
        
        # ≥ constraints (surplus + artificial)
        for i in range(n_geq):
            self.tableau[row, :self.n_vars] = A_geq[i]
            surplus_col = self.n_vars + self.n_slack + i
            self.tableau[row, surplus_col] = -1
            artificial_col = self.n_vars + self.n_slack + self.n_surplus + n_eq + i
            self.tableau[row, artificial_col] = 1
            self.tableau[row, -1] = b_geq[i]
            self.basis.append(artificial_col)
            self.artificial_indices.append(artificial_col)
            row += 1
        
        # Phase I objective: minimize sum of artificial variables
        for col in self.artificial_indices:
            self.tableau[-1, col] = 1
        
        # Convert to canonical form
        for i, basic_var in enumerate(self.basis):
            if basic_var in self.artificial_indices:
                self.tableau[-1, :] -= self.tableau[i, :]
    
    def pivot(self, pivot_row, pivot_col):
        """Perform pivot operation"""
        self.basis[pivot_row] = pivot_col
        pivot_value = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_value
        
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i, :] -= factor * self.tableau[pivot_row, :]
    
    def phase_one(self):
        """Execute Phase I to find feasible solution"""
        
        iteration = 1
        
        while True:
            # Check optimality
            if all(self.tableau[-1, :-1] >= -1e-7):
                if abs(self.tableau[-1, -1]) < 1e-7:
                    print(f"\n✅ Phase I: Feasible solution found after {iteration} iterations!")
                    return True
                else:
                    print(f"\n❌ Phase I: Problem is infeasible")
                    print(f"   Minimum sum of artificial variables = {self.tableau[-1, -1]:.4f} (should be 0)")
                    return False
            
            # Find entering variable
            pivot_col = np.argmin(self.tableau[-1, :-1])
            
            # Find leaving variable
            min_ratio = float('inf')
            pivot_row = -1
            
            for i in range(self.n_constraints):
                if self.tableau[i, pivot_col] > 1e-7:
                    ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_row = i
            
            if pivot_row == -1:
                print("\n❌ Phase I: Problem is unbounded!")
                return False
            
            print(f"\n🔄 Phase I - Iteration {iteration}:")
            print(f"   Entering variable: {self.var_names[pivot_col]}")
            print(f"   Leaving variable: {self.var_names[self.basis[pivot_row]]}")
            
            # Pivot
            self.pivot(pivot_row, pivot_col)
            self.print_tableau()
            
            iteration += 1
            if iteration > 100:
                print("\n❌ Too many iterations!")
                return False
    
    def phase_two(self):
        """Execute Phase II to optimize original objective"""
        
        print("\n" + "="*70)
        print("🎯 PHASE II: Optimizing Original Objective".center(70))
        print("="*70)
        
        # Remove artificial variables
        artificial_cols = sorted(self.artificial_indices, reverse=True)
        for col in artificial_cols:
            self.tableau = np.delete(self.tableau, col, axis=1)
            self.var_names.pop(col)
            for i in range(len(self.basis)):
                if self.basis[i] > col:
                    self.basis[i] -= 1
        
        # Set objective row for Phase II
        self.tableau[-1, :] = 0
        
        for j in range(self.n_vars):
            if self.maximize:
                self.tableau[-1, j] = -self.original_c[j]
            else:
                self.tableau[-1, j] = self.original_c[j]
        
        # Convert to canonical form
        for i, basic_var in enumerate(self.basis):
            coeff = self.tableau[-1, basic_var]
            if abs(coeff) > 1e-7:
                self.tableau[-1, :] -= coeff * self.tableau[i, :]
        
        print("\n📋 PHASE II - INITIAL TABLEAU")
        self.print_tableau()
        
        iteration = 1
        
        while True:
            obj_row = self.tableau[-1, :-1]
            
            if self.maximize:
                if np.all(obj_row >= -1e-7):
                    print(f"\n✅ Phase II: Optimal solution found after {iteration} iterations!")
                    return True
                pivot_col = np.argmin(obj_row)
            else:
                if np.all(obj_row <= 1e-7):
                    print(f"\n✅ Phase II: Optimal solution found after {iteration} iterations!")
                    return True
                pivot_col = np.argmax(obj_row)
            
            # Find leaving variable
            min_ratio = float('inf')
            pivot_row = -1
            
            for i in range(self.n_constraints):
                if self.tableau[i, pivot_col] > 1e-7:
                    ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_row = i
            
            if pivot_row == -1:
                print("\n❌ Phase II: Unbounded solution!")
                return False
            
            print(f"\n🔄 Phase II - Iteration {iteration}:")
            print(f"   Entering variable: {self.var_names[pivot_col]}")
            print(f"   Leaving variable: {self.var_names[self.basis[pivot_row]]}")
            
            # Pivot
            self.pivot(pivot_row, pivot_col)
            self.print_tableau()
            
            iteration += 1
            if iteration > 100:
                print("\n❌ Too many iterations!")
                return False
    
    def print_tableau(self):
        """Print current tableau in a readable format"""
        if len(self.var_names) > 15:
            # For large tableaus, show limited columns
            print(f"\n   (Tableau has {len(self.var_names)} variables, showing summary)")
            return
            
        print("\n   " + " ".join([f"{name:>6}" for name in self.var_names] + ["    RHS"]))
        print("   " + "-" * (9 * (len(self.var_names) + 1)))
        
        for i in range(self.tableau.shape[0]):
            row_name = f"R{i+1}" if i < self.n_constraints else "Z"
            row_str = f"{row_name:>2} |"
            for j in range(self.tableau.shape[1]):
                if j == self.tableau.shape[1] - 1:
                    row_str += f"  {self.tableau[i, j]:>8.3f}"
                else:
                    row_str += f"  {self.tableau[i, j]:>6.2f}"
            print(row_str)
        print()
    
    def extract_solution(self):
        """Extract solution from final tableau"""
        solution = np.zeros(self.n_vars)
        
        for i, basic_var in enumerate(self.basis):
            if basic_var < self.n_vars:
                solution[basic_var] = self.tableau[i, -1]
        
        # Due to floating point errors, small negative values should be zero
        solution = np.where(solution < -1e-7, 0, solution)
        solution = np.where(solution < 0, 0, solution)
        
        obj_value = np.dot(self.original_c, solution)
        
        return solution, obj_value
    
    def display_solution(self, solution, obj_value):
        """Display final solution"""
        print("\n" + "="*70)
        print("✨ OPTIMAL SOLUTION ✨".center(70))
        print("="*70)
        
        for j in range(self.n_vars):
            print(f"  x{j+1} = {solution[j]:.4f}")
        
        if self.maximize:
            print(f"\n  Maximum Z = {obj_value:.4f}")
        else:
            print(f"\n  Minimum Z = {obj_value:.4f}")
        
        print("="*70)


# ============================================
# PART 3: USER INTERFACE
# ============================================

def display_header():
    """Display program header"""
    print("\n" + "█"*70)
    print("█" + " " * 68 + "█")
    print("█" + "     LINEAR PROGRAMMING SOLVER - PROFESSIONAL EDITION".center(68) + "█")
    print("█" + "     Decision Support Systems Course".center(68) + "█")
    print("█" + "     Developed for Educational Purposes".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█"*70)


def display_method_info():
    """Display information about solution methods"""
    print("\n" + "="*70)
    print("📚 SOLUTION METHODS INFORMATION".center(70))
    print("="*70)
    
    print("\n📐 GRAPHICAL METHOD:")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ • Works only for problems with 2 variables (x₁, x₂)        │")
    print("   │ • Shows visual representation of feasible region           │")
    print("   │ • Displays all vertices and evaluates objective function   │")
    print("   │ • Best for understanding LP geometrically                  │")
    print("   └─────────────────────────────────────────────────────────────┘")
    
    print("\n🧮 SIMPLEX METHOD (Two-Phase):")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ • Works for any number of variables                         │")
    print("   │ • Supports all constraint types (≤, =, ≥)                   │")
    print("   │ • Phase I: Find feasible solution (minimize artificial)    │")
    print("   │ • Phase II: Optimize original objective function           │")
    print("   │ • Shows step-by-step tableau iterations                    │")
    print("   └─────────────────────────────────────────────────────────────┘")
    
    print("\n💡 TIPS:")
    print("   • For large problems, enter constraints systematically")
    print("   • Type 'done' to finish entering constraints of each type")
    print("   • The Simplex method can handle problems with many variables")
    print("="*70)


def get_problem_from_user():
    """Get problem input from user with clear instructions"""
    
    print("\n" + "="*70)
    print("📝 PROBLEM INPUT".center(70))
    print("="*70)
    
    # Method selection
    print("\n🔧 Select Solution Method:")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ 1. Graphical Method (2 variables only)                  │")
    print("   │ 2. Simplex Method (any number of variables)             │")
    print("   └─────────────────────────────────────────────────────────┘")
    
    method = input("\n   Enter your choice (1 or 2): ").strip()
    
    while method not in ['1', '2']:
        print("   ❌ Invalid choice. Please enter 1 or 2.")
        method = input("\n   Enter your choice (1 or 2): ").strip()
    
    if method == "1":
        n_vars = 2
        print("\n   ✅ Selected: Graphical Method (x₁, x₂)")
        print("   ℹ️  This method will display a visual plot of the feasible region.")
    else:
        n_vars = int(input("\n   📊 Number of decision variables: "))
        while n_vars < 1:
            print("   ❌ Number of variables must be at least 1.")
            n_vars = int(input("\n   📊 Number of decision variables: "))
        print(f"\n   ✅ Selected: Simplex Method ({n_vars} variables)")
        print("   ℹ️  This method will show step-by-step tableau iterations.")
    
    # Objective function
    print("\n" + "-"*70)
    print("🎯 OBJECTIVE FUNCTION")
    print("-"*70)
    print("   Format: Z = c₁x₁ + c₂x₂ + ... + cₙxₙ")
    print("   Example for 2 variables: c₁ = 3, c₂ = 5 → Z = 3x₁ + 5x₂\n")
    
    c = []
    for j in range(n_vars):
        val = input(f"   Coefficient of x{j+1}: ")
        while True:
            try:
                val = float(val)
                break
            except ValueError:
                print("   ❌ Please enter a numeric value.")
                val = input(f"   Coefficient of x{j+1}: ")
        c.append(val)
    
    # Optimization direction
    print("\n" + "-"*70)
    print("📈 OPTIMIZATION DIRECTION")
    print("-"*70)
    print("   Choose whether to maximize or minimize the objective function\n")
    dir_choice = input("   Maximize (M) or Minimize (N)? [M/N]: ").strip().lower()
    maximize = dir_choice in ['m', 'max', 'maximize']
    
    obj_expr = " + ".join([f"{c[j]}x{j+1}" for j in range(n_vars)])
    print(f"\n   ✅ {'Maximizing' if maximize else 'Minimizing'} Z = {obj_expr}")
    
    # Constraints input
    print("\n" + "-"*70)
    print("📋 CONSTRAINTS INPUT")
    print("-"*70)
    print("\n   You can enter three types of constraints:")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ • ≤  (less than or equal to)                           │")
    print("   │ • =  (equal to)                                        │")
    print("   │ • ≥  (greater than or equal to)                        │")
    print("   └─────────────────────────────────────────────────────────┘")
    print("\n   For each constraint, enter the coefficients and RHS.")
    print(f"   Example for {n_vars} variables: ", end="")
    example = " ".join([f"{i+1}" for i in range(n_vars)]) + " 100"
    print(f"'{example}' means x₁ + 2x₂ + ... + {n_vars}x{n_vars} ≤ 100")
    print("   Type 'done' when you finish each category.\n")
    
    # Get ≤ constraints
    print("="*70)
    print("📝 CONSTRAINT TYPE: ≤ (Less than or equal to)")
    print("="*70)
    print("   (If you have no ≤ constraints, type 'done' immediately)\n")
    A_ineq, b_ineq = [], []
    i = 1
    while True:
        inp = input(f"   Constraint {i}: ").strip()
        if inp.lower() == 'done':
            break
        try:
            vals = list(map(float, inp.split()))
            if len(vals) == n_vars + 1:
                A_ineq.append(vals[:-1])
                b_ineq.append(vals[-1])
                expr = " + ".join([f"{vals[j]}x{j+1}" for j in range(n_vars)])
                print(f"   ✅ Added: {expr} ≤ {vals[-1]}")
                i += 1
            else:
                print(f"   ❌ Error: Expected {n_vars + 1} values, got {len(vals)}")
                print(f"   Format: a1 a2 ... a{n_vars} b")
        except ValueError:
            print("   ❌ Error: Please enter numeric values")
    
    # Get = constraints
    print("\n" + "="*70)
    print("📝 CONSTRAINT TYPE: = (Equal to)")
    print("="*70)
    print("   (If you have no = constraints, type 'done' immediately)\n")
    A_eq, b_eq = [], []
    i = 1
    while True:
        inp = input(f"   Constraint {i}: ").strip()
        if inp.lower() == 'done':
            break
        try:
            vals = list(map(float, inp.split()))
            if len(vals) == n_vars + 1:
                A_eq.append(vals[:-1])
                b_eq.append(vals[-1])
                expr = " + ".join([f"{vals[j]}x{j+1}" for j in range(n_vars)])
                print(f"   ✅ Added: {expr} = {vals[-1]}")
                i += 1
            else:
                print(f"   ❌ Error: Expected {n_vars + 1} values, got {len(vals)}")
                print(f"   Format: a1 a2 ... a{n_vars} b")
        except ValueError:
            print("   ❌ Error: Please enter numeric values")
    
    # Get ≥ constraints
    print("\n" + "="*70)
    print("📝 CONSTRAINT TYPE: ≥ (Greater than or equal to)")
    print("="*70)
    print("   (If you have no ≥ constraints, type 'done' immediately)\n")
    A_geq, b_geq = [], []
    i = 1
    while True:
        inp = input(f"   Constraint {i}: ").strip()
        if inp.lower() == 'done':
            break
        try:
            vals = list(map(float, inp.split()))
            if len(vals) == n_vars + 1:
                A_geq.append(vals[:-1])
                b_geq.append(vals[-1])
                expr = " + ".join([f"{vals[j]}x{j+1}" for j in range(n_vars)])
                print(f"   ✅ Added: {expr} ≥ {vals[-1]}")
                i += 1
            else:
                print(f"   ❌ Error: Expected {n_vars + 1} values, got {len(vals)}")
                print(f"   Format: a1 a2 ... a{n_vars} b")
        except ValueError:
            print("   ❌ Error: Please enter numeric values")
    
    # Summary
    print("\n" + "="*70)
    print("📋 PROBLEM SUMMARY".center(70))
    print("="*70)
    print(f"\n  {'Maximize' if maximize else 'Minimize'} Z = " + 
          " + ".join([f"{c[j]}x{j+1}" for j in range(n_vars)]))
    print("\n  Subject to:")
    for idx, row in enumerate(A_ineq):
        print(f"    {' + '.join([f'{row[j]}x{j+1}' for j in range(n_vars)])} ≤ {b_ineq[idx]}")
    for idx, row in enumerate(A_eq):
        print(f"    {' + '.join([f'{row[j]}x{j+1}' for j in range(n_vars)])} = {b_eq[idx]}")
    for idx, row in enumerate(A_geq):
        print(f"    {' + '.join([f'{row[j]}x{j+1}' for j in range(n_vars)])} ≥ {b_geq[idx]}")
    print("    x₁, x₂, ..., xₙ ≥ 0")
    print("="*70)
    
    confirm = input("\n   Is this correct? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("\n   Please restart problem input.")
        return get_problem_from_user()
    
    return method, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq, maximize, n_vars


def run_examples():
    """Run predefined examples"""
    print("\n" + "█"*70)
    print("📚 BUILT-IN EXAMPLES".center(70))
    print("█"*70)
    
    print("\n   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ 1️⃣  Example 1: Maximize Z = 3x₁ + 5x₂ (from PDF)            │")
    print("   │     Constraints: x₁ ≤ 4, 2x₂ ≤ 12, 3x₁ + 2x₂ ≤ 18            │")
    print("   │     Expected Solution: x₁=2, x₂=6, Z=36                      │")
    print("   │                                                             │")
    print("   │ 2️⃣  Example 2: Maximize Z = 4x₁ + 3x₂ + 6x₃                 │")
    print("   │     Constraints: 3x₁+x₂+3x₃ ≤ 30, 2x₁+2x₂+3x₃ ≤ 40,         │")
    print("   │                   x₁+2x₂+x₃ ≤ 20                             │")
    print("   │     Expected Solution: x₁=0, x₂=6, x₃=8, Z=66                │")
    print("   │                                                             │")
    print("   │ 3️⃣  Example 3: Minimize Z = 4x₁ + x₂                        │")
    print("   │     Constraints: x₁ + x₂ ≥ 6, x₁ ≤ 5, x₂ ≤ 4                 │")
    print("   │     Expected Solution: x₁=2, x₂=4, Z=12                      │")
    print("   │                                                             │")
    print("   │ 4️⃣  Example 4: Maximize Z = 3x₁ + 2x₂                       │")
    print("   │     Constraints: x₁ + x₂ = 10, x₁ ≤ 8, x₂ ≤ 7                │")
    print("   │     Expected Solution: x₁=8, x₂=2, Z=28                      │")
    print("   │                                                             │")
    print("   │ 5️⃣  Example 5: Maximize Z = 50x₁ + 18x₂ (PDF page 25)       │")
    print("   │     Constraints: 2x₁ + x₂ ≤ 100, x₁ + x₂ ≤ 80                │")
    print("   │     Expected Solution: x₁=50, x₂=0, Z=2500                   │")
    print("   └─────────────────────────────────────────────────────────────┘")
    
    choice = input("\n🔢 Select example (1-5): ").strip()
    
    while choice not in ['1', '2', '3', '4', '5']:
        print("   ❌ Invalid choice. Please enter 1-5.")
        choice = input("\n🔢 Select example (1-5): ").strip()
    
    if choice == "1":
        print("\n" + "="*50)
        print("▶ Solving Example 1 with Simplex Method")
        print("="*50)
        solver = SimplexSolver(maximize=True)
        solver.solve([3, 5], [[1,0], [0,2], [3,2]], [4,12,18], [], [], [], [])
        
        print("\n" + "="*50)
        print("▶ Solving Example 1 with Graphical Method")
        print("="*50)
        solve_graphical(3, 5, [[1,0], [0,2], [3,2]], [4,12,18], [], [], [], [], maximize=True)
    
    elif choice == "2":
        print("\n▶ Solving Example 2 with Simplex Method")
        solver = SimplexSolver(maximize=True)
        solver.solve([4,3,6], [[3,1,3], [2,2,3], [1,2,1]], [30,40,20], [], [], [], [])
    
    elif choice == "3":
        print("\n▶ Solving Example 3 with Simplex Method")
        solver = SimplexSolver(maximize=False)
        solver.solve([4, 1], [], [], [], [], [[1,1], [1,0], [0,1]], [6, 5, 4])
        
        print("\n▶ Solving Example 3 with Graphical Method (verification)")
        solve_graphical(4, 1, [[1,0], [0,1]], [5, 4], [], [], [[1,1]], [6], maximize=False)
    
    elif choice == "4":
        print("\n▶ Solving Example 4 with Simplex Method")
        solver = SimplexSolver(maximize=True)
        solver.solve([3,2], [[1,0], [0,1]], [8,7], [[1,1]], [10], [], [])
        
        print("\n▶ Solving Example 4 with Graphical Method")
        solve_graphical(3, 2, [[1,0], [0,1]], [8,7], [[1,1]], [10], [], [], maximize=True)
    
    elif choice == "5":
        print("\n▶ Solving Example 5 with Graphical Method")
        solve_graphical(50, 18, [[2,1], [1,1]], [100, 80], [], [], [], [], maximize=True)


# ============================================
# MAIN PROGRAM
# ============================================

def main():
    display_header()
    
    while True:
        print("\n" + "-"*70)
        print("📌 MAIN MENU".center(70))
        print("-"*70)
        print("   ┌─────────────────────────────────────────────────────────┐")
        print("   │ 1. Solve a new problem                                  │")
        print("   │ 2. Run built-in examples                                │")
        print("   │ 3. View method information                              │")
        print("   │ 4. Exit                                                 │")
        print("   └─────────────────────────────────────────────────────────┘")
        
        choice = input("\n   Enter your choice (1-4): ").strip()
        
        while choice not in ['1', '2', '3', '4']:
            print("   ❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            choice = input("\n   Enter your choice (1-4): ").strip()
        
        if choice == "1":
            method, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq, maximize, n_vars = get_problem_from_user()
            
            if method == "1" and n_vars == 2:
                solve_graphical(c[0], c[1], A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq, maximize)
            elif method == "2" or n_vars > 2:
                solver = SimplexSolver(maximize=maximize)
                solver.solve(c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq)
            else:
                print("\n❌ Graphical method only works with exactly 2 variables!")
        
        elif choice == "2":
            run_examples()
        
        elif choice == "3":
            display_method_info()
        
        elif choice == "4":
            print("\n" + "="*70)
            print("👋 Thank you for using Linear Programming Solver!".center(70))
            print("   Good luck with your Decision Support Systems course!".center(70))
            print("="*70)
            break
        
        input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    main()