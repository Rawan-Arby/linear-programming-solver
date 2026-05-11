"""
Simplex Method Solver for Linear Programming
Uses Two-Phase method to handle all constraint types
"""

import numpy as np
from utils import print_header, print_section, format_tableau, is_optimal


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
        
    def solve_simplex(self, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq):
        """Solve LP using Two-Phase Simplex Method"""
        
        self.original_c = c.copy()
        
        print_header("SIMPLEX METHOD (Two-Phase)")
        
        self._display_problem(c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq)
        
        # Build initial tableau
        self._build_initial_tableau(c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq)
        
        print("\n📋 PHASE I - INITIAL TABLEAU")
        print("   (Minimizing sum of artificial variables to find feasible solution)")
        self._print_tableau()
        
        # Phase I
        if not self._phase_one():
            print("\n❌ ERROR: Problem is infeasible!")
            return None, None
        
        # Phase II
        if not self._phase_two():
            print("\n❌ ERROR: Problem is unbounded!")
            return None, None
        
        # Extract solution
        solution, obj_value = self._extract_solution()
        self._display_solution(solution, obj_value)
        
        return solution, obj_value
    
    def _display_problem(self, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq):
        """Display problem formulation"""
        print("\n📊 PROBLEM FORMULATION:")
        print("-" * 70)
        
        obj_parts = [f"{c[0]}x₁"]
        for j in range(1, len(c)):
            if c[j] >= 0:
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
        print("-" * 70)
    
    def _build_initial_tableau(self, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq):
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
    
    def _pivot(self, pivot_row, pivot_col):
        """Perform pivot operation"""
        self.basis[pivot_row] = pivot_col
        pivot_value = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_value
        
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i, :] -= factor * self.tableau[pivot_row, :]
    
    def _phase_one(self):
        """Execute Phase I to find feasible solution"""
        
        iteration = 1
        
        while True:
            # Check optimality
            if np.min(self.tableau[-1, :-1]) >= -1e-7:
                if abs(self.tableau[-1, -1]) < 1e-7:
                    print(f"\n✅ Phase I: Feasible solution found after {iteration} iterations!")
                    return True
                else:
                    print(f"\n❌ Phase I: Problem is infeasible")
                    print(f"   Minimum sum of artificial variables = {self.tableau[-1, -1]:.4f}")
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
            
            self._pivot(pivot_row, pivot_col)
            self._print_tableau()
            
            iteration += 1
            if iteration > 100:
                print("\n❌ Too many iterations!")
                return False
    
    def _phase_two(self):
        """Execute Phase II to optimize original objective"""
        
        print_header("PHASE II: Optimizing Original Objective")
        
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
        self._print_tableau()
        
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
            
            self._pivot(pivot_row, pivot_col)
            self._print_tableau()
            
            iteration += 1
            if iteration > 100:
                print("\n❌ Too many iterations!")
                return False
    
    def _print_tableau(self):
        """Print current tableau"""
        if len(self.var_names) <= 15:
            print(format_tableau(self.tableau, self.var_names, self.n_constraints))
        else:
            print(f"\n   (Tableau has {len(self.var_names)} variables, showing summary)")
            print(f"   Objective value: {self.tableau[-1, -1]:.4f}")
    
    def _extract_solution(self):
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
    
    def _display_solution(self, solution, obj_value):
        """Display final solution"""
        print("\n" + "=" * 70)
        print("✨ OPTIMAL SOLUTION ✨".center(70))
        print("=" * 70)
        
        for j in range(self.n_vars):
            print(f"  x{j+1} = {solution[j]:.4f}")
        
        if self.maximize:
            print(f"\n  Maximum Z = {obj_value:.4f}")
        else:
            print(f"\n  Minimum Z = {obj_value:.4f}")
        
        print("=" * 70)