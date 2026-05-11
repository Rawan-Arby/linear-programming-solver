"""
Linear Programming Solver - Main Program
Entry point for the application
"""

import sys
import os
from utils import print_header, print_section, get_yes_no_input
from graphical_solver import solve_graphical
from simplex_solver import SimplexSolver


def display_menu():
    """Display the main menu"""
    print("\n" + "-" * 70)
    print("📌 MAIN MENU".center(70))
    print("-" * 70)
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ 1. Solve a new problem                                  │")
    print("   │ 2. Run built-in examples                                │")
    print("   │ 3. View method information                              │")
    print("   │ 4. Exit                                                 │")
    print("   └─────────────────────────────────────────────────────────┘")


def get_constraints_input(n_vars, constraint_type, symbol):
    """Get constraints input from user"""
    print(f"\n📝 CONSTRAINT TYPE: {constraint_type} ({symbol})")
    print("   (If you have no constraints, type 'done' immediately)\n")
    
    A, b = [], []
    i = 1
    
    while True:
        inp = input(f"   Constraint {i}: ").strip()
        if inp.lower() == 'done':
            break
        try:
            vals = list(map(float, inp.split()))
            if len(vals) == n_vars + 1:
                A.append(vals[:-1])
                b.append(vals[-1])
                expr = " + ".join([f"{vals[j]}x{j+1}" for j in range(n_vars)])
                print(f"   ✅ Added: {expr} {symbol} {vals[-1]}")
                i += 1
            else:
                print(f"   ❌ Error: Expected {n_vars + 1} values, got {len(vals)}")
                print(f"   Format: a1 a2 ... a{n_vars} b")
        except ValueError:
            print("   ❌ Error: Please enter numeric values")
    
    return A, b


def get_problem_from_user():
    """Get problem input from user"""
    
    print_header("PROBLEM INPUT")
    
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
        print("\n   ✅ Selected: Graphical Method (x1, x2)")
    else:
        n_vars = int(input("\n   📊 Number of decision variables: "))
        while n_vars < 1:
            print("   ❌ Number of variables must be at least 1.")
            n_vars = int(input("\n   📊 Number of decision variables: "))
        print(f"\n   ✅ Selected: Simplex Method ({n_vars} variables)")
    
    # Objective function
    print_section("OBJECTIVE FUNCTION")
    print("   Format: Z = c1x1 + c2x2 + ... + cnxn\n")
    
    c = []
    for j in range(n_vars):
        while True:
            try:
                val = float(input(f"   Coefficient of x{j+1}: "))
                c.append(val)
                break
            except ValueError:
                print("   ❌ Please enter a numeric value.")
    
    # Optimization direction
    print_section("OPTIMIZATION DIRECTION")
    dir_choice = input("   Maximize (M) or Minimize (N)? [M/N]: ").strip().lower()
    maximize = dir_choice in ['m', 'max', 'maximize']
    
    obj_expr = " + ".join([f"{c[j]}x{j+1}" for j in range(n_vars)])
    print(f"\n   ✅ {'Maximizing' if maximize else 'Minimizing'} Z = {obj_expr}")
    
    # Constraints input
    print_section("CONSTRAINTS INPUT")
    print("\n   You can enter three types of constraints:")
    print("   • ≤ (less than or equal to)")
    print("   • = (equal to)")
    print("   • ≥ (greater than or equal to)")
    print(f"\n   Example for {n_vars} variables: ", end="")
    example = " ".join([f"{i+1}" for i in range(n_vars)]) + " 100"
    print(f"'{example}' means x1 + 2x2 + ... + {n_vars}x{n_vars} ≤ 100")
    print("   Type 'done' when you finish each category.\n")
    
    # Get constraints
    A_ineq, b_ineq = get_constraints_input(n_vars, "≤", "≤")
    A_eq, b_eq = get_constraints_input(n_vars, "=", "=")
    A_geq, b_geq = get_constraints_input(n_vars, "≥", "≥")
    
    # Summary
    print_header("PROBLEM SUMMARY")
    print(f"\n  {'Maximize' if maximize else 'Minimize'} Z = " + 
          " + ".join([f"{c[j]}x{j+1}" for j in range(n_vars)]))
    print("\n  Subject to:")
    for idx, row in enumerate(A_ineq):
        print(f"    {' + '.join([f'{row[j]}x{j+1}' for j in range(n_vars)])} ≤ {b_ineq[idx]}")
    for idx, row in enumerate(A_eq):
        print(f"    {' + '.join([f'{row[j]}x{j+1}' for j in range(n_vars)])} = {b_eq[idx]}")
    for idx, row in enumerate(A_geq):
        print(f"    {' + '.join([f'{row[j]}x{j+1}' for j in range(n_vars)])} ≥ {b_geq[idx]}")
    print("    x1, x2, ..., xn ≥ 0")
    print("=" * 70)
    
    if not get_yes_no_input("\n   Is this correct"):
        print("\n   Please restart problem input.")
        return get_problem_from_user()
    
    return method, c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq, maximize, n_vars


def display_examples():
    """Display built-in examples in a clean format"""
    print_header("📚 BUILT-IN EXAMPLES")
    
    examples = [
        ("1", "Maximize Z = 3x1 + 5x2", "x1 ≤ 4, 2x2 ≤ 12, 3x1 + 2x2 ≤ 18", "x1=2, x2=6, Z=36"),
        ("2", "Maximize Z = 4x1 + 3x2 + 6x3", "3x1+x2+3x3 ≤ 30, 2x1+2x2+3x3 ≤ 40, x1+2x2+x3 ≤ 20", "x1=0, x2=6, x3=8, Z=66"),
        ("3", "Minimize Z = 4x1 + x2", "x1+x2 ≥ 6, x1 ≤ 5, x2 ≤ 4", "x1=2, x2=4, Z=12"),
        ("4", "Maximize Z = 3x1 + 2x2", "x1+x2 = 10, x1 ≤ 8, x2 ≤ 7", "x1=8, x2=2, Z=28"),
        ("5", "Maximize Z = 50x1 + 18x2", "2x1+x2 ≤ 100, x1+x2 ≤ 80", "x1=50, x2=0, Z=2500"),
    ]
    
    for num, obj, cons, sol in examples:
        print(f"\n   ┌─ {num}️⃣ ─────────────────────────────────────────────────────────────")
        print(f"   │  📌 {obj}")
        print(f"   │  📋 {cons}")
        print(f"   │  ✅ {sol}")
        print("   └─────────────────────────────────────────────────────────────────")
    
    choice = input("\n🔢 Select example (1-5): ").strip()
    while choice not in ['1', '2', '3', '4', '5']:
        print("   ❌ Invalid choice. Please enter 1-5.")
        choice = input("\n🔢 Select example (1-5): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 50)
        print("▶ Solving Example 1 with Simplex Method")
        print("=" * 50)
        solver = SimplexSolver(maximize=True)
        solver.solve_simplex([3, 5], [[1,0], [0,2], [3,2]], [4,12,18], [], [], [], [])
        
        print("\n" + "=" * 50)
        print("▶ Solving Example 1 with Graphical Method")
        print("=" * 50)
        solve_graphical(3, 5, [[1,0], [0,2], [3,2]], [4,12,18], [], [], [], [], maximize=True)
    
    elif choice == "2":
        print("\n▶ Solving Example 2 with Simplex Method")
        solver = SimplexSolver(maximize=True)
        solver.solve_simplex([4,3,6], [[3,1,3], [2,2,3], [1,2,1]], [30,40,20], [], [], [], [])
    
    elif choice == "3":
        print("\n▶ Solving Example 3 with Simplex Method")
        solver = SimplexSolver(maximize=False)
        solver.solve_simplex([4, 1], [], [], [], [], [[1,1], [1,0], [0,1]], [6, 5, 4])
        
        print("\n▶ Solving Example 3 with Graphical Method (verification)")
        solve_graphical(4, 1, [[1,0], [0,1]], [5, 4], [], [], [[1,1]], [6], maximize=False)
    
    elif choice == "4":
        print("\n▶ Solving Example 4 with Simplex Method")
        solver = SimplexSolver(maximize=True)
        solver.solve_simplex([3,2], [[1,0], [0,1]], [8,7], [[1,1]], [10], [], [])
        
        print("\n▶ Solving Example 4 with Graphical Method")
        solve_graphical(3, 2, [[1,0], [0,1]], [8,7], [[1,1]], [10], [], [], maximize=True)
    
    elif choice == "5":
        print("\n▶ Solving Example 5 with Graphical Method")
        solve_graphical(50, 18, [[2,1], [1,1]], [100, 80], [], [], [], [], maximize=True)


def display_method_info():
    """Display information about solution methods"""
    print_header("SOLUTION METHODS INFORMATION")
    
    print("\n📐 GRAPHICAL METHOD:")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ • Works only for problems with 2 variables (x1, x2)        │")
    print("   │ • Shows visual representation of feasible region           │")
    print("   │ • Displays all vertices and evaluates objective function   │")
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
    print("=" * 70)


def main():
    """Main entry point"""
    print_header("LINEAR PROGRAMMING SOLVER - PROFESSIONAL EDITION")
    print("   Decision Support Systems Course".center(70))
    print("   Developed for Educational Purposes".center(70))
    
    while True:
        display_menu()
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
                solver.solve_simplex(c, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq)
            else:
                print("\n❌ Graphical method only works with exactly 2 variables!")
        
        elif choice == "2":
            display_examples()
        
        elif choice == "3":
            display_method_info()
        
        elif choice == "4":
            print("\n" + "=" * 70)
            print("👋 Thank you for using Linear Programming Solver!".center(70))
            print("   Good luck with your Decision Support Systems course!".center(70))
            print("=" * 70)
            sys.exit(0)
        
        input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    main()