"""
Utility functions for Linear Programming Solver
Contains helper functions for formatting, input validation, and display
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def format_number(x, decimals=4):
    """Format a number for display"""
    if abs(x) < 1e-7:
        return "0.0000"
    return f"{x:.{decimals}f}"


def format_tableau(tableau, var_names, n_constraints):
    """Format tableau for display"""
    lines = []
    
    # Header
    header = "   " + " ".join([f"{name:>6}" for name in var_names] + ["    RHS"])
    lines.append(header)
    lines.append("   " + "-" * (9 * (len(var_names) + 1)))
    
    # Rows
    for i in range(tableau.shape[0]):
        row_name = f"R{i+1}" if i < n_constraints else "Z"
        row_str = f"{row_name:>2} |"
        for j in range(tableau.shape[1]):
            if j == tableau.shape[1] - 1:
                row_str += f"  {tableau[i, j]:>8.3f}"
            else:
                row_str += f"  {tableau[i, j]:>6.2f}"
        lines.append(row_str)
    lines.append("")
    
    return "\n".join(lines)


def validate_positive(value, name):
    """Validate that a value is positive"""
    try:
        val = float(value)
        if val >= 0:
            return val
        else:
            print(f"❌ Error: {name} must be non-negative")
            return None
    except ValueError:
        print(f"❌ Error: {name} must be a number")
        return None


def is_optimal(tableau, maximize):
    """Check if current tableau is optimal"""
    if maximize:
        return np.min(tableau[-1, :-1]) >= -1e-7
    else:
        return np.max(tableau[-1, :-1]) <= 1e-7


def print_separator(char="=", length=70):
    """Print a separator line"""
    print(char * length)


def print_header(text, char="=", length=70):
    """Print a centered header"""
    print(char * length)
    print(text.center(length))
    print(char * length)


def print_section(text, length=70):
    """Print a section header"""
    print("\n" + "-" * length)
    print(text.center(length))
    print("-" * length)


def get_yes_no_input(prompt):
    """Get yes/no input from user"""
    while True:
        response = input(prompt + " (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("❌ Please enter 'y' or 'n'")


def get_numeric_input(prompt, allow_negative=True):
    """Get numeric input from user with validation"""
    while True:
        try:
            val = float(input(prompt))
            if not allow_negative and val < 0:
                print("❌ Value cannot be negative. Please try again.")
                continue
            return val
        except ValueError:
            print("❌ Please enter a valid number.")


def display_solution(solution, obj_value, maximize, var_names=None):
    """Display solution in a formatted way"""
    print("\n" + "=" * 70)
    print("✨ OPTIMAL SOLUTION ✨".center(70))
    print("=" * 70)
    
    if var_names is None:
        var_names = [f"x{j+1}" for j in range(len(solution))]
    
    for j, name in enumerate(var_names):
        print(f"  {name} = {solution[j]:.4f}")
    
    if maximize:
        print(f"\n  Maximum Z = {obj_value:.4f}")
    else:
        print(f"\n  Minimum Z = {obj_value:.4f}")
    
    print("=" * 70)