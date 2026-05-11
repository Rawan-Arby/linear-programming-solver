"""
Graphical Method Solver for Linear Programming
Solves LP problems with 2 variables only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from utils import print_header, print_section


def solve_graphical(c1, c2, A_ineq, b_ineq, A_eq, b_eq, A_geq, b_geq, maximize=True):
    """Solve LP with 2 variables graphically"""
    
    print_header("GRAPHICAL METHOD SOLUTION")
    
    print("\n📊 PROBLEM FORMULATION:")
    print("-" * 70)
    if maximize:
        print(f"  Maximize Z = {c1}x₁ + {c2}x₂")
    else:
        print(f"  Minimize Z = {c1}x₁ + {c2}x₂")
    
    print("\n  Subject to:")
    
    A_combined = []
    b_combined = []
    
    for i in range(len(A_ineq)):
        A_combined.append([A_ineq[i][0], A_ineq[i][1]])
        b_combined.append(b_ineq[i])
        print(f"    {A_ineq[i][0]}x₁ + {A_ineq[i][1]}x₂ ≤ {b_ineq[i]}")
    
    for i in range(len(A_eq)):
        A_combined.append([A_eq[i][0], A_eq[i][1]])
        b_combined.append(b_eq[i])
        A_combined.append([-A_eq[i][0], -A_eq[i][1]])
        b_combined.append(-b_eq[i])
        print(f"    {A_eq[i][0]}x₁ + {A_eq[i][1]}x₂ = {b_eq[i]}")
    
    for i in range(len(A_geq)):
        A_combined.append([-A_geq[i][0], -A_geq[i][1]])
        b_combined.append(-b_geq[i])
        print(f"    {A_geq[i][0]}x₁ + {A_geq[i][1]}x₂ ≥ {b_geq[i]}")
    
    print("    x₁ ≥ 0, x₂ ≥ 0")
    print("-" * 70)
    
    if not A_combined:
        print("\n❌ No constraints entered!")
        return None, None
    
    # Find vertices
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
                feasible = all(A_combined[j][0]*x1 + A_combined[j][1]*x2 <= b_combined[j] + 1e-7 
                               for j in range(len(A_combined)))
                if feasible:
                    vertices.append((x1, x2))
        
        if abs(a2) > 1e-7:
            x1 = 0
            x2 = rhs / a2
            if x2 >= 0:
                feasible = all(A_combined[j][0]*x1 + A_combined[j][1]*x2 <= b_combined[j] + 1e-7 
                               for j in range(len(A_combined)))
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
                    feasible = all(A_combined[k][0]*x1 + A_combined[k][1]*x2 <= b_combined[k] + 1e-5 
                                   for k in range(len(A_combined)))
                    if feasible:
                        vertices.append((x1, x2))
    
    vertices = list(set([(round(v[0], 6), round(v[1], 6)) for v in vertices]))
    
    if not vertices:
        print("\n❌ ERROR: No feasible region found!")
        return None, None
    
    print_section("FEASIBLE REGION VERTICES")
    
    obj_values = []
    for v in vertices:
        z = c1 * v[0] + c2 * v[1]
        obj_values.append(z)
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
    
    print_header(f"OPTIMAL SOLUTION ({direction})")
    print(f"  x₁ = {optimal_point[0]:.4f}")
    print(f"  x₂ = {optimal_point[1]:.4f}")
    print(f"  Z = {best_value:.4f}")
    print("=" * 70)
    
    # Plot the feasible region
    plot_feasible_region(A_combined, b_combined, vertices, optimal_point, c1, c2)
    
    return optimal_point, best_value


def plot_feasible_region(A_combined, b_combined, vertices, optimal_point, c1, c2):
    """Plot the feasible region and constraints"""
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
                   label=f'Optimal Point ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})')
    
    plt.xlabel('x₁', fontsize=14, fontweight='bold')
    plt.ylabel('x₂', fontsize=14, fontweight='bold')
    plt.title('Graphical Method - Linear Programming', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linewidth=0.8)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.xlim(-0.5, max_x)
    plt.ylim(-0.5, max_x)
    plt.show()