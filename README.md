# Linear Programming Solver

A professional Python-based Linear Programming Solver implementing both Graphical and Two-Phase Simplex methods for solving optimization problems.

## 📋 Description

This project was developed as a final project for the **Decision Support Systems** course. It provides an interactive command-line interface to solve linear programming problems with support for:

- Maximization and Minimization
- Multiple constraint types (≤, =, ≥)
- Any number of decision variables (Simplex method)
- 2-variable problems (Graphical method with visualization)

## ✨ Features

- **Graphical Method** - Visual representation for 2-variable problems
- **Two-Phase Simplex Method** - Handles all constraint types for any number of variables
- **Step-by-step iterations** - Shows tableau evolution through each iteration
- **Interactive input** - User-friendly interface with input validation
- **Built-in examples** - Predefined examples from course materials
- **Professional output** - Formatted tables and clear solution display

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **NumPy** - Matrix operations and numerical computations
- **Matplotlib** - Graphical visualization of feasible regions

## 📁 Project Structure
linear-programming-solver/
│
├── main.py # Main entry point with menu system
├── simplex_solver.py # Simplex method implementation (Two-Phase)
├── graphical_solver.py # Graphical method for 2-variable problems
├── utils.py # Helper functions for formatting and validation
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore file
│
├── examples/ # Example problem files
│ ├── example1.txt
│ ├── example2.txt
│ ├── example3.txt
│ ├── example4.txt
│ └── example5.txt


text

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/linear-programming-solver.git
cd linear-programming-solver
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the program
bash
python main.py
📖 Usage Examples
Example 1: Maximization with ≤ constraints
text
Maximize Z = 3x₁ + 5x₂
Subject to:
x₁ ≤ 4
2x₂ ≤ 12
3x₁ + 2x₂ ≤ 18
x₁, x₂ ≥ 0

Expected Solution: x₁ = 2, x₂ = 6, Z = 36
Example 2: Minimization with ≥ constraints
text
Minimize Z = 4x₁ + x₂
Subject to:
x₁ + x₂ ≥ 6
x₁ ≤ 5
x₂ ≤ 4
x₁, x₂ ≥ 0

Expected Solution: x₁ = 2, x₂ = 4, Z = 12

🎯 Features in Detail
Graphical Method
Plots all constraint lines

Identifies feasible region

Finds all vertices

Evaluates objective function at each vertex

Displays optimal solution on plot

Simplex Method (Two-Phase)
Phase I: Finds feasible solution by minimizing artificial variables

Phase II: Optimizes original objective function

Shows complete tableau evolution

Handles unbounded and infeasible cases

📝 Input Format
For each constraint, enter coefficients and RHS separated by spaces:

For ≤: a1 a2 ... an b

For =: a1 a2 ... an b

For ≥: a1 a2 ... an b

Example for 3 variables: 2 1 3 30 means 2x₁ + 1x₂ + 3x₃ ≤ 30

Type done to finish entering constraints of each type.

👨‍💻 Author
Course: Decision Support Systems

Purpose: Educational Project

📄 License
This project is for educational purposes only.

🙏 Acknowledgments
Course instructor for guidance

Open-source libraries (NumPy, Matplotlib)