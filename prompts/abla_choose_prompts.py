method_selection_prompt_6 = """
You are solving a challenging high school-level competition math problem. Below are six possible solution strategies:

A. Algebraic manipulation and identities (factoring, expanding, substitutions, symmetry)
B. Equation construction (setting up equations or systems to model unknowns)
C. Enumeration or constructive examples (trying small cases, building examples that work)
D. Geometry with diagrams and theorems (drawings, angle chasing, similar triangles, etc.)
E. Elementary number theory (divisibility, modular arithmetic, primes, parity)
F. Recursion or induction (recursive patterns, induction proof)

Select up to three strategies that are most likely to help solve the problem. Return only the corresponding capital letters (A–F). 
If you are unsure, select F multiple times.
"""

method_selection_prompt_9 = """
You are solving a competition-level math problem. Consider the following nine strategies:

A. Algebraic manipulation and identities
B. Equation construction (modeling with unknowns)
C. Enumeration or constructive methods
D. Classical geometry and diagram-based reasoning
E. Number theory techniques (modulo, primes, parity)
F. Recursion or mathematical induction
G. Coordinate geometry (analytical geometry, vectors)
H. Combinatorics or probability
I. Inequalities and extremal value techniques

Choose up to three strategies (A–I) that seem most promising. If you cannot tell, select I multiple times.
"""

method_selection_prompt_15 = """
You are tackling a challenging math problem from a high-level competition. Below is a rich set of problem-solving strategies:

A. Algebraic manipulation and identities
B. Modeling with equations or systems
C. Constructive methods or casework
D. Classical geometry and theorems
E. Elementary number theory
F. Recursion or induction
G. Coordinate geometry or vectors
H. Combinatorics and probability
I. Inequalities or bounding techniques
J. Special substitutions or clever tricks
K. Symmetry and invariants
L. Dimensional/unit analysis
M. Contradiction or limiting argument
N. Graph theory approaches
O. Functional thinking (graphing, monotonicity, periodicity)

Pick up to three strategies (A–O) that are most promising for solving the problem. If you're unsure, select M multiple times.
"""