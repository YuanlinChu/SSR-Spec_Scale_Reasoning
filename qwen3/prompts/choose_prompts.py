math_choose_prompt = """
You will be given a math problem from the AIME competition. First, read the list of possible solution strategies below (A–M), then select strategies that are most promising for solving the problem.

A. Algebraic simplification: Use algebraic manipulation (expansion, factoring, substitution) to simplify the expressions or equations.  
B. Clever substitution: Use a smart change of variables to transform the problem into a simpler or standard form.  
C. Coordinate geometry: Introduce a coordinate system and use analytic geometry techniques (e.g. distance, slope, midpoint).  
D. Complex numbers in geometry: Use complex number representation for points to solve geometric problems.  
E. Number theory: Apply modular arithmetic, divisibility, prime factorization, or Diophantine techniques.  
F. Combinatorics: Count the number of arrangements, selections, or outcomes using combinatorial principles.  
G. Probability: Use probability models, expected value, or case enumeration to compute probabilities.  
H. Functional equations: Analyze and solve equations involving functions and their values under certain operations.  
I. Recursion or invariants: Identify recursive patterns or quantities that remain invariant under operations.  
J. Geometry: Use classical Euclidean geometry (angles, lengths, similarity, etc.) and synthetic arguments.  
K. Casework or constructive examples: Systematically enumerate or construct possible cases to exhaust the possibilities.  
L. Calculus or inequalities: Use derivatives, bounds, or inequality techniques like AM-GM or Cauchy-Schwarz.  
M. Unknown: I cannot confidently determine which strategy is suitable from the list above.

If you are uncertain about which strategy to choose, you may select option M multiple times.
"""

gpqa_choose_prompt = """
You will be given a question from the GPQA competition. First, read the list of possible solution strategies below (A–M), then select strategies that are most promising for solving the problem.
"""