# Stage 2: Prompt templates for each strategy A–L plus M (fallback)

# A. Algebraic simplification
A_prompt = """Try simplifying the expressions or equations algebraically. Look for opportunities to factor, expand, or manipulate terms in a clean way."""

# B. Clever substitution
B_prompt = """Consider whether a clever substitution or change of variables could simplify the structure of the problem or make the expressions more tractable."""

# C. Coordinate geometry
C_prompt = """If the problem involves geometric objects, try placing them in the coordinate plane and using equations, distances, or slopes to analyze them."""

# D. Complex numbers in geometry
D_prompt = """Consider whether representing geometric points as complex numbers could simplify angle or length relationships."""

# E. Number theory
E_prompt = """Think about modular arithmetic, divisibility, or prime factorization. Check if properties of integers or residues could help."""

# F. Combinatorics
F_prompt = """Try counting the number of valid arrangements, combinations, or permutations. Look for patterns or apply standard counting principles."""

# G. Probability
G_prompt = """If the problem involves uncertainty or randomness, consider modeling with probability and computing expected values or probabilities."""

# H. Functional equations
H_prompt = """Check if the problem involves a function with a special recursive or algebraic relationship, and explore its properties step by step."""

# I. Recursion or invariants
I_prompt = """Try identifying a quantity that stays invariant during changes, or formulate a recursive relationship that builds toward the answer."""

# J. Geometry
J_prompt = """Use classical Euclidean geometry arguments such as triangle similarity, angle chasing, circle theorems, or area ratios."""

# K. Casework or constructive examples
K_prompt = """Consider splitting the problem into multiple cases or constructing small examples to test and verify the general behavior."""

# L. Calculus or inequalities
L_prompt = """Try applying inequality techniques like AM-GM or Cauchy-Schwarz, or think about bounds and extrema if the problem suggests it."""

# M. Unknown / fallback
M_prompt = """None""" 