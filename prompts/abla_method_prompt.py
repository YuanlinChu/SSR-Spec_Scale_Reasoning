# Stage 2: Prompt templates for each strategy A–L plus M (fallback)
#策略池为6，prompt集合
A_prompt = """Try simplifying or transforming the expressions using algebraic identities such as factoring, expanding, or completing the square."""

B_prompt = """Consider defining unknown variables and setting up equations or a system of equations that reflect the problem's conditions."""

C_prompt = """Try solving the problem by testing small cases or constructing examples that satisfy the given constraints."""

D_prompt = """Consider drawing a diagram and applying geometric theorems or constructions (e.g. similar triangles, angle bisectors)."""

E_prompt = """Look for patterns in divisibility, parity, or remainders. Modular arithmetic or prime factorization may help."""

F_prompt = """Try identifying a recurrence relation or pattern. Consider using mathematical induction if applicable."""

#策略池为9，额外的prompt
G_prompt = """Consider placing the objects in a coordinate plane and using analytic geometry (like distance formulas or vector operations)."""

H_prompt = """Try to count the number of ways something can happen, or compute the required probability by enumeration or formula."""

I_prompt = """Estimate or bound quantities using known inequalities (AM-GM, Cauchy-Schwarz, etc.) or consider optimization approaches."""

#策略池为15，额外的prompt
J_prompt = """Try unusual substitutions or creative constructs—look for surprising simplifications or auxiliary elements."""

K_prompt = """Check if the problem has symmetry or invariant properties that can simplify reasoning or reduce cases."""

L_prompt = """Analyze the dimensions or units of the problem. Consider whether consistency of measurement or structure gives insights."""

P_prompt = """Assume the opposite of what you want to prove, or consider limiting behavior to derive contradictions."""

N_prompt = """If the problem involves relationships or connections, try modeling it using graph theory concepts like edges, paths, or components."""

O_prompt = """Try analyzing the function's graph, monotonicity, periodicity, or fixed points to extract insights."""

# M. Unknown / fallback
M_prompt = """None""" 