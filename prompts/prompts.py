system_prompt_1 = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.

Example:
Problem: Rosa has a bag containing $5$ red balls and $7$ blue balls. She randomly draws balls one at a time **without replacement** until she draws a red ball. Let the expected number of balls Rosa draws be $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.

Solution:
Define a random variable $X$ to be the number of balls Rosa draws to obtain the first red ball.  
Initially, there are $5$ red and $7$ blue balls, making $12$ balls in total.  
The expected value $E$ can be computed by considering two possibilities: drawing a red ball on the first draw or drawing a blue ball first.  
If Rosa draws a red ball immediately (probability $\\frac{5}{12}$), it takes $1$ draw. Otherwise (probability $\\frac{7}{12}$), she draws a blue ball and must continue, with the expected additional number of draws being $E'$, where $E'$ is the expected number of draws starting from $11$ balls, still $5$ red balls remaining.  

Thus,  
\\[
E = 1 \\times \\frac{5}{12} + (1 + E') \\times \\frac{7}{12}
\\]  
Simplifying,  
\\[
E = 1 + \\frac{7}{12}E'
\\]  
Similarly, $E'$ satisfies  
\\[
E' = 1 \\times \\frac{5}{11} + (1+E'') \\times \\frac{6}{11}
\\]  
and so on, recursively.  
Continuing this process, solving step-by-step, we eventually find  
\\[
E = \\frac{84}{25}
\\]  
Thus, $m+n = 84+25 = \\boxed{109}$.

Problem: {problem}
"""

system_prompt_2 = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.

Example:
Problem: Rosa has a bag containing $5$ red balls and $7$ blue balls. She randomly draws balls one at a time **without replacement** until she draws a red ball. Let the expected number of balls Rosa draws be $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.

Solution:
Let $E(x,y)$ denote the expected number of draws starting with $x$ red balls and $y$ blue balls.  
Initially, $E(5,7)$ is what we seek.  
The recurrence is:  
\\[
E(x,y) = 1 + \\frac{y}{x+y}E(x,y-1)
\\]  
because with probability $\\frac{x}{x+y}$ Rosa draws a red ball immediately (taking $1$ move), and with probability $\\frac{y}{x+y}$ she draws a blue ball, consuming $1$ move and reducing the number of blue balls by $1$.  

Base case:  
\\[
E(x,0) = 1
\\]  
since if no blue balls are left, the first ball drawn must be red.  

Working recursively, we have:  
\\[
E(5,0) = 1
\\]
\\[
E(5,1) = 1 + \\frac{1}{6} \\times 1 = \\frac{7}{6}
\\]
\\[
E(5,2) = 1 + \\frac{2}{7} \\times \\frac{7}{6} = \\frac{10}{6} = \\frac{5}{3}
\\]
and so on, up to $E(5,7)$.  
Following through all steps carefully, we eventually compute  
\\[
E(5,7) = \\frac{84}{25}
\\]  
thus again $m+n = \\boxed{109}$.

Problem: {problem}
"""

system_prompt_3 = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.

Example:
Problem: Rosa has a bag containing $5$ red balls and $7$ blue balls. She randomly draws balls one at a time **without replacement** until she draws a red ball. Let the expected number of balls Rosa draws be $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.

Solution:
The probability that the first red ball is drawn on the $k$-th draw can be computed.  
Specifically, for each $k$, the probability is the chance that the first $k-1$ balls are all blue and the $k$-th ball is red.  

The probability that the first ball is red is $\\frac{5}{12}$.  
The probability that the first ball is blue and the second ball is red is $\\frac{7}{12} \\times \\frac{5}{11}$.  
The probability that the first two balls are blue and the third is red is $\\frac{7}{12} \\times \\frac{6}{11} \\times \\frac{5}{10}$.  
Proceeding similarly, the expected value is:

\\[
E = 1 \\times \\frac{5}{12} + 2 \\times \\frac{7}{12} \\times \\frac{5}{11} + 3 \\times \\frac{7}{12} \\times \\frac{6}{11} \\times \\frac{5}{10} + \\cdots
\\]  

Calculating term-by-term (noticing the probabilities decrease significantly after a few terms), we find:  
\\[
E = \\frac{84}{25}
\\]  
thus again $m+n = \\boxed{109}$.

Problem: {problem}
"""