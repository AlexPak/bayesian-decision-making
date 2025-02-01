from bayesnet.decision_making import Variable, Factor, BayesianNetwork

# Example Usage
X, Y, Z = Variable('x', 2), Variable('y', 2), Variable('z', 2)
ϕ = Factor([X, Y, Z], {
    (1,1,1): 0.08, (1,1,2): 0.31,
    (1,2,1): 0.09, (1,2,2): 0.37,
    (2,1,1): 0.01, (2,1,2): 0.05,
    (2,2,1): 0.02, (2,2,2): 0.07
})
utilities = {
    'x': {(1,): -0.1, (2,): -1},
    'y': {(1,): 0.9, (2,): 1}
}

dn = BayesianNetwork([X, Y, Z], [ϕ], [('x', 'y'), ('y', 'z')], utilities)

dn.normalize_factors()
evidence = {'x': 1, 'y': 2, 'z': 1}
probability_result = dn.probability(evidence) # !
expected_utility = dn.expected_utility(evidence) # !

D = [{'x': 1, 'y': 2, 'z': 1}, {'x': 2, 'y': 1, 'z': 2}, {'x': 1, 'y': 2, 'z': 2}, {'x': 2, 'y': 1, 'z': 1}]
bayesian_score_result = dn.bayesian_score(D) # !

print(f"Probability of evidence {evidence}: {probability_result}")
print(f"Expected Utility: {expected_utility}")
print(f"Bayesian Score: {bayesian_score_result}")

dn.plot_graph()