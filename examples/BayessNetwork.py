from bayesnet.decision_making import Variable, Factor, BayesianNetwork
from bayesnet.visualization import plot_bayesian_network

X = Variable('x', 2)
Y = Variable('y', 2)
Z = Variable('z', 2)

ϕ = Factor([X, Y, Z], {
    (1, 1, 1): 0.08,
    (1, 1, 2): 0.31,
    (1, 2, 1): 0.09,
    (1, 2, 2): 0.37,
    (2, 1, 1): 0.01,
    (2, 1, 2): 0.05,
    (2, 2, 1): 0.02,
    (2, 2, 2): 0.07
})

bn = BayesianNetwork([X, Y, Z], [ϕ], [(1, 2), (2, 3)])
plot_bayesian_network(bn)