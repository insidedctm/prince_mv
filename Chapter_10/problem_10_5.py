# Gibbs sampling of the probability distribution for problem 10.5
import numpy as np

def main(samples=1000000000, burnin=10000):
    # intialise
    results = initialize_results_dict()
    x = initial_x()
    print('initialise x to {}'.format(x))

    for s in range(samples):
        x1, x2, x3, x4 = x
        # sample pr(x_1 | x_2, x_4)
        x1 = gibbs(x2, x4)

        # sample pr(x_2 | x_1, x_3)
        x2 = gibbs(x1, x3)

        # sample pr(x_3 | x_2, x_4)
        x3 = gibbs(x2, x4)

        # sample pr(x_4 | x_3, x_1)
        x4 = gibbs(x3, x1)

        # increment results
        x = (x1, x2, x3, x4)
        if s >= burnin:
            results[x] = results[x] + 1

    prob_total = 0.0 
    for state in sorted(results.keys()):
        expectation = results[state]*1.0/(samples-burnin)
        prob_total += expectation
        print('{} : {:.6f}'.format(state, expectation))

    print('Total of probabilities (sanity check): {}'.format(prob_total))

def gibbs(cond_rv1, cond_rv2):
    if cond_rv1 == 0 and cond_rv2 == 0:
        # p(x_i = 0 | rv1 = 0, rv2 = 0) = 0.9901
        return 0 if np.random.rand() < 0.9901 else 1
    elif cond_rv1 == 1 and cond_rv2 == 1:
        # p(x_i = 0 | rv1 = 1, rv2 = 1) = 0.0025
        return 0 if np.random.rand() < 0.0025 else 1
    else:
        # p(x_i = 0 | rv1 = 0, rv2 = 1) or p(x_i = 0 | rv1 = 1, rv2 = 0) = 0.3333
        return 0 if np.random.rand() < 0.3333 else 1


def initial_x():
    return (rand_0_1(), rand_0_1(), rand_0_1(), rand_0_1())

def rand_0_1():
    return 1 if np.random.rand() > 0.5 else 0

def initialize_results_dict():
    results = {}
    for x1 in [0,1]:
        for x2 in [0,1]:
            for x3 in [0,1]:
                for x4 in [0,1]:
                    results[(x1, x2, x3, x4)] = 0
    return results

if __name__ == '__main__':
    main()
