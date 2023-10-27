#!/usr/bin/env python3

##
#
# A script for comparing the scalability of different linear policy
# parameterizations for the swimmer example. 
#
##

from swimmer import train, evaluate
import pickle
import matplotlib.pyplot as plt

def gather_data():
    architectures = ["parallel", "series", "hierarchy"]
    num_blocks = [1, 2]

    # Dictionary for storing the results. The keys are tuples of the form
    # (architecture, num_blocks) and the values are tuples of (average reward, std).
    results = {}

    for arch in architectures:
        for nb in num_blocks:
            print("Training", arch, "with", nb, "blocks")
            train(arch, nb)
            print("Evaluating", arch, "with", nb, "blocks")
            avg_reward, std_reward = evaluate()
            print("Average reward:", avg_reward, "+/-", std_reward)
            results[(arch, nb)] = (avg_reward, std_reward)
            print("\n\n")

    return results

def plot_results(results):
    """
    Plot num_blocks vs. average reward for each architecture.
    """
    # Separate the results by architecture
    parallel_results = []
    series_results = []
    hierarchy_results = []
    for key, value in results.items():
        if key[0] == "parallel":
            parallel_results.append((key[1], value[0], value[1]))
        elif key[0] == "series":
            series_results.append((key[1], value[0], value[1]))
        elif key[0] == "hierarchy":
            hierarchy_results.append((key[1], value[0], value[1]))

    # Sort the results by number of blocks
    parallel_results.sort(key=lambda x: x[0])
    series_results.sort(key=lambda x: x[0])
    hierarchy_results.sort(key=lambda x: x[0])

    # Plot the results
    plt.figure()
    plt.errorbar([x[0] for x in parallel_results], [x[1] for x in parallel_results], 
                 yerr=[x[2] for x in parallel_results], label="Parallel")
    plt.errorbar([x[0] for x in series_results], [x[1] for x in series_results], 
                 yerr=[x[2] for x in series_results], label="Series")
    plt.errorbar([x[0] for x in hierarchy_results], [x[1] for x in hierarchy_results], 
                 yerr=[x[2] for x in hierarchy_results], label="Hierarchy")
    plt.xlabel("Number of blocks")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()


if __name__=="__main__":
    LOAD_SAVED_RESULTS = True

    if LOAD_SAVED_RESULTS:
        # Load the results from disk
        with open("linear_swimmer_comparison_results.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        # Collect the data
        results = gather_data()

        # Save the results to disk (for backup)
        with open("linear_swimmer_comparison_results.pkl", "wb") as f:
            pickle.dump(results, f)

    # Make a plot of the results
    plot_results(results)

