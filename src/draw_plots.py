import matplotlib.pyplot as plt

def error_random_active(num_queries, active, random, name, fname):
    plt.plot(num_queries, active)
    plt.plot(num_queries, random)

    plt.legend(["active learner", "random learner"])
    plt.xlabel("Number of Queries")
    plt.ylabel("Error")
    plt.title(name)
    plt.savefig(fname)
    plt.close()
