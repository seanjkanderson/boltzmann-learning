
import matplotlib.pyplot as plt


def plot_simulation_results(iters, costs, average_costs, entropies, average_cost_bounds=None, title_prefix=""):
    """
    Plots the simulation results including costs, average costs, entropy, and average cost bounds.

    Parameters:
    - iters: List or array of iteration indices.
    - costs: List or array of costs at each iteration.
    - average_costs: List or array of average costs up to each iteration.
    - entropies: List of players' array of entropy values at each iteration.
    - average_cost_bounds: (Optional) List or array of average cost bounds up to each iteration.
    - title_prefix: (Optional) Prefix string for plot titles.
    """
    slc = slice(int(.95 * len(iters)), len(iters))

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Plot costs, average cost, and entropy, from start
    ax_twin = ax[0].twinx()
    lines = ax[0].plot(iters, costs, '.', label='cost', color='tab:blue')
    lines += ax[0].plot(iters, average_costs, '-', label='average cost', color='tab:orange')
    if average_cost_bounds is not None:
        lines += ax[0].plot(iters, average_cost_bounds, '-', label='average cost bound', color='tab:cyan')
    # for idx, entropy in enumerate(entropies):
    #     lines += ax_twin.plot(iters, entropy, '.', label='entropy: P{}'.format(idx+1), color='tab:green')

    ax[0].grid()
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('cost')
    ax[0].set_ylim((-1.1, 1.1))
    ax_twin.set_ylabel('entropy')
    ax[0].set_title(f"{title_prefix} Costs, Average Costs, and Entropy (Start)")
    ax_twin.legend(lines, [line.get_label() for line in lines])

    # Plot costs, average cost, and entropy, skipping start
    ax_twin = ax[1].twinx()
    lines = ax[1].plot(iters[slc], average_costs[slc], '-', label='average cost', color='tab:orange')
    if average_cost_bounds is not None:
        lines += ax[1].plot(iters[slc], average_cost_bounds[slc], '-', label='average cost bound', color='tab:cyan')
    for idx, entropy in enumerate(entropies):
        lines += ax_twin.plot(iters[slc], entropy[slc], '.', label='entropy: P{}'.format(idx+1), color='tab:green')

    ax[1].grid()
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('cost')
    ax_twin.set_ylabel('entropy')
    ax[1].set_title(f"{title_prefix} Costs, Average Costs, and Entropy (End)")
    ax_twin.legend(lines, [line.get_label() for line in lines])

    ax_twin.set_zorder(ax[0].get_zorder() + 1)
    fig.tight_layout()
    plt.show()

    # Plot average costs for the first 100 iterations
    # fig, ax = plt.subplots()
    # ax.plot(average_costs[:100])
    # ax.set_title(f"{title_prefix} Average Costs (First 100 iterations)")
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('Average Cost')
    # ax.grid()
    # plt.show()