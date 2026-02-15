"""
Compare fishlists created by the harmonics module in order to create a fishlist with
fishes present in all fishlists.

- `consistent_fishes()`: Compares a list of fishlists and builds a consistent fishlist.
- `plot_selected_frequencies()`: plot all fundamental frequencies and mark selected frequencies with a bar.
"""

import numpy as np

from .harmonics import fundamental_freqs


def find_consistency(fundamentals, df_th=1.0):
    """
    Compares lists of floats to find these values consistent in every list.
    (with a certain threshold)

    Every value of the first list is compared to the values of the other
    lists. The consistency_help array consists in the beginning of ones,
    has the same length as the first list and is used to keep the
    information about values that are available in several lists. If the
    difference between a value of the first list and a value of another
    list is below the threshold the entry of the consistency_help array
    is added 1 to at the index of the value in the value in the first
    list. The indices of the consistency_help array that are equal to
    the amount of lists compared are the indices of the values of the
    first list that are consistent in all lists. The consistent value
    array and the indices are returned.


    Parameters
    ----------
    fundamentals: 2-D array
        List of lists containing the fundamentals of a fishlist.
        fundamentals = [ [f1, f1, ..., f1, f1], [f2, f2, ..., f2, f2], ..., [fn, fn, ..., fn, fn] ]
    df_th: float
        Frequency threshold for the comparison of different fishlists in Hertz. If the fundamental
        frequencies of two fishes from different fishlists vary less than this threshold they are
        assigned as the same fish.

    Returns
    -------
    consistent_fundamentals: 1-D array
        List containing all values that are available in all given lists.
    index: 1-D array
        Indices of the values that are in every list relating to the fist list in fishlists.
    """
    consistency_help = np.ones(len(fundamentals[0]), dtype=int)

    for enu, fundamental in enumerate(fundamentals[0]):
        for list in range(1, len(fundamentals)):
            if np.sum(np.abs(fundamentals[list] - fundamental) < df_th) > 0:
                consistency_help[enu] += 1

    index = np.arange(len(fundamentals[0]))[consistency_help == len(fundamentals)]
    consistent_fundamentals = fundamentals[0][index]

    return consistent_fundamentals, index


def consistent_fishes(fishlists, df_th=1):
    """
    Compares several fishlists to create a fishlist only containing these fishes present in all these fishlists.

    Therefore several functions are used to first extract the fundamental frequencies of every fish in each fishlist,
    before comparing them and building a fishlist only containing these fishes present in all fishlists.

    Parameters
    ----------
    fishlists: list of list of 2D array
        List of fishlists with harmonics and each frequency and power.
        fishlists[fishlist][fish][harmonic][frequency, power]
    df_th: float
        Frequency threshold for the comparison of different fishlists in Hertz. If the fundamental
        frequencies of two fishes from different fishlists vary less than this threshold they are
        assigned as the same fish.

    Returns
    -------
    filtered_fishlist: list of 2-D arrays
        New fishlist with the same structure as a fishlist in fishlists only
        containing these fishes that are available in every fishlist in fishlists.
        fishlist[fish][harmonic][frequency, power]
    """
    fundamentals = fundamental_freqs(fishlists)
    if len(fundamentals) == 0:
        return []

    consistent_fundamentals, index = find_consistency(fundamentals)

    # creates a filtered fishlist only containing the data of the fishes consistent in several fishlists.
    filtered_fishlist = []
    for idx in index:
        filtered_fishlist.append(fishlists[0][idx])

    return filtered_fishlist


def plot_selected_frequencies(ax, group_list, selected_groups, label=None,
                              freq_style=dict(ls='none', marker='o',
                                              color='k', ms=10),
                              bar_style=dict(ls='-', color='c', lw=12)):
    """
    Plot all fundamental frequencies and mark selected frequencies with a bar.

    Parameters
    ----------
    ax: matplotlib.Axes
        Axes for plotting.
    group_list: list of list of 2-D arrays of float
        Harmonic groups as returned by severla calls to extract_fundamentals()
        or harmonic_groups() with the element [0, 0] of the harmonic groups
        being the fundamental frequency, and element[0, 1] the corresponding
        power.
    selected_groups: list of 2-D arrays of float
        Frequencies and power of selected harmonic groups from `group_list`.
    label: str or None
        Plot label for the selected groups that is added to the legend.
    freq_style: dict
        Plot style for marking fundamental frequencies of `group_list`.
    bar_style: dict
        Plot style for marking frequencies from `selected_groupss`.
    """
    # mark selected frequencies:
    for index in range(len(selected_groups)):
        x = [0.75, len(group_list) + 0.25]
        y = [selected_groups[index][0, 0]]*2
        if index == 0 and label is not None:
            ax.plot(x, y, label=label, **bar_style)
        else:
            ax.plot(x, y, **bar_style)
    # mark all frequencies:
    for index in range(len(group_list)):
        for group in group_list[index]:
            ax.plot(index + 1, group[0, 0], **freq_style)
    ax.set_xlim([0.25, len(group_list) + 0.75])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_ylabel('frequency [Hz]')
    ax.set_xlabel('group index')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def mkg(f, n=5):
        return np.array([(h*f, 0) for h in range(1, n)])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained', sharey=True)
    
    
    group_list = [[mkg(350.0), mkg(700.2), mkg(960.4)],
                  [mkg(274.1), mkg(523.7), mkg(350.2), mkg(699.8)],
                  [mkg(349.7), mkg(523.4), mkg(700.8), mkg(959.8)],
                  [mkg(349.8), mkg(700.4), mkg(960.3)]]
    selected_groups = consistent_fishes(group_list)
    ax1.set_title('consistent groups')
    plot_selected_frequencies(ax1, group_list, selected_groups)

    # check almost empty fishlist:
    group_list = [[], [mkg(523.7)], [mkg(523.4)], []]
    selected_groups = consistent_fishes(group_list)
    ax2.set_title('consistent in mostly empty group list')
    plot_selected_frequencies(ax2, group_list, selected_groups)

    # check single fishlist:
    group_list = [[mkg(523.7)]]
    selected_groups = consistent_fishes(group_list)
    ax3.set_title('consistent in single group list')
    plot_selected_frequencies(ax3, group_list, selected_groups)

    ## TODO: move to tests/ !
    # check empty fishlist:
    group_list = [[], []]
    selected_groups = consistent_fishes(group_list)

    # check really empty fishlist:
    group_list = []
    selected_groups = consistent_fishes(group_list)
    
    plt.show()
