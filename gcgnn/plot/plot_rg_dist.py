import os
import proplot as pplt
import numpy as np
from gcgnn.plot import adjacent_values


def plot_rg_dist(PLOT_DIR, COLORS, topo, rg_mean, rg_std):

    ylabel = r"$\langle \mathit{R}_{\mathrm{g}}^2\rangle$"
    labels = [r"MW$_{low}$", r"MW$_{mid}$", r"MW$_{hi}$"]

    rg_stat = rg_mean
    YLABEL = ylabel

    fig, ax = pplt.subplots(refwidth=12, refheight=1.5, ncols=1, nrows=2, sharey=False)

    for i in range(6):
        vmaxs = []
        for j, dp in enumerate([40, 90, 190]):
            unique_label = ["linear", "branch", "comb", "star", "cyclic", "dendrimer"]

            idx = np.where(topo[dp] == unique_label[i])
            data = rg_stat[dp][idx]

            violin = ax[0].violinplot(i * 4 + j, data)

            for pc in violin:
                pc.set_facecolor(COLORS[j])
                pc.set_edgecolor("black")
                pc.set_alpha(1)

            sorted_data = np.sort(data)

            quartile1, medians, quartile3 = np.percentile(sorted_data, [25, 50, 75])
            whisker_min, whisker_max = adjacent_values(sorted_data, quartile1, quartile3)

            ax[0].scatter(i * 4 + j, medians, marker="o", color="white", s=10, zorder=3)
            ax[0].vlines(i * 4 + j, quartile1, quartile3, color="k", linestyle="-", lw=4)
            ax[0].vlines(
                i * 4 + j, whisker_min, whisker_max, color="k", linestyle="-", lw=1
            )

            mean_val = np.mean(sorted_data)
            ax[0].plot(
                [i * 4 + j - 0.5, i * 4 + j + 0.5],
                [mean_val, mean_val],
                "--",
                zorder=1,
                c=COLORS[j],
            )

            if i == 0:
                ax[0].scatter(
                    i * 4 + j,
                    0,
                    marker="o",
                    color=COLORS[j],
                    s=20,
                    zorder=3,
                    label=labels[j],
                )
            else:
                ax[0].scatter(i * 4 + j, 0, marker="o", color=COLORS[j], s=20, zorder=3)

            vmaxs.append(np.max(sorted_data))


    ax[0].xaxis.set_tick_params(which="both", direction="out", top=True)
    ax[0].yaxis.set_tick_params(which="both", direction="out", right=True)

    ax[0].format(
        ylabel=YLABEL,
        xlabelsize=15,
        ylabelsize=15,
        xticklabelsize=15,
        yticklabelsize=13,
        ylim=[1, 200],
        yscale="log",
        yticks=[1, 10, 100],
        xticks=[1, 5, 9, 13, 17, 21],
        xticklabels=[unique_label[k].capitalize() for k in range(6)],
        grid="off",
    )


    ylabel = r"$\mathit{\sigma} \left( \mathit{R}_{\mathrm{g}}^2\right)$"

    rg_stat = rg_std
    YLABEL = ylabel

    for i in range(6):
        vmaxs = []
        for j, dp in enumerate([40, 90, 190]):
            unique_label = ["linear", "branch", "comb", "star", "cyclic", "dendrimer"]

            idx = np.where(topo[dp] == unique_label[i])
            data = rg_stat[dp][idx]

            violin = ax[1].violinplot(i * 4 + j, data)

            for pc in violin:
                pc.set_facecolor(COLORS[j])
                pc.set_edgecolor("black")
                pc.set_alpha(1)

            sorted_data = np.sort(data)

            quartile1, medians, quartile3 = np.percentile(sorted_data, [25, 50, 75])
            whisker_min, whisker_max = adjacent_values(sorted_data, quartile1, quartile3)

            ax[1].scatter(i * 4 + j, medians, marker="o", color="white", s=10, zorder=3)
            ax[1].vlines(i * 4 + j, quartile1, quartile3, color="k", linestyle="-", lw=4)
            ax[1].vlines(
                i * 4 + j, whisker_min, whisker_max, color="k", linestyle="-", lw=1
            )

            mean_val = np.mean(sorted_data)
            ax[1].plot(
                [i * 4 + j - 0.5, i * 4 + j + 0.5],
                [mean_val, mean_val],
                "--",
                zorder=1,
                c=COLORS[j],
            )

            if i == 0:
                ax[1].scatter(
                    i * 4 + j,
                    0,
                    marker="o",
                    color=COLORS[j],
                    s=20,
                    zorder=3,
                    label=labels[j],
                )
            else:
                ax[1].scatter(i * 4 + j, 0, marker="o", color=COLORS[j], s=20, zorder=3)


    ax[1].legend(ncols=1, prop={"size": 13}, lw=1)

    ax[1].xaxis.set_tick_params(which="both", direction="out", top=True)
    ax[1].yaxis.set_tick_params(which="both", direction="out", right=True)

    ax[1].format(
        ylabel=YLABEL,
        xlabelsize=15,
        ylabelsize=15,
        xticklabelsize=15,
        yticklabelsize=13,
        ylim=[0.1, 100],
        yscale="log",
        yticks=[0.1, 1, 10, 100],
        xticks=[1, 5, 9, 13, 17, 21],
        xticklabels=[unique_label[k].capitalize() for k in range(6)],
        grid="off",
    )


    output = os.path.join(PLOT_DIR, "rg_mean_std_distribution.svg")
    fig.save(output, dpi=300)
