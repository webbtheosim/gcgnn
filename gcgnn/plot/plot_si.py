import os
import pickle
import proplot as pplt
import numpy as np


def plot_si_rg_mean_std(PLOT_DIR, COLORS, topo, topo_unique, rg_mean, rg_std):

    fig, ax = pplt.subplots(refwidth=7 / 3, refheight=7 / 3, ncols=3, nrows=1, share=True)

    dps = [40, 90, 190]
    labels = [r"MW$_{low}$", r"MW$_{mid}$", r"MW$_{hi}$"]

    for i in range(3):
        xx = np.log10(rg_mean[dps[i]])
        yy = np.log10(rg_std[dps[i]])

        for j, u in enumerate(topo_unique):
            idx = np.where(topo[dps[i]] == u)[0]

            ax[i].plot(
                xx[idx], yy[idx], ".", c=COLORS[j], alpha=1, label=u.capitalize(), s=1
            )

            ax[i].xaxis.set_tick_params(which="both", direction="out", top=True)
            ax[i].yaxis.set_tick_params(which="both", direction="out", right=True)

        if i == 0:
            ylabel = r"log$_{10}$ $\sigma\left(\mathit{R}_{\mathrm{g}}^2\right)$"
            ax[i].legend(
                ncols=2,
                loc="upper left",
                prop={"size": 11},
                lw=1,
                markersize=10,
                handlelength=1.0,
            )
        else:
            ylabel = None

        if i == 1:
            xlabel = r"log$_{10}$ $\langle \mathit{R}_{\mathrm{g}}^2\rangle$"
        else:
            xlabel = None

        vmin = -1.0
        vmax = 2.0

        ax[i].plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5)

        ax[i].format(
            xlabel=xlabel,
            ylabel=ylabel,
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=11,
            yticklabelsize=11,
            xlim=[vmin, vmax],
            ylim=[vmin, vmax],
            xticks=[-1, 0, 1, 2],
            yticks=[-1, 0, 1, 2],
            title=labels[i],
            titlesize=12,
            grid="off",
        )

    output = os.path.join(PLOT_DIR, "rg_mean_std_relation_log10.png")
    fig.save(output, dpi=300)


def plot_si_sim_theo(DATA_DIR, PLOT_DIR, COLORS, topo, pat, mode, rg_mean, rg_std, dps=40, vmin_fix=0.4, vmax_fix=1.6, task="mean"):
    fig, ax = pplt.subplots(refwidth=7 / 3, refheight=7 / 3, ncols=3, nrows=2, share=False)

    if task == "mean":
        LABEL = r"$\mathrm{log}_{10} \langle \mathit{R}_{\mathrm{g}}^2\rangle$"
    elif task == "std":
        LABEL = r"$\mathrm{log}_{10} \mathit{\sigma} \left( \mathit{R}_{\mathrm{g}}^2\right)$"

    vmin_all, vmax_all = [], []

    for k, dp in enumerate([dps]):
        unique_label = np.unique(topo[dp])
        labels = [
            r"Homo $\mathit{\alpha}$",
            r"Homo $\mathit{\beta}$",
            "Co Random",
            "Co Regular",
        ]

        plot_data_file = os.path.join(DATA_DIR, f"rg2_baseline_{dp}_new.pickle")

        if task == "mean":
            with open(plot_data_file, "rb") as handle:
                rg2_baseline = pickle.load(handle)
        elif task == "std":
            with open(plot_data_file, "rb") as handle:
                _ = pickle.load(handle)
                rg2_baseline = pickle.load(handle)

        label_unique = np.unique(topo[dp])
        vmins, vmaxs = [], []

        for i, u in enumerate(label_unique):
            vmin, vmax = [], []

            for j in range(3):
                flag1 = topo[dp] == u
                if j != 2:
                    flag2 = np.array(pat[dp]) == 1 - j
                else:
                    flag2 = np.array(pat[dp]) == j

                idx = np.where(flag1 & flag2)[0]
                xx = rg2_baseline[idx, 0] * 3
                if task == "mean":
                    yy = rg_mean[dp][idx]
                elif task == "std":
                    yy = rg_std[dp][idx]
                xx = np.log10(xx)
                yy = np.log10(yy)
                color = COLORS[j]

                if j == 2:
                    s = 2
                else:
                    s = 2

                if j != 2:
                    ax[i].scatter(
                        xx, yy, edgecolor="k", lw=0.0, s=s, c=color, label=labels[j]
                    )
                else:
                    rand_mode = np.array(mode[dp]).astype("int")[idx]

                    if u == "dendrimer":
                        idx_temp1 = np.where(rand_mode <= 5)[0]
                        idx_temp2 = np.where(rand_mode > 5)[0]
                        ax[i].scatter(
                            xx[idx_temp2],
                            yy[idx_temp2],
                            label=labels[j],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[5],
                        )
                        ax[i].scatter(
                            xx[idx_temp1],
                            yy[idx_temp1],
                            label=labels[j + 1],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[4],
                        )
                    elif u == "comb":
                        idx_temp1 = np.where(rand_mode <= 15)[0]
                        idx_temp2 = np.where(rand_mode > 15)[0]
                        ax[i].scatter(
                            xx[idx_temp2],
                            yy[idx_temp2],
                            label=labels[j],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[5],
                        )
                        ax[i].scatter(
                            xx[idx_temp1],
                            yy[idx_temp1],
                            label=labels[j + 1],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[4],
                        )
                    elif u == "branch":
                        idx_temp1 = np.where(rand_mode <= 19)[0]
                        idx_temp2 = np.where(rand_mode > 19)[0]
                        ax[i].scatter(
                            xx[idx_temp2],
                            yy[idx_temp2],
                            label=labels[j],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[5],
                        )
                        ax[i].scatter(
                            xx[idx_temp1],
                            yy[idx_temp1],
                            label=labels[j + 1],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[4],
                        )
                    elif u == "star":
                        idx_temp1 = np.where(rand_mode <= 9)[0]
                        idx_temp2 = np.where(rand_mode > 9)[0]
                        ax[i].scatter(
                            xx[idx_temp2],
                            yy[idx_temp2],
                            label=labels[j],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[5],
                        )
                        ax[i].scatter(
                            xx[idx_temp1],
                            yy[idx_temp1],
                            label=labels[j + 1],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[4],
                        )
                    elif u == "linear":
                        idx_temp1 = np.where(rand_mode <= 3)[0]
                        idx_temp2 = np.where(rand_mode > 3)[0]
                        ax[i].scatter(
                            xx[idx_temp2],
                            yy[idx_temp2],
                            label=labels[j],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[5],
                        )
                        ax[i].scatter(
                            xx[idx_temp1],
                            yy[idx_temp1],
                            label=labels[j + 1],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[4],
                        )
                    elif u == "cyclic":
                        idx_temp1 = np.where(rand_mode <= 3)[0]
                        idx_temp2 = np.where(rand_mode > 3)[0]
                        ax[i].scatter(
                            xx[idx_temp2],
                            yy[idx_temp2],
                            label=labels[j],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[5],
                        )
                        ax[i].scatter(
                            xx[idx_temp1],
                            yy[idx_temp1],
                            label=labels[j + 1],
                            edgecolor="k",
                            lw=0.0,
                            s=s,
                            c=COLORS[4],
                        )

                vmin.append(np.min([xx.min(), yy.min()]))
                vmax.append(np.max([xx.max(), yy.max()]))

            ax[i].text(
                0.05,
                0.95,
                rf"{unique_label[i].capitalize()}",
                transform=ax[i].transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=12,
                color="black",
                fontweight="bold",
            )

            vmin = np.min(vmin)
            vmax = np.max(vmax)

            vmins.append(vmin)
            vmaxs.append(vmax)

        vmin_all.append(np.array(vmins))
        vmax_all.append(np.array(vmaxs))

    vmin_all = np.array(vmin_all).min(axis=0)
    vmax_all = np.array(vmax_all).max(axis=0)

    for i in range(6):

        ax[i].plot([vmin_fix, vmax_fix], [vmin_fix, vmax_fix], "k--", alpha=0.5)
        ax[i].xaxis.set_tick_params(which="both", direction="out", top=True)
        ax[i].yaxis.set_tick_params(which="both", direction="out", right=True)

        if i == 0 or i == 3:
            ylabel = "Simulated " + LABEL
        else:
            ylabel = None

        if i == 4:
            xlabel = "Theoretical " + LABEL
        else:
            xlabel = None

        if i == 4:
            ax[i].legend(ncol=1, loc="lower left", markersize=18)

        ax[i].format(
            xlabel=xlabel,
            ylabel=ylabel,
            xlabelsize=12,
            ylabelsize=12,
            xlim=[vmin_fix, vmax_fix],
            ylim=[vmin_fix, vmax_fix],
            grid="off",
            xticklabelsize=11,
            yticklabelsize=11,
        )

        if task == "mean":

            if dp == 40:
                if i != 3:
                    ax[i].set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
                    ax[i].set_yticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
                else:
                    ax[i].set_xlim([0.5, 1.0])
                    ax[i].set_ylim([0.5, 1.0])
                    ax[i].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
                    ax[i].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])

            elif dp == 90:
                if i != 3:
                    ax[i].set_xticks([.6, .8, 1.0, 1.2, 1.4, 1.6, 1.8])
                    ax[i].set_yticks([.6, .8, 1.0, 1.2, 1.4, 1.6, 1.8])
                else:
                    ax[i].set_xlim([0.7, 1.0])
                    ax[i].set_ylim([0.7, 1.0])
                    ax[i].set_xticks([.7, .8, .9, 1])
                    ax[i].set_yticks([.7, .8, .9, 1])

            elif dp == 190:
                if i != 3:
                    ax[i].set_xticks([ .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
                    ax[i].set_yticks([ .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
                else:
                    ax[i].set_xlim([0.9, 1.1])
                    ax[i].set_ylim([0.9, 1.1])
                    ax[i].set_xticks([.9, 1.0, 1.1])
                    ax[i].set_yticks([.9, 1.0, 1.1])

        elif task == "std":

            if dp == 40:
                if i != 3:
                    ax[i].set_xticks([-1, -0.5, 0, 0.5, 1.0])
                    ax[i].set_yticks([-1, -0.5, 0, 0.5, 1.0])
                else:
                    ax[i].set_xlim([-0.8, -0.4])
                    ax[i].set_ylim([-0.8, -0.4])
                    ax[i].set_xticks([-0.8, -0.6, -0.4])
                    ax[i].set_yticks([-0.8, -0.6, -0.4])

            elif dp == 90:
                if i != 3:
                    ax[i].set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5])
                    ax[i].set_yticks([-1, -0.5, 0, 0.5, 1.0, 1.5])
                else:
                    ax[i].set_xlim([-1, -0.58])
                    ax[i].set_ylim([-1, -0.58])
                    ax[i].set_xticks([-1, -0.8, -0.6])
                    ax[i].set_yticks([-1, -0.8, -0.6])

            elif dp == 190:
                if i != 3:
                    ax[i].set_xticks([-1, 0, 1, 2])
                    ax[i].set_yticks([-1, 0, 1, 2])
                else:
                    ax[i].set_xlim([-1, -0.58])
                    ax[i].set_ylim([-1, -0.58])
                    ax[i].set_xticks([-1, -0.8, -0.6])
                    ax[i].set_yticks([-1, -0.8, -0.6])

    if task == "mean":
        output = os.path.join(PLOT_DIR, f"rg_{dp}_baseline_simulated.png")
    elif task == "std":
        output = os.path.join(PLOT_DIR, f"srg_{dp}_baseline_simulated.png")
    fig.save(output, dpi=300)

