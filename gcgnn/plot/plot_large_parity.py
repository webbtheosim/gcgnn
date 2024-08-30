import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import proplot as pplt
import numpy as np
import os
import pickle
from matplotlib.ticker import LogLocator
from gcgnn.analysis import hyper_best
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def plot_large_parity(
    rg_mean,
    COLORS,
    PLOT_DIR,
    TRAIN_RESULT_DIR,
    pattern=0,
    y_type="mean",
    scoring="RMSE",
    plot_type="linear",
    rerun=False,
):

    methods = ["GNN", "Baseline", "GNN_Guided_Baseline_Simple"]

    if y_type == "mean":
        vmin = 0
        vmax = 150
        ticks = [0, 50, 100, 150]
    elif y_type == "std":
        vmin=-1
        vmax=60
        ticks=[0, 20, 40, 60]

    gridsize = 50

    fig, ax = pplt.subplots(ncols=3, nrows=3, share=False, refwidth=2, refheight=2)

    for s in range(3):

        if s == 0:
            idx = 2
        elif s == 1:
            idx = 0
        elif s == 2:
            idx = 1

        pickle_temp_file = (
            f"../result_temp/parity_{s}_{pattern}_{y_type}_{scoring}_v2.pickle"
        )

        if not os.path.exists(pickle_temp_file) or rerun:

            v_metric, y_metric, file = hyper_best(
                TRAIN_RESULT_DIR,
                methods=methods,
                split_type=s,
                pattern=pattern,
                y_type=y_type,
                scoring=scoring,
            )

            with open(pickle_temp_file, "wb") as handle:
                pickle.dump(v_metric, handle)
                pickle.dump(y_metric, handle)
                pickle.dump(file, handle)

        else:

            with open(pickle_temp_file, "rb") as handle:
                v_metric = pickle.load(handle)
                y_metric = pickle.load(handle)
                file = pickle.load(handle)

        for i in range(3):

            with open(file[i], "rb") as handle:
                y_true = pickle.load(handle)
                y_pred = pickle.load(handle)

            if "_1.pickle" in file[i]:
                x = 10**y_true
                y = 10**y_pred

            else:
                x = y_true
                y = y_pred

            if plot_type == "log":
                x = np.log10(x)
                y = np.log10(y)

            if s == 0:
                mmin = np.min([rg_mean[40].min(), rg_mean[90].min()])
                mmax = np.max([rg_mean[40].max(), rg_mean[90].max()])
            elif s == 1:
                mmin = np.min([rg_mean[90].min(), rg_mean[190].min()])
                mmax = np.max([rg_mean[90].max(), rg_mean[190].max()])
            elif s == 2:
                mmin = np.min([rg_mean[40].min(), rg_mean[190].min()])
                mmax = np.max([rg_mean[40].max(), rg_mean[190].max()])

            if i == 0:
                im = ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap="Greys",
                    extent=[vmin, vmax, vmin, vmax],
                )
                ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap="Dusk",
                    extent=[vmin, vmax, vmin, vmax],
                )
            elif i ==1:
                cmap = "Fire"
                ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap=cmap,
                    extent=[vmin, vmax, vmin, vmax],
                )

            elif i == 2:
                marine_cmap = pplt.Colormap("Marine")
                colors = marine_cmap(np.linspace(0, 1, marine_cmap.N))
                colors[0] = [1, 1, 1, 1]
                cmap = mcolors.ListedColormap(colors)

                ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap=cmap,
                    extent=[vmin, vmax, vmin, vmax],
                )

            ax[idx * 3 + i].plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, lw=1.1)

            ax[idx * 3 + i].plot(
                [vmin, vmax], [mmin, mmin], "--", c='r', alpha=1, lw=1.1
            )

            ax[idx * 3 + i].plot(
                [vmin, vmax], [mmax, mmax], "--", c='r', alpha=1, lw=1.1
            )

            if i != 0:
                ylabel = None
                yticklabels = [""] * len(ticks)
            else:
                yticklabels = ticks

            ax[idx * 3 + i].format(
                xlabel=None,
                ylabel=None,
                xlabelsize=12,
                ylabelsize=12,
                xticklabelsize=11,
                yticklabelsize=11,
                xlim=[vmin, vmax],
                ylim=[vmin, vmax],
                grid="off",
                xticks=ticks,
                yticks=ticks,
            )

            ax[idx * 3 + i].set_yticklabels(yticklabels)

            xloc = 0.05
            yloc1 = 0.91
            yloc2 = 0.81
            yloc3 = 0.71

            ax[idx * 3 + i].text(
                xloc,
                yloc1,
                f"RMSE = {mean_squared_error(x, y)**0.5:0.2f}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="left",
                fontsize=12,
                color="black",
            )
            ax[idx * 3 + i].text(
                xloc,
                yloc2,
                rf"$\mathit{{R}}^2$ = {r2_score(x, y):0.2f}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="left",
                fontsize=12,
                color="black",
            )

            ax[idx * 3 + i].text(
                xloc,
                yloc3,
                rf"$\mathit{{r}}$ = {pearsonr(x, y)[0]:0.2f}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="left",
                fontsize=12,
                color="black",
            )

            xloc = 0.95
            if y_type == "mean":
                yloc1 = 0.06
                yloc2 = 0.16
            elif y_type == "std":
                yloc2 = 0.215
                yloc1 = 0.115

            if idx == 0:
                testset = r"MW$_\mathrm{low}$"
            elif idx == 1:
                testset = r"MW$_\mathrm{mid}$"
            elif idx == 2:
                testset = r"MW$_\mathrm{hi}$"

            if i == 0:
                model = "GNN"
            elif i == 1:
                model = "GC"
            elif i == 2:
                model = "GC-GNN"

            ax[idx * 3 + i].text(
                xloc,
                yloc1,
                f"{testset}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="right",
                fontsize=12,
                color="black",
            )

            ax[idx * 3 + i].text(
                xloc,
                yloc2,
                f"{model}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="right",
                fontsize=12,
                color="black",
            )

            ax[idx * 3 + i].tick_params(axis="both", which="both", width=1, top=True, right=True)
            for spine in ax[idx * 3 + i].spines.values():
                spine.set_linewidth(1)

        cb = fig.colorbar(
            im,
            ax=ax[idx * 3 + i],
            label="Frequency",
            labelsize=12,
            extend="both",
        )

        log_locator = LogLocator(base=10, subs="auto", numticks=10)

        cb.ax.yaxis.set_minor_locator(log_locator)
        cb.ax.tick_params(axis="both", which="both", width=1.0)
        cb.outline.set_linewidth(1.0) 

    if y_type == "mean":
        xlabel = r"Simulated $\langle \mathit{R}_{\mathrm{g}}^2 \rangle$"
        ylabel = r"Predicted $\langle \mathit{R}_{\mathrm{g}}^2 \rangle$"
    elif y_type == "std":
        xlabel = r"Simulated $\mathit{\sigma} \left( \mathit{R}_{\mathrm{g}}^2 \right)$"
        ylabel = r"Predicted $\mathit{\sigma} \left( \mathit{R}_{\mathrm{g}}^2 \right)$"

    ax[7].set_xlabel(xlabel, size=13)
    ax[3].set_ylabel(ylabel, size=13)

    fig_name = os.path.join(
        PLOT_DIR, f"parity_all_{pattern}_{y_type}_{scoring}_{plot_type}.png"
    )
    fig.save(fig_name, dpi=300, bbox_inches="tight")


def plot_large_parity_4row(
    rg_mean,
    COLORS,
    PLOT_DIR,
    TRAIN_RESULT_DIR,
    pattern=0,
    y_type="mean",
    scoring="RMSE",
    plot_type="linear",
    rerun=False,
):

    methods = ["GNN", "Baseline", "GNN_Guided_Baseline_Simple"]

    if y_type == "mean":
        vmin = 0
        vmax = 150
        ticks = [0, 50, 100, 150]
    elif y_type == "std":
        vmin = -1
        vmax = 80
        ticks = [0, 20, 40, 60, 80]

    gridsize = 50

    fig, ax = pplt.subplots(ncols=3, nrows=4, share=False, refwidth=2, refheight=2)

    for s in range(4):

        if s == 0:
            idx = 3
            split = 0
        elif s == 1:
            idx = 1
            split = 1
        elif s == 2:
            idx = 2
            split = 2
        elif s == 3:
            idx = 0
            split = 0
            pattern = 1

        pickle_temp_file = (
            f"../result_temp/parity_{split}_{pattern}_{y_type}_{scoring}_v2.pickle"
        )

        if not os.path.exists(pickle_temp_file) or rerun:
            v_metric, y_metric, file = hyper_best(
                TRAIN_RESULT_DIR,
                methods=methods,
                split_type=split,
                pattern=pattern,
                y_type=y_type,
                scoring=scoring,
            )

            with open(pickle_temp_file, "wb") as handle:
                pickle.dump(v_metric, handle)
                pickle.dump(y_metric, handle)
                pickle.dump(file, handle)

        else:

            with open(pickle_temp_file, "rb") as handle:
                v_metric = pickle.load(handle)
                y_metric = pickle.load(handle)
                file = pickle.load(handle)

        for i in range(3):

            with open(file[i], "rb") as handle:
                y_true = pickle.load(handle)
                y_pred = pickle.load(handle)

            if "_1.pickle" in file[i]:
                x = 10**y_true
                y = 10**y_pred

            else:
                x = y_true
                y = y_pred

            if plot_type == "log":
                x = np.log10(x)
                y = np.log10(y)

            if s == 0:
                mmin = np.min([rg_mean[40].min(), rg_mean[90].min()])
                mmax = np.max([rg_mean[40].max(), rg_mean[90].max()])
            elif s == 1:
                mmin = np.min([rg_mean[90].min(), rg_mean[190].min()])
                mmax = np.max([rg_mean[90].max(), rg_mean[190].max()])
            elif s == 2:
                mmin = np.min([rg_mean[40].min(), rg_mean[190].min()])
                mmax = np.max([rg_mean[40].max(), rg_mean[190].max()])

            if i == 0:
                im = ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap="Greys",
                    extent=[vmin, vmax, vmin, vmax],
                )
                ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap="Dusk",
                    extent=[vmin, vmax, vmin, vmax],
                )
            elif i == 1:
                cmap = "Fire"
                ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap=cmap,
                    extent=[vmin, vmax, vmin, vmax],
                )

            elif i == 2:
                marine_cmap = pplt.Colormap("Marine")
                colors = marine_cmap(np.linspace(0, 1, marine_cmap.N))
                colors[0] = [1, 1, 1, 1]
                cmap = mcolors.ListedColormap(colors)

                ax[idx * 3 + i].hexbin(
                    x,
                    y,
                    gridsize=gridsize,
                    norm=matplotlib.colors.LogNorm(),
                    cmap=cmap,
                    extent=[vmin, vmax, vmin, vmax],
                )

            ax[idx * 3 + i].plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, lw=1.1)

            ax[idx * 3 + i].plot(
                [vmin, vmax], [mmin, mmin], "--", c='r', alpha=1, lw=1.1
            )

            ax[idx * 3 + i].plot(
                [vmin, vmax], [mmax, mmax], "--", c='r', alpha=1, lw=1.1
            )

            if i != 0:
                ylabel = None
                yticklabels = [""] * len(ticks)
            else:
                yticklabels = ticks

            ax[idx * 3 + i].format(
                xlabel=None,
                ylabel=None,
                xlabelsize=12,
                ylabelsize=12,
                xticklabelsize=11,
                yticklabelsize=11,
                xlim=[vmin, vmax],
                ylim=[vmin, vmax],
                grid="off",
                xticks=ticks,
                yticks=ticks,
            )

            ax[idx * 3 + i].set_yticklabels(yticklabels)

            xloc = 0.05
            yloc1 = 0.91
            yloc2 = 0.81
            yloc3 = 0.71

            ax[idx * 3 + i].text(
                xloc,
                yloc1,
                f"RMSE = {mean_squared_error(x, y)**0.5:0.2f}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="left",
                fontsize=12,
                color="black",
            )
            ax[idx * 3 + i].text(
                xloc,
                yloc2,
                rf"$\mathit{{R}}^2$ = {r2_score(x, y):0.2f}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="left",
                fontsize=12,
                color="black",
            )

            ax[idx * 3 + i].text(
                xloc,
                yloc3,
                rf"$\mathit{{r}}$ = {pearsonr(x, y)[0]:0.2f}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="left",
                fontsize=12,
                color="black",
            )

            xloc = 0.95
            if y_type == "mean":
                yloc1 = 0.06
                yloc2 = 0.16
            elif y_type == "std":
                yloc2 = 0.2
                yloc1 = 0.1

            if idx == 0:
                testset = r"All MWs"
            elif idx == 1:
                testset = r"MW$_\mathrm{mid,hi}$"
            elif idx == 2:
                testset = r"MW$_\mathrm{low,hi}$"
            elif idx == 3:
                testset = r"MW$_\mathrm{low,mid}$"

            if i == 0:
                model = "GNN"
            elif i == 1:
                model = "GC"
            elif i == 2:
                model = "GC-GNN"

            ax[idx * 3 + i].text(
                xloc,
                yloc1,
                f"{testset}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="right",
                fontsize=12,
                color="black",
            )

            ax[idx * 3 + i].text(
                xloc,
                yloc2,
                f"{model}",
                transform=ax[idx * 3 + i].transAxes,
                verticalalignment="baseline",
                horizontalalignment="right",
                fontsize=12,
                color="black",
            )

            ax[idx * 3 + i].tick_params(
                axis="both", which="both", width=1, top=True, right=True
            )
            for spine in ax[idx * 3 + i].spines.values():
                spine.set_linewidth(1)

        cb = fig.colorbar(
            im,
            ax=ax[idx * 3 + i],
            label="Frequency",
            labelsize=12,
            extend="both",
        )

        log_locator = LogLocator(base=10, subs="auto", numticks=10)

        cb.ax.yaxis.set_minor_locator(log_locator)
        cb.ax.tick_params(axis="both", which="both", width=1.0)
        cb.outline.set_linewidth(1.0)

    if y_type == "mean":
        xlabel = r"Simulated $\langle \mathit{R}_{\mathrm{g}}^2 \rangle$"
        ylabel = r"Predicted $\langle \mathit{R}_{\mathrm{g}}^2 \rangle$"
    elif y_type == "std":
        xlabel = r"Simulated $\mathit{\sigma} \left( \mathit{R}_{\mathrm{g}}^2 \right)$"
        ylabel = r"Predicted $\mathit{\sigma} \left( \mathit{R}_{\mathrm{g}}^2 \right)$"

    ax[10].set_xlabel(xlabel, size=13)

    fig.text(
        0.006, 
        0.5,
        ylabel,
        va="center",
        size=13,
        rotation=90,
    )

    fig_name = os.path.join(
        PLOT_DIR, f"parity_all_{pattern}_{y_type}_{scoring}_{plot_type}.png"
    )
    fig.save(fig_name, dpi=300, bbox_inches="tight")
