import proplot as pplt
import numpy as np
import matplotlib
import os
import pickle
from matplotlib.ticker import LogLocator
from gcgnn.analysis import get_metrics


def plot_dendrimer(
    TRAIN_RESULT_DIR,
    PLOT_DIR,
    COLORS,
    label_data,
    input_file,
    cbar_vmax=None,
    bar_ratio=1.0,
    ylim1=[0, 0.8],
    ylim2=[-1, 1],
    first_ticks=[-1, -0.5, 0, 0.5, 1],
    yticklabels=[-1, -0.5, 0, 0.5, 1],
    vmin=0.1,
    vmax=0.6,
    second_ticks=[0.1, 0.15, 0.2],
):

    in_file = os.path.join(TRAIN_RESULT_DIR, input_file)

    if "mean" in input_file:
        mode = "mean"
    elif "std" in input_file:
        mode = "std"

    with open(in_file, "rb") as handle:
        y_true = pickle.load(handle)
        y_pred = pickle.load(handle)

    topos = np.copy(label_data[190])
    np.random.RandomState(42).shuffle(topos)
    unique_topos = np.unique(topos)

    xx = np.arange(len(unique_topos))

    yy = []

    for u in unique_topos:

        idx = np.where(topos == u)[0]
        y_true_temp = 10 ** y_true[idx]
        y_pred_temp = 10 ** y_pred[idx]

        outs = get_metrics(y_true_temp, y_pred_temp)

        yy.append(outs)

    yy_old = np.array(yy)

    xticks = unique_topos
    xticks = [t.capitalize() for t in xticks]

    new_xx = ["Brn.", "Comb", "Cyc.", "Den.", "Lin.", "Star"]

    fig, ax = pplt.subplots(ncols=1, nrows=2, sharey=False, refwidth=3, refheight=1.3)

    yy = np.copy(yy_old)

    yy[3, 2] = yy[3, 2] / bar_ratio

    ax[0].bar(
        xx, yy[:, -1], color=COLORS[0], edgecolor="k", width=0.5, label="MAE", lw=1
    )

    ax[1].bar(
        xx, yy[:, 2], color=COLORS[1], edgecolor="k", width=0.5, label="R", lw=1.0
    )

    ax[0].format(
        xticks=xx,
        xticklabels=new_xx,
        ylabel="MAPE",
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
        grid="off",
        ylim=ylim1,
    )

    ax[1].format(
        xticks=xx,
        xticklabels=new_xx,
        ylabel=r"$\mathit{R}^2$",
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
        ylim=ylim2,
        yticks=first_ticks,
        grid="off",
    )

    ax[1].set_yticklabels(yticklabels)

    ax[0].set_xticks([], minor=True)
    ax[0].yaxis.set_tick_params(
        labelleft=True, labelright=False, left=True, right=True, which="both"
    )
    ax[1].yaxis.set_tick_params(
        labelleft=True, labelright=False, left=True, right=True, which="both"
    )

    for ax_ in ax:
        ax_.tick_params(axis="both", which="both", width=1)
        for spine in ax_.spines.values():
            spine.set_linewidth(1)

    out_name = f"metric_topology_{input_file.split('.pickle')[0]}.svg"
    out_file = os.path.join(PLOT_DIR, out_name)

    fig.savefig(out_file)
    print(["RMSE", "MAE", "R2", "R", "MAPE"])
    print(np.round(yy[:, 3], 4))

    # second figure
    u = "dendrimer"
    idx = np.where(topos == u)[0]
    y_true_temp = 10 ** y_true[idx]
    y_pred_temp = 10 ** y_pred[idx]

    slope, intercept = np.polyfit(y_true_temp, y_pred_temp, 1)
    xx = np.linspace(vmin, vmax, 100)
    yy = xx * slope + intercept
    print(f"Slope: {slope:0.4f}")

    if mode == "mean":
        xlabel = r'Simulated $\langle \mathit{R}_{\mathrm{g}}^2 \rangle$'
        ylabel = r'Predicted $\langle \mathit{R}_{\mathrm{g}}^2 \rangle$'
    elif mode == "std":
        xlabel = r"Simulated $\sigma \left( \mathit{R}_{\mathrm{g}}^2 \right)$"
        ylabel = r"Predicted $\sigma \left( \mathit{R}_{\mathrm{g}}^2 \right)$"

    fig, ax = pplt.subplots(refwidth=3, refheight=3)

    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=1.0, lw=1)
    ax.plot(xx, yy, "k--", alpha=1, lw=2)

    # ax.scatter(y_true_temp, y_pred_temp, s=40, c=COLORS[1], edgecolor="w", alpha=0.9)

    im = ax.hexbin(
        y_true_temp,
        y_pred_temp,
        gridsize=50,
        norm=matplotlib.colors.LogNorm(),
        cmap="Dusk",
        extent=[vmin, vmax, vmin, vmax],
    )

    im.set_clim(vmin=None, vmax=cbar_vmax)

    cb = fig.colorbar(
        im,
        ax=ax,
        label="Frequency",
        labelsize=12,
        extend="both",
    )

    log_locator = LogLocator(base=10, subs="auto", numticks=10)
    cb.ax.yaxis.set_minor_locator(log_locator)
    cb.ax.tick_params(axis="both", which="both", width=1.0)
    cb.outline.set_linewidth(1.0)

    ax.format(
        xlabel=xlabel,
        ylabel=ylabel,
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
        xlim=[vmin, vmax],
        ylim=[vmin, vmax],
        xticks=second_ticks,
        yticks=second_ticks,
        grid="off",
    )

    ax.yaxis.set_tick_params(
        labelleft=True, labelright=False, left=True, right=True, which="both"
    )
    ax.xaxis.set_tick_params(
        labeltop=False, labelbottom=True, top=True, bottom=True, which="both"
    )

    for ax_ in ax:
        ax_.tick_params(axis="both", which="both", width=1)
        for spine in ax_.spines.values():
            spine.set_linewidth(1)

    out_name = f"metric_topology_{input_file.split('.pickle')[0]}_dendrimer.svg"
    out_file = os.path.join(PLOT_DIR, out_name)

    fig.savefig(out_file)
