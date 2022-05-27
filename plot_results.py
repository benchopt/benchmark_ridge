import re
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


usetex = mpl.checkdep_usetex(True)
params = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.usetex": usetex,
}
mpl.rcParams.update(params)


# matplotlib style config
titlesize = 22
ticksize = 16
labelsize = 20

MARKERS = list(plt.Line2D.markers.keys())[:-4]
CMAP = plt.get_cmap('tab20')
#####

regex = re.compile(r'\[(.*?)\]')


SAVEFIG = True
PLOT_COLUMN = "objective_value"
fignames = ["ridge_supp_mat"]

BENCH_FILES = [
    './outputs/ridge_neurips.csv',
]
FLOATING_PRECISION = 1e-8
MIN_XLIM = 1e-5

GPU_SOLVERS = ['snapml[gpu=True]', 'cuml']


DICT_XLIM = {
    # "rcv1[scaled=False]": 1e-2,
    # "rcv1[scaled=True]": 1e-2,
    # "news20[scaled=False]": 1e-1,
    # "news20[scaled=True]": 1e-1,
    # "ijcnn1[scaled=False]": 5e-3,
    # "ijcnn1[scaled=True]": 5e-3,
    # "criteo[scaled=False]": 5e-1,
    # "criteo[scaled=True]": 5e-1,
}

YTICKS = (1e4, 1, 1e-4, 1e-8)

IDX_ROWS = [
    {
        ("bodyfat", "", "objective_value"): (0, "bodyfat"),
        ("leukemia", "", "objective_value"): (1, "leukemia"),
        ("gisette", "", "objective_value"): (2, "gisette"),
        ("n_features=1000,n_samples=500,rho=0]", "", "objective_value"): (
            3, "Simulated"
        ),
        ("n_features=1000,n_samples=500,rho=0.6", "", "objective_value"): (
            4, r"Simulated[$\rho=0.6$]"
        ),
    }
]

IDX_COLUMNS = [
    {
        ("", "lmbd=0.01]", ""): (0, r"$\lambda=0.01$"),
        ("", "lmbd=0.1]", ""): (1, r"$\lambda=0.1$"),
        ("", "lmbd=1]", ""): (2, r"$\lambda=1$"),
    }
]

all_solvers = {
    "sklearn[solver=cholesky]": "scikit-learn[cholesky]",
    "sklearn[solver=saga]": "scikit-learn[saga]",
    "GD[use_acceleration=False]": "GD",
    "cd": "CD",
    "sklearn[solver=svd]": "scikit-learn[svd]",
    "sklearn[solver=lsqr]": "scikit-learn[lsqr]",
    "GD[use_acceleration=True]": "Accelerated GD",
    "Lightning": "lightning[cd]",
    "sklearn[solver=sparse_cg]": "scikit-learn[cg]",
    "snapml[gpu=False]": "snapML[cpu]",
    "snapml[gpu=True]": "snapML[gpu]",
    # 'cuml': 'cuML[gpu]',
}

df = pd.read_csv(BENCH_FILES[0], header=0, index_col=0)
solvers = df["solver_name"].unique()
print(solvers)
STYLE = {solver_name: (CMAP(i), MARKERS[i], all_solvers[solver_name])
         for i, solver_name in enumerate(solvers)}

fontsize = 20
labelsize = 20


def filter_data_and_obj(dataset, objective, idx):
    for (p_data, p_obj, col), res in idx.items():
        if ((p_data is None or p_data in dataset)
                and (p_obj is None or p_obj in objective)):
            return (*res, col)
    return None, None, None


for figname, idx_rows, idx_cols in zip(fignames, IDX_ROWS, IDX_COLUMNS):

    plt.close('all')

    n_rows, n_cols = len(idx_rows), len(idx_cols)
    main_fig, axarr = plt.subplots(
        n_rows,
        n_cols,
        sharex='row',
        sharey='row',
        figsize=[11, 1 + 2 * n_rows],
        constrained_layout=True, squeeze=False
    )

    for bench_file in BENCH_FILES:
        df = pd.read_csv(bench_file, header=0, index_col=0)
        datasets = df["data_name"].unique()
        objectives = df["objective_name"].unique()
        solvers = df["solver_name"].unique()
        solvers = np.array(sorted(solvers, key=str.lower))
        for dataset in datasets:
            for objective in objectives:
                idx_col, clabel, obj_col = filter_data_and_obj(
                    dataset, objective, idx_cols
                )
                idx_row, rlabel, obj_col_ = filter_data_and_obj(
                    dataset, objective, idx_rows
                )
                obj_col = obj_col or obj_col_
                if None in [idx_row, idx_col]:
                    continue
                df2 = df.query(
                    'data_name == @dataset & objective_name == @objective'
                )
                ax = axarr[idx_row, idx_col]
                print(idx_row, idx_col, dataset, objective)

                if obj_col == "objective_value":
                    c_star = np.min(df2[obj_col]) - FLOATING_PRECISION
                else:
                    c_star = 0
                for i, solver_name in enumerate(all_solvers):

                    # Get style if it exists or create a new one
                    color, marker, label = STYLE.get(solver_name)

                    df3 = df2.query('solver_name == @solver_name')
                    curve = df3.groupby('stop_val').median()

                    q1 = df3.groupby('stop_val')['time'].quantile(.1)
                    q9 = df3.groupby('stop_val')['time'].quantile(.9)
                    y = curve[obj_col] - c_star

                    ls = "--" if solver_name in GPU_SOLVERS else None
                    ax.loglog(
                        curve["time"], y, color=color, marker=marker,
                        label=label, linewidth=2, markevery=3, ls=ls,
                        markersize=6,
                    )

                ax.set_xlim(DICT_XLIM.get(dataset, MIN_XLIM), ax.get_xlim()[1])

                x1, x2 = ax.get_xlim()
                x1, x2 = np.ceil(np.log10(x1)), np.floor(np.log10(x2))

                y1, y2 = ax.get_ylim()
                # ax.set_ylim(y1, 1e5 if 'criteo' not in dataset else 1e8)

                xticks = 10 ** np.arange(x1, x2+1)
                ax.set_xticks(xticks)
                axarr[idx_row, 0].set_yticks(YTICKS)

                axarr[0, idx_col].set_title(clabel, fontsize=labelsize)
                axarr[n_rows-1, idx_col].set_xlabel(
                    "Time (s)", fontsize=labelsize
                )

                ax.tick_params(axis='both', which='major', labelsize=ticksize)
                ax.grid()

                axarr[idx_row, 0].set_ylabel(rlabel, fontsize=labelsize)

    plt.show(block=False)

    # plot legend on separate fig
    leg_fig, ax2 = plt.subplots(1, 1, figsize=(20, 4))
    n_col = 4
    if n_col is None:
        n_col = len(axarr[0, 0].lines)

    # take first ax, more likely to have all solvers converging
    ax = axarr[0, 0]
    lines_ordered = list(itertools.chain(
        *[ax.lines[i::n_col] for i in range(n_col)]
    ))
    legend = ax2.legend(
        lines_ordered, [line.get_label() for line in lines_ordered],
        ncol=n_col, loc="upper center"
    )
    leg_fig.canvas.draw()
    leg_fig.tight_layout()
    width = legend.get_window_extent().width
    height = legend.get_window_extent().height
    leg_fig.set_size_inches((width / 80,  max(height / 80, 0.5)))
    plt.axis('off')
    plt.show(block=False)

    if SAVEFIG:
        Path('./figures').mkdir(exist_ok=True)
        main_fig_name = f"figures/{figname}.pdf"
        main_fig.savefig(main_fig_name, dpi=300)
        leg_fig_name = f"figures/{figname}_legend.pdf"
        leg_fig.savefig(leg_fig_name, dpi=300)
