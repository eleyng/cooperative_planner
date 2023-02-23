""" Various auxiliary utilities """
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import yaml

WINDOW_W = 1200
WINDOW_H = 600
hspace, vspace = (WINDOW_W / 100, WINDOW_H / 100)

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ffbb11",
    "#2ca02c",
    "#d62728",
    "#bbf90f",
    "#00FFFF",
    "#000080",
    "#FFFF00",
    "#008080",
]


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def load_cfg(yaml_filepath):
    # Read YAML experiment definition file
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.safe_load(stream)
    # cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def save_checkpoint(state, is_best, filename, best_filename):
    """Save state in filename. Also save in best_filename if is_best."""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def plot_map(table_init, table_goal, obstacles, fig=None, ax=None):

    ca = plt.gca()
    obstacle_size = 50  # FIXME: hardcoded
    ca.add_patch(
        patches.Circle(
            (table_init[0], table_init[1]),
            radius=obstacle_size,
            facecolor=(0.0, 0.0, 0.0, 0.9),  # black
            zorder=0,
            alpha=0.5,
        )
    )

    for i in range(obstacles.shape[0]):
        obstacle_w = obstacle_size
        obstacle_h = obstacle_size
        obstacle_x = obstacles[i, 0]  # - obstacle_w / 2.0
        obstacle_y = obstacles[i, 1]  # - obstacle_h / 2.0
        if obstacle_x == 0 or obstacle_y == 0:
            continue
        ca.add_patch(
            patches.Rectangle(
                (obstacle_x - obstacle_w / 2, obstacle_y + obstacle_h / 2),
                obstacle_w,
                obstacle_h,
                facecolor=(1.0, 0.0, 0.0, 0.9),
                zorder=0,
                alpha=0.8,
            )
        )
    ca.add_patch(
        patches.Rectangle(
            (table_goal[0] - 175 / 2, table_goal[1] - 275 / 2),
            175,
            275,
            facecolor=(1.0, 0.72, 0.06, 0.9),  # gold
            zorder=0,
            alpha=0.2,
        )
    )

    plt.gca().set_aspect("equal")  # , adjustable='box')
    plt.xlim([0, WINDOW_W])
    plt.ylim([0, WINDOW_H])
    plt.axis("off")


def plot_trajectory(data, fig=None, init_x=None, init_y=None, scaling=None, **kwargs):
    plt.gca()
    plt.scatter(init_x, init_y, c="black")
    plt.plot(
        (init_x[-1] + scaling * torch.cumsum(data[..., 0], dim=0))
        .detach()
        .cpu()
        .numpy(),
        (init_y[-1] + scaling * torch.cumsum(data[..., 1], dim=0))
        .detach()
        .cpu()
        .numpy(),
        **kwargs
    )


def plot_theta(data, fig=None, **kwargs):
    plt.gca()
    plt.plot(data.detach().cpu().numpy(), **kwargs)


def save_plot(filename, fig, title=None, t=None):

    plt.xlabel("xlabel", fontsize=18)
    plt.ylabel("ylabel", fontsize=16)
    plt.savefig(filename, dpi=100)
    plt.close()
