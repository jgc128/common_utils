import numpy as np
import visdom


def get_visdom(window_name, env_name='main', close_window_if_exists=True):
    vis = visdom.Visdom(env=env_name)

    if close_window_if_exists and vis.win_exists(window_name):
        vis.close(window_name)

    return vis


def visdom_plot_losses(vis, window_name, epoch, **kwargs):
    losses = sorted(kwargs.keys())

    Y = np.array([kwargs[l] for l in losses]).reshape(1, -1)
    X = np.array([epoch for l in losses]).reshape(1, -1)

    if len(losses) == 1:
        Y = Y.flatten()
        X = X.flatten()

    if not vis.win_exists(window_name):
        update = None
        opts = {'title': window_name, 'legend': losses, 'xlabel': 'epochs', 'ylabel': 'loss'}
    else:
        update = 'append'
        opts = None

    vis.line(Y=Y, X=X, win=window_name, update=update, opts=opts)
