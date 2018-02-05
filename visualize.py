from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'tab10'
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


def init_plot():
    cmap = plt.get_cmap()
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    patches = [mpatches.Patch(color=cmap.colors[i], label=str(i)) for i in range(10)]
    return fig, ax, patches

def savegif(Y_seq, labels, fig_name, path):
    fig, ax, patches = init_plot()

    def init():
        return scatter,

    def update(i):
        if (i+1) % 50 == 0:
            print('[%d / %d] Animating frames' % (i+1, len(Y_seq)))
        ax.clear()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.legend(handles=patches, loc='upper right')
        ax.scatter(Y_seq[i][:, 0], Y_seq[i][:, 1], 1, labels)
        ax.set_title('%s (epoch %d)' % (fig_name, i))
        return ax, scatter
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(Y_seq), interval=50)
    print('[*] Saving animation as %s' % path)
    anim.save(path, writer='imagemagick', fps=30)

def savepng(Y, labels, fig_name, path):
    fig, ax, patches = init_plot()
    ax.scatter(Y[:, 0], Y[:, 1], 1, labels)
    ax.set_title(fig_name)
    print('[*] Saving figure as %s' % path)
    plt.savefig(path)

def scatter(Y, labels):
    fig, ax, patches = init_plot()
    ax.scatter(Y[:, 0], Y[:, 1], 1, labels)
    plt.show()
