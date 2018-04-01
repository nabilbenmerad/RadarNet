
import numpy as np
import matplotlib.pyplot as plt

from flowlib import flow_to_image


__all__ = [
    'plot_middlebury_color_code',
    'plot_flow_vector',
    'plot_flow_color',
    'flow_color'
]


def plot_middlebury_color_code(xlim=(-10, 10), ylim=(-10, 10), resolution=100, **kwargs):
    '''For some reason, colors are flipped on the y axis
    with respect to real middlebury color code.
    '''

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)
    C = np.concatenate([X, Y], axis=-1)
    plt.title('Middlebury color code')
    plt.imshow(flow_to_image(C), origin='lower', **kwargs)
    plt.plot([0, resolution-1], [resolution / 2, resolution / 2], c='black')
    plt.plot([resolution / 2, resolution / 2], [0, resolution-1], c='black')
    plt.axis('off')


def plot_flow_vector(flow, flow_target=None, img=None):
    '''Plots vector field with/without associated image
            according to method given.
            flow: np.ndarray of with size (2, x_dim, y_dim)
            flow_target: np.ndarray of with size (2, x_dim, y_dim)
            img: np.ndarray with size (x_dim, y_dim)

    '''

    if img is not None:
        plt.imshow(img, origin='lower')

    X, Y = np.meshgrid(range(flow.shape[1]), range(flow.shape[2]))

    if flow_target is not None:
        plt.quiver(X[::3, ::3], Y[::3, ::3], flow_target[0, ::3, ::3], flow_target[1, ::3, ::3],
                    color='r', pivot='mid', units='inches')

    plt.quiver(X[::3, ::3], Y[::3, ::3], flow[0, ::3, ::3], flow[1, ::3, ::3],
               pivot='mid', units='inches')



def plot_flow_color(flow, **kwargs):
    plt.imshow(flow_to_image(flow.transpose(1, 2, 0)), origin='lower', **kwargs)
    

def flow_color(flow):
    return flow_to_image(flow.transpose(1, 2, 0))
