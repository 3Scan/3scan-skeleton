# from scipy import stats
import numpy as np
import pywt
import matplotlib.pyplot as plt
from statsmodels.robust import stand_mad


def coef_pyramid_plot(coefs, first=0, scale='uniform', ax=None):
    """
    Parameters
    ----------
    coefs : array-like
        Wavelet Coefficients. Expects an iterable in order Cdn, Cdn-1, ...,
        Cd1, Cd0.
    first : int, optional
        The first level to plot.
    scale : str {'uniform', 'level'}, optional
        Scale the coefficients using the same scale or independently by
        level.
    ax : Axes, optional
        Matplotlib Axes instance

    Returns
    -------
    Figure : Matplotlib figure instance
        Either the parent figure of `ax` or a new pyplot.Figure instance if
        `ax` is None.
    """

    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, axisbg='lightgrey')
    else:
        fig = ax.figure

    n_levels = len(coefs)
    n = 2**(n_levels - 1)  # assumes periodic

    if scale == 'uniform':
        biggest = [np.max(np.abs(np.hstack(coefs)))] * n_levels
    else:
        # multiply by 2 so the highest bars only take up .5
        biggest = [np.max(np.abs(i)) * 2 for i in coefs]

    for i in range(first, n_levels):
        x = np.linspace(2**(n_levels - 2 - i), n - 2**(n_levels - 2 - i), 2**i)
        ymin = n_levels - i - 1 + first
        yheight = coefs[i] / biggest[i]
        ymax = yheight + ymin
        ax.vlines(x, ymin, ymax, linewidth=1.1)

    ax.set_xlim(0, n)
    ax.set_ylim(first - 1, n_levels)
    ax.yaxis.set_ticks(np.arange(n_levels - 1, first - 1, -1))
    ax.yaxis.set_ticklabels(np.arange(first, n_levels))
    ax.tick_params(top=False, right=False, direction='out', pad=6)
    ax.set_ylabel("Levels", fontsize=14)
    ax.grid(True, alpha=.85, color='white', axis='y', linestyle='-')
    ax.set_title('Wavelet Detail Coefficients', fontsize=16, position=(.5, 1.05))
    fig.subplots_adjust(top=.89)

    return fig

grey = np.load('/media/pranathi/KINGSTON/resultSlices-GoodRegion/goodResults/goodRegionGreyscale.npy')
nblck = grey.reshape(grey.size)
noisy_coefs = pywt.wavedec(nblck, 'db8', level=11, mode='per')


sigma = stand_mad(noisy_coefs[-1])
uthresh = sigma * np.sqrt(2 * np.log(len(nblck)))
denoised = noisy_coefs[:]
denoised[1:] = (pywt.thresholding.soft(i, value=uthresh) for i in denoised[1:])

signal = pywt.waverec(denoised, 'db8', mode='per')

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 8))
ax1, ax2 = axes

ax1.plot(signal)
ax1.set_xlim(0, 2**10)
ax1.set_title("Recovered Signal")
ax1.margins(.1)

ax2.plot(nblck)
ax2.set_title("Noisy Signal")

for ax in fig.axes:
    ax.tick_params(labelbottom=False, top=False, bottom=False, left=False, right=False)

fig.tight_layout()


def histogramNonzero():
    from scipy.ndimage.filters import convolve
    import os
    root = input("please enter a path where your numpy arrays are located-----")
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    for npy in listOfNpys:
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        skeletonUc = np.load(npy)
        convImage = convolve(np.uint8(skeletonUc), kernel, mode='constant', cval=0)
        convImage[skeletonUc == 0] = 0
        print(np.sum(convImage == 1))
        hist = np.array([np.sum(convImage == i) for i in range(1, 28)])
        return hist
