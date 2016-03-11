from six.moves import cPickle
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


def histogramNonzero():
    from scipy.ndimage.filters import convolve
    import os
    root = input("please enter a path where your numpy arrays are located-----")
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort
    listOfOddfiles = []
    hist = np.zeros((27))
    diffBins = {el: [] for el in range(1, 28)}
    for npy in listOfNpys:
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        skeletonUc = np.load(npy)
        convImage = convolve(np.uint8(skeletonUc), kernel, mode='constant', cval=0)
        convImage[skeletonUc == 0] = 0
        # print(np.sum(convImage == 1))
        hist += np.array([np.sum(convImage == i) for i in range(1, 28)])
        nonzeros = [i for i in np.unique(convImage).tolist() if i != 0]
        for j in nonzeros:
            l = list(diffBins[j])
            l.append(npy)
            diffBins[j] = l
        if np.max(convImage[:]) > 7:
            listOfOddfiles.append(npy)
    for j in range(1, 28):
        numArrays = len(diffBins[j])
        noRandom = int(numArrays / 10)
        np.random.random_integers(1, numArrays, noRandom)
    return hist


def saveAsbin(ng, path):
    f = open(path)
    cPickle.dump(ng, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def rotatePlot():
    # Import a few modules.
    from mayavi.modules.api import Streamline
    from mayavi.api import Engine

    # Create a new mayavi scene.
    e = Engine()
    e.start()
    s = e.new_scene()
    # A streamline.
    st = Streamline()
    e.add_module(st)
    # Position the seed center.
    st.seed.widget.center = 3.5, 0.625, 1.25
    st.streamline_type = 'tube'

    # Save the resulting image to a PNG file.
    s.scene.save('test.png')

    # Make an animation:
    for i in range(36):
        # Rotate the camera by 10 degrees.
        s.scene.camera.azimuth(10)

        # Resets the camera clipping plane so everything fits and then
        # renders.
        s.scene.reset_zoom()

        # Save the scene.
        s.scene.save_png('anim%d.png' % i)


def plot():
    from mayavi import mlab
    import os
    import numpy as np
    root = '/media/pranathi/DATA/subsubVolumethresholds'
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    for npy in listOfNpys:
        npySkeleton = npy.replace('threshold', 'skeleton')
        threshold = np.load(npy)
        skeleton = np.load(npySkeleton)
        nameImage = npySkeleton.replace('npy', 'png')
        # Make a few contours.
        # mlab.options.offscreen = False
        mlab.contour3d(np.uint8(threshold), contours=5).actor.property.representation = 'points'
        mlab.contour3d(np.uint8(skeleton), colormap='gray')
        mlab.options.offscreen = True
        mlab.savefig(nameImage, magnification=2)

if __name__ == '__main__':
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
    # import numpy as np
    # a = np.random.random((4, 4))
    # e = Engine()
    # e.start()
    # from mayavi.sources.api import ArraySource
    # src = ArraySource(scalar_data=a)
    # e.add_source(src)
    # from mayavi.filters.api import WarpScalar, PolyDataNormals
    # warp = WarpScalar()
    # e.add_filter(warp, obj=src)
    # normals = PolyDataNormals()
    # e.add_filter(normals, obj=warp)
    # from mayavi.modules.api import Surface
    # surf = Surface()
    # e.add_module(surf, obj=normals)
