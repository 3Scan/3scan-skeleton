from six.moves import cPickle
import numpy as np
import copy
import pywt
import matplotlib.pyplot as plt
from statsmodels.robust import stand_mad
from scipy.ndimage.filters import convolve
import os
from scipy.misc import imsave, imread
from scipy import ndimage
from skimage.filters import threshold_otsu


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
    root = input("please enter a path where your numpy arrays are located-----")
    formatOfFiles = 'npy'
    listOfNpys = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfNpys.sort()
    listOfOddfiles = []
    hist = np.zeros((27))
    for npy in listOfNpys:
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        skeletonUc = np.load(npy)
        convImage = convolve(np.uint8(skeletonUc), kernel, mode='constant', cval=0)
        convImage[skeletonUc == 0] = 0
        hist += np.array([np.sum(convImage == i) for i in range(1, 28)])
        if np.max(convImage[:]) > 7:
            listOfOddfiles.append(npy)
    saveAsbin(listOfOddfiles, "oddfiles.bin")
    saveAsbin(hist, "histogram.bin")


def saveAsbin(ng, path):
    f = open(path, 'wb')
    cPickle.dump(ng, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def mosaic(npy):
    threshold = np.load(npy)
    skeleton = np.load(npy.replace("threshold", "skeleton"))
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    npyIndex = npy.replace(".npy", "")
    strs = npyIndex.split('/')[-1].split('_')
    i, j, k = [int(s) for s in strs if s.isdigit()]
    i = i - 190
    subList = listOfJpgs[i - 9: i + 10]
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    count = 0
    for fileName in subList:
        print(fileName)
        subVolume[count][:][:] = imread(fileName)
        count += 1
    subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
    subSubvolume = 255 - subSubvolume
    interpolatedIm = ndimage.interpolation.zoom(subSubvolume, [5 / 0.7037037, 1, 1], order=2, prefilter=False)
    for i in range(threshold.shape[0]):
        imsave('sectwodMaximaslicesGoodRegionOT/' + 'thresholdot%i.png' % i, threshold[i] * 255)

    for i in range(subSubvolume.shape[0]):
        imsave(root + 'twodGreyslicesBadRegion(298, 2307, 7067)/' + 'GreyGR%i.png' % i, subSubvolume[i])

    for i in range(interpolatedIm.shape[0]):
        imsave('sectwodInterpGreyslicesbadRegion/' + 'interpGreyGR%i.png' % i, interpolatedIm[i])

    for i in range(skeleton.shape[0]):
        imsave('sectwodSkeletonslicesbadRegion/' + 'skeletonBR%i.png' % i, skeleton[i])
    numOdd = 2
    for I in range(0, skeleton.shape[0]):
        plt.subplot(1, 3, 1)
        # maxip = np.amax(interpolatedIm[I:I + 7], 0)
        plt.imshow(interpolatedIm[I], cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(threshold[I], cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(skeleton[I], cmap='gray')
        plt.savefig('Mosaic%i.png' % I, bbox_inches='tight')
    convImage = convolve(np.uint8(skeleton), kernel, mode='constant', cval=0)
    convImage[skeleton == 0] = 0
    x = list(range(1, 28))
    hist = np.array([np.sum(convImage == i) for i in x])
    f = open('/home/pranathi/mosaic%i/hist.vasc' % numOdd, 'wb')
    cPickle.dump(hist, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    plt.plot(x, hist)
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.savefig('/home/pranathi/mosaic%i/hist.png' % numOdd, bbox_inches='tight')


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


def histogramEqualize(subSubvolume):
    l1 = [y for y in subSubvolume.ravel() if y != 0]
    hist, bins = np.histogram(l1, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    subSubvolume2 = cdf[subSubvolume]
    return subSubvolume2


def avgimprovisedOtsu(subVolumeCopy):
    histData, bins = np.histogram(subVolumeCopy.flatten(), 256, [0, 256])
    sumation = np.sum(histData * bins[0:-1])
    weight = np.sum(histData)
    avg = sumation / weight
    subVolumeCopy[subVolumeCopy < int(avg)] = 0
    l1 = [y for y in subVolumeCopy.ravel() if y != 0]
    t = threshold_otsu(np.array(l1))
    print("avg value improvised otsu threshold", t)
    o = subVolumeCopy > t
    return o, t


def avgimprovisedequalisedOtsu(subVolumeCopy):
    histData, bins = np.histogram(subVolumeCopy.flatten(), 256, [0, 256])
    sumation = np.sum(histData * bins[0:-1])
    weight = np.sum(histData)
    avg = sumation / weight
    subVolumeCopy[subVolumeCopy < int(avg)] = 0
    l1 = [y for y in subVolumeCopy.ravel() if y != 0]
    hist, bins = np.histogram(l1, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    subSubvolume2 = cdf[subVolumeCopy]
    l1 = [y for y in subVolumeCopy.ravel() if y != 0]
    t = threshold_otsu(np.array(l1))
    print("avg value improvised otsu threshold", t)
    o = subSubvolume2 > t
    return o, t


def smoothedHistotsu(subSubvolume):
    histData, bins = np.histogram(subSubvolume.flatten(), 256, [0, 256])
    bins = bins[0:-1]
    Mean = sum(bins * histData) / sum(histData)
    standardDeviation = np.sqrt(abs(sum((bins - Mean) ** 2 * histData) / sum(histData)))
    maxVal = histData.max()
    # fit = lambda t: maxVal * np.exp(- (t - Mean) ** 2 / (2 * standardDeviation ** 2))
    # plt.plot(fit(bins))
    # plt.show()
    print("Parameters height, standard deviation and mean are {}, {}, {}".format(maxVal, standardDeviation, Mean))
    subSubvolumecopy = copy.deepcopy(subSubvolume)
    subSubvolumecopy[subSubvolume < int(Mean + (3 * standardDeviation))] = 0
    l1 = [y for y in subSubvolumecopy.ravel() if y != 0]
    smoothedThresh = threshold_otsu(np.array(l1))
    print("smoothed Thresh is", smoothedThresh)
    thresholded = subSubvolume > smoothedThresh
    return thresholded, smoothedThresh


def localMaximaImprovisedOtsu(subSubvolume):
    subSubvolumeint = subSubvolume.astype(int)
    maskx = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.int64)
    masky = np.array([[[0, -1, 0], [0, -1, 0], [0, -1, 0]],
                     [[0, -1, 0], [0, 8, 0], [0, -1, 0]],
                     [[0, -1, 0], [0, -1, 0], [0, -1, 0]]], dtype=np.int64)
    maskz = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]],
                     [[0, 0, 0], [-1, 8, -1], [0, 0, 0]],
                     [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]], dtype=np.int64)
    convolvex = convolve(subSubvolumeint, maskx)
    convolvey = convolve(subSubvolumeint, masky)
    convolvez = convolve(subSubvolumeint, maskz)
    maxi = np.maximum(convolvex, convolvey)
    maxf = np.maximum(maxi, convolvez)
    maxfi = np.ones(maxi.shape, dtype=bool)
    maxfi[maxf < 0] = 0
    subSubvolumenew = subSubvolume * maxfi
    l1 = [y for y in subSubvolumenew.ravel() if y != 0]
    # l2 = [y for y in maxf.ravel() if y != 0]
    # plt.subplot(2, 1, 1)
    # plt.hist(subSubvolume.ravel(), bins=256, range=(0.0, 256), fc='k', ec='k')
    # plt.subplot(2, 1, 2)
    # plt.hist(np.array(l1), bins=256, range=(0.0, 256), fc='k', ec='k')
    # plt.savefig("histogram_{}_{}_{}.png".format(i, j, k))
    newThresh = threshold_otsu(np.array(l1))
    print("local maxima improvised threshold", newThresh)
    o = subSubvolume > newThresh
    return o, newThresh


def groundTruthBinarystack(root):
    subVolumeThresh = np.zeros((19, 135, 135), dtype=np.uint8)
    count = 0
    for i in range(0, 19):
        # print(fileName)
        subVolumeThresh[i][:][:] = imread(root + 'GreyGR%iGT.png' % i)
        count += 1
    subVolumeThresh[subVolumeThresh != 255] = 0


def otsuImprovements(i, j, k):
    groundRoot = input("please enter a path for your ground truth")
    maxip2 = groundTruthBinarystack(groundRoot)
    subList = listOfJpgs[i - 9: i + 10]
    subVolume = np.zeros((19, 17480, 8026), dtype=np.uint8)
    count = 0
    for fileName in subList:
        # print(fileName)
        subVolume[count][:][:] = imread(fileName)
        count += 1
    subSubvolume = subVolume[:, j - 67:j + 68, k - 67: k + 68]
    subSubvolume = 255 - subSubvolume
    oldThresh = threshold_otsu(subSubvolume)
    maxip = np.amax(subSubvolume > oldThresh, 0)
    thresholdedIm, newThresh = localMaximaImprovisedOtsu(subSubvolume)
    maxip3 = np.amax(thresholdedIm, 0)
    o, t = avgimprovisedOtsu(subSubvolume)
    maxip4 = np.amax(o, 0)
    thresholded, smoothedThresh = smoothedHistotsu(subSubvolume)
    maxip5 = np.amax(thresholded, 0)
    plt.subplot(2, 3, 1)
    plt.imshow(np.amax(subSubvolume, 0), cmap='gray')
    plt.title("grey scale mip")
    plt.subplot(2, 3, 2)
    plt.imshow(maxip2, cmap='gray')
    plt.title("ground truth mip")
    plt.subplot(2, 3, 3)
    plt.imshow(maxip, cmap='gray')
    plt.title("only otsu threshold mip and threshold is %i" % oldThresh)
    plt.subplot(2, 3, 4)
    plt.imshow(maxip3, cmap='gray')
    plt.title("local maxima improvised mip threshold is %i" % newThresh)
    plt.subplot(2, 3, 5)
    plt.imshow(maxip4, cmap='gray')
    plt.title("avg value improvised mip threshold is %i" % t)
    plt.subplot(2, 3, 6)
    plt.imshow(maxip5, cmap='gray')
    plt.title("gaussian smoothed otsu threshold is %i" % smoothedThresh)


def softThreshold():
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

if __name__ == '__main__':
    root = '/media/pranathi/DATA/ii-5016-15-ms-brain_1920/filt/'
    formatOfFiles = 'png'
    listOfJpgs = [os.path.join(root, files) for files in os.listdir(root) if formatOfFiles in files]
    listOfJpgs.sort()
    # i = 766 - 190; j = 6507; k = 3987
