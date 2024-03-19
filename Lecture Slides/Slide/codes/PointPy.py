# -.-. PointPy .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
# A Python package to assist with the Digital Image Processing Lectures taught at MCI.

# D.T. McGuiness
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

# Import specific packages

import cv2 as cv  # used in digital image analysis
import numpy as np  # everything related to data analysis
from matplotlib import pyplot as plt  # all related to plotting
import matplotlib.colors  # all related to colours
from scipy import signal, special  # for use in signal analysis invoke scipy
from scipy.fft import fft2, fftshift
from matplotlib.image import imread
from numpy.fft import fft
import colour  # for colour science
from colour.plotting import *   # to install use colour-science

import skimage.io as io
from skimage import segmentation, color, graph, transform, filters, measure, img_as_float
from skimage.color import rgb2hsv, rgb2gray, rgb2hed, hed2rgb
from skimage.transform import swirl
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.filters import sobel, threshold_otsu, threshold_niblack, threshold_sauvola, threshold_multiotsu, window
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, mark_boundaries

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

# Class focused on plotting functions with a sense of visual aesthetics
class Plotting:
    # Set the colour parameter for plots to fit beamer metropolis theme
    plt.rcParams["figure.facecolor"] = "(0.98, 0.98, 0.98)"
    plt.rcParams.update({'axes.facecolor': '(0.98, 0.98, 0.98)'})
    matplotlib.colors.ColorConverter.colors['bg1'] = (0.98, 0.98, 0.98)

    @staticmethod
    def image_subplot_style(row, column, image_array, publish=None, show=None, rgb=None, title=None,
                            cmap_array=None, set_cmap=False):
        """Presents a set of images in a grid of subplots.

        :param figsize: Figure size for your subplot.
        :param title: Add title to your plot. Treated as array.
        :param cmap_array: colormap array, if none entered it is treated as None.
        :param row: Number of rows in an image.
        :param column: Number of columns in an image.
        :param image_array: Write here the images in an array you want in the plot in the order you want it to show.
        :param publish: Write the name of the file you want to save it as (.eps, 200 dpi).
        :param show: Just activates plt.show().
        :param rgb: Sets print colour to true.
        """

        # noinspection PyTypeChecker
        fig_plot, axes_plot = plt.subplots(row, column, sharex=True, sharey=True)

        for ind, ax_loop in enumerate(axes_plot.flatten()):

            Plotting.remove_borders(ax_loop)  # remove unnecessary borders

            ax_loop.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            ax_loop.imshow(image_array[ind][:][:])  # print image for subplot

            if rgb:  # convert image bgr to rgb
                if set_cmap is True:
                    ax_loop.imshow(Plotting.bgr2rgb(image_array[ind][:][:]), cmap=cmap_array[ind])
                else:
                    ax_loop.imshow(Plotting.bgr2rgb(image_array[ind][:][:]))

            if title is None:

                # Determine where to put the title on the image
                if ind < column:
                    ax_loop.set_title("(" + str(chr(ord('a') + ind)) + ")", fontsize=12)
                else:
                    ax_loop.set_xlabel("(" + str(chr(ord('a') + ind)) + ")", fontsize=12)

            else:
                # Determine where to put the title on the image
                if ind < column:
                    ax_loop.set_title(title[ind], fontsize=12)
                else:
                    ax_loop.set_xlabel(title[ind], fontsize=12)

        Plotting.printer(show, publish)

    @staticmethod
    def plot_subplot_style(fig_subplot, axes_subplot):
        """Some standard aesthetics for the matplotlib function

        :param fig_subplot: figure for the subplot.
        :param axes_subplot: axes for the subplot.
        """

        axes_subplot.spines['top'].set_visible(False)
        axes_subplot.spines['right'].set_visible(False)
        axes_subplot.xaxis.set_tick_params(width=2)
        axes_subplot.yaxis.set_tick_params(width=2)
        plt.rcParams['axes.linewidth'] = 2
        fig_subplot.tight_layout()

    @staticmethod
    def bgr2rgb(image_in_bgr):
        """Converts an image from BGR space to RGB.

        :param image_in_bgr: image in BGR format.
        :return: image in RGB format.
        """
        return cv.cvtColor(image_in_bgr, cv.COLOR_BGR2RGB)

    @classmethod
    def compare_images(cls, original, altered):
        """Creates two individual windows to showcase the filters effect

        :param original: Original unaltered image
        :param altered: Output altered image
        """
        # A static function to showcase both the original and the altered image
        cv.imshow('Original Image', original)
        cv.imshow("Output Image", altered)

        # Wait and close all windows
        cv.waitKey(0)
        cv.destroyAllWindows()

    @classmethod
    def printer(cls, show=None, publish=None):

        # To print the plot in a nice .eps file in 200 dpi format.
        if publish:
            plt.savefig(publish + ".eps", format='eps', dpi=200, bbox_inches='tight')

        # To lazy to write plt.show()
        if show:
            plt.show()

    @classmethod
    def color_loop(cls):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        return colors

    @classmethod
    def remove_borders(cls, axes_rm_borders):
        axes_rm_borders.spines['top'].set_visible(False)
        axes_rm_borders.spines['right'].set_visible(False)
        axes_rm_borders.spines['bottom'].set_visible(False)
        axes_rm_borders.spines['left'].set_visible(False)
        axes_rm_borders.tick_params(which='both', size=0, labelsize=0)


# Class containing fundamental visual aids in signal processing
class SignalFundamentals:

    @classmethod
    def convolution(cls, publish=None, show=None):
        """A simple function to showcase the effect of convolution in a plot

        :param publish: Save figure in an .eps format.
        :param show: Show the plot.
        """
        sig = np.repeat([0., 1., 0.], 100)  # Original pulse
        win = signal.windows.hann(50)  # Filter impulse response
        filtered = signal.convolve(sig, win, mode='same') / sum(win)  # filtered convoluted response

        # noinspection PyTypeChecker
        fig_conv, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        axes_array = [ax1, ax2, ax3]  # axes array
        figure_array = [sig, win, filtered]  # figure array

        # Title array
        title = ["Original pulse $f\,(t)$", "Filter impulse response $g\,(t)$", "Filtered signal $(f*g)(t)$"]

        ind = 0  # indices for loop

        for axes_conv in axes_array:  # Loop through the array
            axes_conv.plot(figure_array[ind], linewidth=4, color="orange")
            axes_conv.set_title(title[ind])
            axes_conv.margins(0, 0.1)
            Plotting.plot_subplot_style(fig_conv, axes_conv)
            ind = ind + 1

        fig_conv.tight_layout()

        Plotting.printer(show, publish)

    @staticmethod
    def convolution2d(input_array, kernel, bias=0, display=None):
        """Calculates the 2D convolution of an input and kernel matrices.

        :param input_array: Enter an input matrix
        :param kernel: kernel used in convolusion
        :param bias: enter bias value.
        :param display: Prints information to the command line
        :return: returns the convoluted value.
        """
        # kernel can be asymmetric but still needs to be odd

        k_height, k_width = kernel.shape
        m_height, m_width = input_array.shape
        k_size = max(k_height, k_width)
        padded = np.pad(input_array, (int(k_size / 2), int(k_size / 2)))

        if k_size > 1:
            if k_height == 1:
                padded = padded[1:-1, :]
            elif k_width == 1:
                padded = padded[:, 1:-1]

        # iterates through matrix, applies kernel, and sums
        output = []
        for i in range(m_height):
            for j in range(m_width):
                between = padded[i:k_height + i, j:k_width + j] * kernel
                output.append(np.sum(between))

        output = np.array(output).reshape((m_height, m_width)) + bias

        if display:
            return print(output)

        return output

    @staticmethod
    def sample_signal(sampling_interval=1 / 10000, number=100):

        # set the time array
        time_array = np.arange(0, 1, 1 / sampling_interval)

        # set the frequency array
        frequency = np.zeros(number)

        # set the amplitude array
        amplitude = np.zeros(number)

        # set the signal array
        signal_sample = np.zeros((number, len(time_array)))

        # create the signal
        for i in range(number):
            frequency[i], amplitude[i] = np.sqrt(i ** 3), np.sqrt(i)
            signal_sample[i, :] = amplitude[i] * np.sin(2 * np.pi * frequency[i] * time_array)

        return sum(signal_sample)  # returns the sum and gives a 1D array

    # Custom matrix

    @staticmethod
    def fourier_transform(input_signal=None, sr=200):

        if input_signal is None:
            input_signal = SignalFundamentals.sample_signal(sr)

        ts = 1 / sr  # sampling interval

        t = np.arange(0, 1, ts)

        # DFT Calculation
        k = np.arange(0, len(input_signal), 1)  # sampling array
        complex_matrix = np.exp(-2j * np.pi * k[:, None] * k / len(input_signal))  # calculate the complex values
        fourier_matrix = np.abs(np.matmul(input_signal, complex_matrix))  # calculate the DFT of the signal

        half = np.ceil(len(fourier_matrix) / 2)  # Calculate the one-sided spectrum

        # create one-sided frequency
        one_sided = fourier_matrix[0:int(half - 1)] * 2 / len(input_signal)

        freq = np.arange(len(fourier_matrix)) * sr / len(fourier_matrix)

        fig_fourier, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, input_signal, linewidth=4, color="orange")

        ax2.stem(freq[0:int(half - 1)], one_sided, "#2ca02c", markerfmt="bo", basefmt="-b")
        ax1.set_xlim(0, 1), ax2.set_xlim(0, sr / 2)
        ax1.set_xlabel("Time (s)"), ax2.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude"), ax2.set_ylabel("FFT Amplitude |X(freq)|")
        Plotting.plot_subplot_style(fig_fourier, ax1), Plotting.plot_subplot_style(fig_fourier, ax2)

        Plotting.printer(show=True)


class ColourScience:

    @classmethod
    def wavelength_to_rgb(cls, wavelength, gamma=0.8):
        ''' This converts a given wavelength of light to an
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).
        '''
        wavelength = float(wavelength)
        if 380 <= wavelength <= 750:
            alpha = 1.
        else:
            alpha = 0.5

        if wavelength < 380:
            wavelength = 380.
        if wavelength > 750:
            wavelength = 750.
        if 380 <= wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            r, g, b = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma, 0.0, (1.0 * attenuation) ** gamma
        elif 440 <= wavelength <= 490:
            r, g, b = 0.0, ((wavelength - 440) / (490 - 440)) ** gamma, 1.0
        elif 490 <= wavelength <= 510:
            r, g, b = 0.0, 1.0, (-(wavelength - 510) / (510 - 490)) ** gamma
        elif 510 <= wavelength <= 580:
            r, g, b = ((wavelength - 510) / (580 - 510)) ** gamma, 1.0, 0.0
        elif 580 <= wavelength <= 645:
            r, g, b = 1.0, (-(wavelength - 645) / (645 - 580)) ** gamma, 0.0
        elif 645 <= wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            r, g, b = (1.0 * attenuation) ** gamma, 0.0, 0.0
        else:
            r, g, b = 0.0, 0.0, 0.0
        return r, g, b, alpha

    @classmethod
    def colour_spectrum_plot(cls, show=None, publish=None):

        clim = (350, 780)  # Sets the limit of the human vision
        norm = plt.Normalize(*clim)
        wl = np.arange(clim[0], clim[1] + 1, 2)
        colorlist = list(zip(norm(wl), [ColourScience.wavelength_to_rgb(w) for w in wl]))
        spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

        fig, axs = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)

        wavelengths = np.linspace(200, 1000, 1000)
        spectrum = (5 + np.sin(wavelengths * 0.1) ** 2) * np.exp(-0.00002 * (wavelengths - 600) ** 2)
        plt.plot(wavelengths, spectrum, color='darkred')

        y = np.linspace(0, 6, 100)
        X, Y = np.meshgrid(wavelengths, y)

        extent = (np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))
        Plotting.plot_subplot_style(axs, )
        plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.fill_between(wavelengths, spectrum, 8, color="bg1")

        Plotting.printer(show, publish)

    @classmethod
    def cie_1931(cls):
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        illuminant = colour.SDS_ILLUMINANTS["D65"]
        patch_name = "neutral 5 (.70 D)"
        patch_sd = colour.SDS_COLOURCHECKERS["ColorChecker N Ohta"][patch_name]
        XYZ = colour.sd_to_XYZ(patch_sd, cmfs, illuminant)

        RGB = colour.models.eotf_inverse_sRGB(np.array([[79, 2, 45], [87, 12, 67]]) / 255)

        fig, axes = plot_chromaticity_diagram_CIE1931(show=False)
        Plotting.plot_subplot_style(axes, )
        axes.set_xlim(-0.1, 1), axes.set_ylim(-0.1, 1)

        xy = colour.XYZ_to_xy(XYZ)

        plt.rcParams["figure.facecolor"] = "(0.98, 0.98, 0.98)"
        plt.rcParams.update({'axes.facecolor': '(0.98, 0.98, 0.98)'})
        plt.rcParams['axes.linewidth'] = 2
        plt.savefig("colour" + ".eps", format='eps', dpi=200, bbox_inches='tight')
        plt.show()


# Class to generate a variety of noise onto a signal / image
class Noise(object):

    @staticmethod
    def gauss_noise(image, var, mean):
        """Generates gaussian noise onto an image with a given mean and variance

        :param var: variance
        :param mean: mean
        :return: noisy image and SNR
        """
        sigma = var ** 0.5
        original = cv.imread(image)

        gauss = np.random.normal(mean, sigma, original.size)
        gauss = gauss.reshape(original.shape[0], original.shape[1], original.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv.add(original, gauss)

        img1 = original
        snr = cv.PSNR(img1, img_gauss)

        return img_gauss, snr

    @staticmethod
    def add_noise(snr_dB, signal_input=None, showcase=None):
        # frequency
        frequency = 0.1
        f_sampling = 80.0  # sampling frequency

        # Amplitude of the signal
        signal_amplitude = 0.75

        # time array (x-axis)
        time_array = np.arange(1024)

        # Desired linear SNR
        snr = 10.0 ** (snr_dB / 10.0)

        if signal_input is None:
            # Measure power of signal
            signal_input = signal_amplitude \
                           * np.sin(2 * np.pi * frequency / f_sampling * time_array) \
                           * np.cos(2 * np.pi * frequency / f_sampling * time_array)

        # Signal power
        signal_power = signal_input.var()

        # Calculate required noise power for desired SNR
        time_array = signal_power / snr

        # Generate noise with calculated power
        noise_data = np.sqrt(time_array) * np.random.randn(1024)

        # Add noise to signal
        noisy_signal = signal_input + noise_data

        if showcase == "signal":

            i1, o1, n1 = Noise().add_noise(50)
            i2, o2, n2 = Noise().add_noise(40)
            i3, o3, n3 = Noise().add_noise(30)
            i4, o4, n4 = Noise().add_noise(20)
            i5, o5, n5 = Noise().add_noise(10)
            i6, o6, n6 = Noise().add_noise(1)

            o = [o1, o2, o3, o4, o5, o6]
            title = ["SNR = 50", "SNR = 40", "SNR = 30", "SNR = 20", "SNR = 10", "SNR = 1"]
            fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

            for i, ax in enumerate(fig.axes):
                ax.plot(o[i], color="orange", linewidth=2)
                Plotting.plot_subplot_style(ax, )
                ax.set_title(title[i])
                ax.set_xlim(0, 1000)
                ax.set_ylim(-1, 1)

            # plt.savefig("snr" + ".eps", format='eps', dpi=200, bbox_inches='tight')

            plt.show()

        elif showcase == "image":

            i1, s = Noise().gauss_noise("Fruit.jpg", 1, 0)
            i2, s = Noise().gauss_noise("Fruit.jpg", 5, 0)
            i3, s = Noise().gauss_noise("Fruit.jpg", 10, 0)
            i4, s = Noise().gauss_noise("Fruit.jpg", 20, 0)
            i5, s = Noise().gauss_noise("Fruit.jpg", 30, 0)
            i6, s = Noise().gauss_noise("Fruit.jpg", 40, 0)

            Plotting.image_subplot_style(2, 3, Plotting.bgr2rgb(i1), Plotting.bgr2rgb(i2), Plotting.bgr2rgb(i3),
                                         Plotting.bgr2rgb(i4), Plotting.bgr2rgb(i5), Plotting.bgr2rgb(i6), show=True)

        return signal_input, noisy_signal, time_array

    @staticmethod
    def photon_noise(mu=[1, 2, 5, 10], k=np.arange(0, 20, 0.1),
                     legend_text=[r'$\rho T = 1$', r"$\rho T = 2$", r"$\rho T = 5$",
                                  r"$\rho T = 10$"], publish=None, show=None):

        """Function to plot the photon noise experienced in image acquisition.

        :param mu: Mean value. Sets to 1,2,5,10. Different values can be entered.
        :param k: Occurrence values. Set to (0,20,0.1). Different values can be entered.
        :param legend_text: Legend text. Initial value set.
        :param publish: Prints value to an .eps files set value to the name of the file.
        :param show: Shows the value.
        """

        ind = 0  # Set the loop indicator

        for mean in mu:
            poisson = np.exp(-mean) * np.power(mean, k) / special.factorial(k)
            plt.plot(k, poisson, linewidth=4, label=legend_text[ind])
            ax = plt.gca()
            ax.set_title("Poisson Process for Photon Noise")
            ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
            ax.set_xlabel("Number of Occurrences ($p$)")
            ax.set_ylabel("$P(x=p)$")
            ind += 1

        plt.legend(loc='upper right'), plt.legend(frameon=False)

        Plotting.printer(show, publish)


class HistogramOperations(object):

    # initialise class parameters
    def __init__(self, picture):
        self.picture = picture  # required image for analysis

    # method for plotting histogram
    def hist_plot(self, publish=None, show=None, ylim=7000):
        """ Plots the histogram of an image.

        :param ylim: Y-limit axis.
        :return: returns the array or list of arrays (n) and bins.
        :param publish: publishes the image of the histogram (200 dpi, .eps).
        :param show: presents the hist plot.
        """
        input_hist = ImageProcess(self.picture).grayscale()  # read image

        fig, ax = plt.subplots()  # access plot variables

        n, bins, patches = plt.hist(input_hist.ravel(), 256, [0, 256])  # plot histogram (using matplotlib)

        plt.xticks(np.arange(0, 257, 32))  # set x tick label distances
        Plotting.plot_subplot_style(fig, ax)  # set particular properties of the plot

        ax.set_xlabel("Grayscale Intensity Value (0-256)")  # set xlabel
        ax.set_ylabel("Number of Occurrences")  # set ylabel
        ax.set_xlim([0, 256]), ax.set_ylim([0, ylim])  # set limits of axes

        Plotting.printer(show, publish)  # invoke printing function

        return n, bins

    # plot a red, green, blue distribution of the image
    def bgr_plot(self, publish=None, show=None):
        """Prints the blue-green-red histogram of an image

        :param publish: publishes the image of the histogram (200 dpi, .eps)
        :param show: presents the hist plot
        """
        # define the colours (remember cv uses bgr not rgb!!)
        colour_bgr = ('b', 'g', 'r')

        fig, ax = plt.subplots()  # access plot variables

        # loop for each individual colour
        for i, col in enumerate(colour_bgr):
            histr = cv.calcHist([cv.imread(self.picture)], [i], None, [256], [0, 256])
            plt.plot(histr, color=col, linewidth=2)
            ax = plt.gca()

            plt.xlim([0, 256])

        Plotting.plot_subplot_style(fig, ax)  # set particular properties of the plot

        plt.legend(["b", "g", "r"], ncol=3, loc="upper left", frameon=False)
        plt.xticks(np.arange(0, 257, 32))

        ax.set_xlabel("Colour Intensity Value (0-256)")
        ax.set_ylabel("Number of Occurrences")

        ax.set_xlim([0, 256]), ax.set_ylim([0, 7000])

        Plotting.printer(show, publish)

    def masking(self):
        original = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)
        # create a mask
        mask = np.zeros(original.shape[:2], np.uint8)
        mask[100:300, 100:400] = 255
        masked_original = cv.bitwise_and(original, original, mask=mask)
        # Calculate histogram with mask and without mask
        # Check third argument for mask
        hist_full = cv.calcHist([original], [0], None, [256], [0, 256])
        hist_mask = cv.calcHist([original], [0], mask, [256], [0, 256])

        return original, mask, masked_original, hist_full, hist_mask

    def contrast_stretching(self):

        original = cv.imread(self.picture)
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        altered = cv.LUT(original, table)

        return original, altered

    def hist_contrast_stretching_plot(self, show=None, publish=None):
        """ Function to plot the histograms of the two images with one being the original and the second being the image
        with equalisation applied. This function takes no arguments. Returns plotting data.
        :param show: Shows the plot.
        :param publish: Publishes the plot to .eps format set value to the name of the file."""

        # Access the original and altered images of the equalisation process
        original, altered = HistogramOperations(self.picture).contrast_stretching()

        # Set these images onto an array
        images = [original, altered]

        # Define the plotting variables
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

        axis = [ax1, ax2]  # define axis labels

        for ind in range(len(axis)):
            # calculate mean value from RGB channels and flatten to 1D array
            values = images[ind].mean(axis=2).flatten()

            # plot histogram with 255 bins (color_loop to iterate over standard matplotlib values)
            axis[ind].hist(values, 255, color=Plotting.color_loop()[ind])

            # Set x and y labels with y being the common label
            axis[ind].set_xlabel("Bins"), axis[0].set_ylabel("Frequency")
            axis[ind].set_xlim([0, 255]), axis[ind].set_ylim([0, 8000])

            # Set the plotting style
            #            Plotting.plot_subplot_style(axis[ind])
            ind += 1  # loop iterator incrementation

        fig.tight_layout()  # set the figure layout to tight

        Plotting.printer(show, publish)  # Use to set figures as show only or the print to .eps file

    def hist_equalisation(self, show=None, publish=None):
        """ Function to plot the histograms of the two images with one being the original and the second being the image
        with equalisation applied. This function takes no arguments. Returns plotting data.
        :param show: Shows the plot.
        :param publish: Publishes the plot to .eps format set value to the name of the file."""

        # Access the original and altered images of the equalisation process
        original = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)  # turn original image to grayscale
        altered = cv.equalizeHist(original)  # altered image with histogram equalised

        if show:
            # Set these images onto an array
            images = [original, altered]

            # Define the plotting variables
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

            axis = [ax1, ax2]  # define axis labels

            for ind in range(len(axis)):
                # calculate mean value from RGB channels and flatten to 1D array
                values = images[ind].mean(axis=2).flatten()

                # plot histogram with 255 bins (color_loop to iterate over standard matplotlib values)
                axis[ind].hist(values, 255, color=Plotting.color_loop()[ind])

                # Set x and y labels with y being the common label
                axis[ind].set_xlabel("Bins"), axis[0].set_ylabel("Frequency")
                axis[ind].set_xlim([0, 255]), axis[ind].set_ylim([0, 8000])

                # Set the plotting style
                #            Plotting.plot_subplot_style(axis[ind])
                ind += 1  # loop iterator incrementation

            fig.tight_layout()  # set the figure layout to tight

            plt.show()

        Plotting.printer(show, publish)  # Use to set figures as show only or the print to .eps file

        return original, altered


class ImageProcess(object):

    def __init__(self, picture):
        self.picture = picture

    def grayscale(self, option="fast"):
        """Converts an image to grayscale

        :param option: Chooses between the fast or slow method. Fast method is chosen as default.
        :return: Returns the gray image.
        """

        # Creates an empty matrix for data manipulation
        colour_image = imread(self.picture)

        heigth, width = colour_image.shape[:2]
        gray_image = np.zeros((heigth, width), np.uint8)

        # Fast option (uses numpy multiplication)

        if option == "fast":
            matrix = np.array([[[0.07, 0.72, 0.21]]])  # rgb -> grayscale conversion matrix
            gray_image = np.sum(colour_image * matrix, axis=2)

        # Slow option (uses two nested for loops)
        if option == "slow":
            for i in range(heigth):
                for j in range(width):
                    gray_image[i, j] = np.clip(
                        0.07 * colour_image[i, j, 0] + 0.72 * colour_image[i, j, 1] + 0.21 * colour_image[i, j, 2], 0,
                        255)

        return gray_image  # returns a gray image

    def fourier_image(self):
        """Converts an image to its frequency components (i.e, fourier transform)

        :return: returns two images greyed out original image and frequency of the image
        """
        image = img_as_float(ImageProcess(self.picture).grayscale())

        wimage = image * window('hann', image.shape)

        image_f = np.abs(fftshift(fft2(image)))
        wimage_f = np.abs(fftshift(fft2(wimage)))

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.ravel()
        ax[0].set_title("Original image")
        ax[0].imshow(image, cmap='gray')
        ax[1].set_title("Windowed image")
        ax[1].imshow(wimage, cmap='gray')
        ax[2].set_title("Original FFT (frequency)")
        ax[2].imshow(np.log(image_f), cmap='magma')
        ax[3].set_title("Window + FFT (frequency)")
        ax[3].imshow(np.log(wimage_f), cmap='magma')
        plt.show()

    def rgb_channels(self):
        """ Splits an RGB image (n by m by 3) to its red-blue-green components

        :rtype: returns red (n by m), blue (n by m), green (n by m) matrix value.
        """
        colour_image = imread(self.picture)  # read colour image

        # Split image to three channels
        red, green, blue = colour_image[:, :, 0], colour_image[:, :, 1], colour_image[:, :, 2]

        return red, green, blue  # returns the channel values

    def hsv_channels(self, show=None, publish=None, rgb=None):

        hsv_image = rgb2hsv(cv.imread(self.picture))

        # Split image to three channels
        hue_channel, saturation_channel, value_channel = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        title = ["HSV Image", "Hue Channel", "Saturation Channel", "Value Channel"]
        image_array = [hsv_image, hue_channel, saturation_channel, value_channel]

        Plotting.image_subplot_style(2, 2, image_array=image_array, title=title, show=show, publish=publish, rgb=rgb)

        return hsv_image, hue_channel, saturation_channel, value_channel

    def cymk_channels(self, show=None, publish=None):
        original = np.array(cv.imread(self.picture))

        K_channel = 1 - np.max(original, axis=2)
        C_channel = (1 - original[:, :, 2] - K_channel)

    def laplacian(self, show=None, publish=None, k_height=5, k_width=5, k_size=5):
        # Call the Gaussian blur function
        original, gaussian_blur = KernelOperations(self.picture).blur_gaussian(show=None,
                                                                               k_height=k_height,
                                                                               k_width=k_width)
        gray_image = ImageProcess(self.picture).grayscale()  # convert the image to gray scale
        laplacian = cv.Laplacian(gray_image, cv.CV_8UC1, ksize=k_size)  # calculate the laplacian of the image
        # Use Unsigned 8bits uchar 0~255 (CV_8UC1)
        # plotting function
        if show:
            Plotting.image_subplot_style(1, 2, image_array=[Plotting.bgr2rgb(original), laplacian],
                                         show=show,
                                         publish=publish,
                                         rgb=None)

        return original, laplacian  # return the variables for later manipulation

    def rag_thresholding(self, show=None, publish=None, compactness=20, n_segments=500, start_label=1):

        original = Plotting.bgr2rgb(cv.imread(self.picture))  # access the original image

        # Segments image using k - means clustering in Color - (x, y, z) space.
        segmented = segmentation.slic(original, compactness=compactness, n_segments=n_segments, start_label=start_label)

        # Return an RGB image where color - coded labels are painted over the image.
        out1 = color.label2rgb(segmented, original, kind='avg', bg_label=0)

        g = graph.rag_mean_color(original, segmented)
        labels2 = graph.cut_threshold(segmented, g, 10)
        out2 = color.label2rgb(labels2, original, kind='avg', bg_label=0)

        Plotting.image_subplot_style(1, 2, image_array=[out1, out2], show=show, publish=publish)

    def swirl(self, rotation=0, strength=10, radius=240, show=None, publish=None):

        original = cv.imread(self.picture)  # access the original image

        # Process the image so that it has a swirl in the centre of the image.
        swirled = swirl(original, rotation=rotation, strength=strength, radius=radius)

        # Plot the image in a subplot style.
        Plotting.image_subplot_style(1, 2, image_array=[original, swirled], show=show, publish=publish)

        return original, swirled

    def orb_feature_detector(self, rotate=180, n_keypoints=200):
        original = rgb2gray(io.imread(self.picture))  # first access the image and declare it original
        rotated = transform.rotate(original, rotate)  # rotate the image to test the feature detector

        transformed = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -100))
        warped = transform.warp(original, transformed)

        descriptor_extractor = ORB(n_keypoints=n_keypoints)

        image_array = [original, rotated, warped]
        keypoints, descriptors = [0, 0, 0], [0, 0, 0]

        ind = 0
        for im in image_array:
            descriptor_extractor.detect_and_extract(im)
            keypoints[ind] = descriptor_extractor.keypoints
            descriptors[ind] = descriptor_extractor.descriptors
            ind += 1

        matches_12 = match_descriptors(descriptors[0], descriptors[1], cross_check=True)
        matches_13 = match_descriptors(descriptors[0], descriptors[2], cross_check=True)

        fig, ax = plt.subplots(nrows=2, ncols=1)

        plt.gray()

        plot_matches(ax[0], original, rotated, keypoints[0], keypoints[1], matches_12)
        ax[0].axis('off')
        ax[0].set_title("Original Image vs. Transformed Image")

        plot_matches(ax[1], original, warped, keypoints[0], keypoints[2], matches_13)
        ax[1].axis('off')
        ax[1].set_title("Original Image vs. Transformed Image")

        plt.show()

    def segmentation_and_superpixel_algorithms_comparison(self, show=None, publish=None):

        original = Plotting.bgr2rgb(cv.imread(self.picture))

        segments_fz = felzenszwalb(original, scale=100, sigma=0.5, min_size=50)
        segments_slic = slic(original, n_segments=250, compactness=10, sigma=1, start_label=1)
        segments_quick = quickshift(original, kernel_size=3, max_dist=6, ratio=0.5)
        segments_watershed = watershed(sobel(rgb2gray(original)), markers=250, compactness=0.001)

        title = ["Felzenszwalbs's method", "SLIC", "Quickshift", "Compact watershed"]
        image_array = [mark_boundaries(original, segments_fz), mark_boundaries(original, segments_slic),
                       mark_boundaries(original, segments_quick), mark_boundaries(original, segments_watershed)]

        Plotting.image_subplot_style(2, 2, image_array=image_array, title=title, show=show, publish=publish)

    def pixelation(self, wid, hei, show=None, publish=None, rgb=None):
        original = cv.imread(self.picture)  # Input image
        height, width = original.shape[:2]  # Get input size
        w, h = (wid, hei)  # Desired "pixelated" size

        # Resize input to "pixelated" size
        temp = cv.resize(original, (w, h), interpolation=cv.INTER_LINEAR)

        # Initialize output image
        pixel = cv.resize(temp, (width, height), interpolation=cv.INTER_NEAREST)

        if show or publish:
            title = ["Original", "Pixel size: " + str(w) + " by " + str(h)]
            Plotting.image_subplot_style(1, 2, image_array=[original, pixel],
                                         show=show,
                                         publish=publish,
                                         rgb=rgb,
                                         title=title)

        return original, pixel

    def edge_detection_canny(self, lower=100, upper=200, show=None, publish=None, rgb=None):
        """Implement a canny edge detection

        :param rgb:
        :param publish:
        :param show:
        :param lower: lower threshold
        :param upper: higher threshold
        :return: original image, altered image
        """
        original = cv.imread(self.picture)  # read original image file
        altered = cv.Canny(original, lower, upper)  # apply canny edge detection

        Plotting.image_subplot_style(1, 2, image_array=[original, altered], show=show, publish=publish, rgb=rgb)

        return original, altered

    def edge_detection(self, show=None, publish=None, option=None):

        original_grayscale = ImageProcess(self.picture).grayscale()  # read original image in grayscale

        if option == "Sobel":
            # Here we define the matrices associated with the Sobel filter
            g_x = np.array([[1.0, 0.0, -1.0],
                            [2.0, 0.0, -2.0],
                            [1.0, 0.0, -1.0]])
            g_y = np.array([[1.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, -2.0, -1.0]])

        if option == "Scharr":
            # Here we define the matrices associated with the Scharr filter
            g_x = np.array([[47.0, 0.0, -47.0],
                            [162.0, 0.0, -162.0],
                            [47.0, 0.0, -47.0]])
            g_y = np.array([[47.0, 162.0, 47.0],
                            [0.0, 0.0, 0.0],
                            [-47.0, -162.0, -47.0]])

        if option == "Prewitt":
            # Here we define the matrices associated with the Scharr filter
            g_x = np.array([[1.0, 0.0, -1.0],
                            [1.0, 0.0, -1.0],
                            [1.0, 0.0, -1.0]])
            g_y = np.array([[1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, -1.0, -1.0]])

        [rows, columns] = np.shape(original_grayscale)  # we need to know the shape of the input grayscale image

        # initialization of the output image array (all elements are 0)
        grad_x, grad_y = np.zeros(shape=(rows, columns)), np.zeros(shape=(rows, columns))
        sobel_filtered_image = np.zeros(shape=(rows, columns))

        # Now we "sweep" the image in both x and y directions and compute the output
        for i in range(rows - 2):
            for j in range(columns - 2):
                grad_x[i + 1, j + 1] = np.sum(np.multiply(g_x, original_grayscale[i:i + 3, j:j + 3]))  # x direction
                grad_y[i + 1, j + 1] = np.sum(np.multiply(g_y, original_grayscale[i:i + 3, j:j + 3]))  # y direction
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(
                    grad_x[i + 1, j + 1] ** 2 + grad_y[i + 1, j + 1] ** 2)  # calculate the "hypotenuse"

        Plotting.image_subplot_style(2, 2, image_array=[original_grayscale, grad_x, grad_y, sobel_filtered_image],
                                     show=show,
                                     publish=publish)

        return grad_x, grad_y, sobel_filtered_image

    def edge_detection_prewitt(self, show=None, publish=None, rgb=None, figsize=(8, 6)):

        original = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)  # read original image in grayscale
        img_gaussian = cv.GaussianBlur(original, (3, 3), 0)

        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv.filter2D(img_gaussian, -1, kernel_x)
        img_prewitty = cv.filter2D(img_gaussian, -1, kernel_y)

        title = ["Prewitt-X", "Prewitt-Y", "Prewitt Combined"]

        Plotting.image_subplot_style(1, 3, image_array=[img_prewittx, img_prewitty, img_prewittx + img_prewittx],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb,
                                     title=title,
                                     figsize=figsize)

        return img_prewittx, img_prewitty

    def line_transform_hough(self, args, show=None, publish=None, rgb=None, aperture_size=3,
                             min_line_length=400,
                             max_line_gap=10):

        original = ImageProcess(self.picture).grayscale()  # read original image in grayscale

        edges = cv.Canny(original, 50, 100, apertureSize=aperture_size)

        if args == "norm":
            lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
                added_lines = cv.line(cv.imread(self.picture), (x1, y1), (x2, y2), (0, 0, 255), 2)

        elif args == "prob":
            lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_line_length,
                                   maxLineGap=max_line_gap)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                added_lines = cv.line(original, (x1, y1), (x2, y2), (0, 255, 0), 2)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), added_lines], show=show,
                                     publish=publish, rgb=rgb)

    def thresholding_simple(self, show=None, publish=None, rgb=None):
        original = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)  # read original image in grayscale
        ret, thresh1 = cv.threshold(original, 127, 255, cv.THRESH_BINARY)
        ret, thresh2 = cv.threshold(original, 127, 255, cv.THRESH_BINARY_INV)
        ret, thresh3 = cv.threshold(original, 127, 255, cv.THRESH_TRUNC)
        ret, thresh4 = cv.threshold(original, 127, 255, cv.THRESH_TOZERO)
        ret, thresh5 = cv.threshold(original, 127, 255, cv.THRESH_TOZERO_INV)
        titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']

        image_array = [original, thresh1, thresh2, thresh3, thresh4, thresh5]

        Plotting.image_subplot_style(2, 3, image_array, show=show, title=titles, rgb=rgb, publish=publish)

        return thresh1, thresh2, thresh3, thresh4, thresh5

    def thresholding_adaptive(self, threshold=127, show=None, publish=None, rgb=None):
        original = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)  # read original image in grayscale
        blurred = cv.medianBlur(original, 5)  # create median blur on the image
        ret, th1 = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        th3 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        titles = ["Original Image", "Global Thresholding (v = " + str(threshold) + ")",
                  "Adaptive Mean Thresholding", "Adaptive Gaussian Thresholding"]
        image_array = [blurred, th1, th2, th3]

        Plotting.image_subplot_style(2, 2, image_array, show=show, title=titles, rgb=rgb, publish=publish)

        return blurred, th1, th2, th3

    def thresholding_niblack_sauvola(self, w_size=25, niblack_k=0.8, show=None, publish=None, set_cmap=False, rgb=None):

        image = rgb2gray(io.imread(self.picture))
        binary_global = image > threshold_otsu(image)

        window_size = w_size
        thresh_niblack = threshold_niblack(image, window_size=window_size, k=niblack_k)
        thresh_sauvola = threshold_sauvola(image, window_size=window_size)

        binary_niblack = image > thresh_niblack
        binary_sauvola = image > thresh_sauvola

        titles = ["Original", "Global Threshold", "Niblack Threshold", "Sauvola Threshold"]

        Plotting.image_subplot_style(2, 2, image_array=[image, binary_global, binary_niblack, binary_sauvola],
                                     cmap_array=["gray", "gray", "gray", "gray"],
                                     show=show, title=titles, publish=publish, set_cmap=set_cmap, rgb=rgb)

    def thresholding_multi_otsu(self, show=None, publish=None):
        # The input image.
        image = ImageProcess(self.picture).grayscale()
        # Applying multi-Otsu threshold for the default value, generating
        # three classes.
        thresholds = threshold_multiotsu(image)

        # Using the threshold values, we generate the three regions.
        regions = np.digitize(image, bins=thresholds)

        titles = ["Original", "Multi-Otsu Result"]
        Plotting.image_subplot_style(1, 2, image_array=[image, regions], cmap_array=["gray", "jet"], show=show,
                                     title=titles, publish=publish)

        return image, thresholds, regions

    @classmethod
    def contour_finding(cls, grid_size=100, z_data=None, show=None, publish=None):

        # Construct some test data
        x, y = np.ogrid[-np.pi:np.pi:grid_size * 1j, -np.pi:np.pi:grid_size * 1j]

        if z_data is None:
            z_data = np.sin(np.exp(np.sin(x) ** 3 + np.cos(y) ** 2))

        # Find contours at a constant value of 0.8
        contours = measure.find_contours(z_data, 0.8)

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(z_data, cmap="gray")

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        ax.axis('image')
        ax.set_xticks([]), ax.set_yticks([])
        Plotting.printer(show, publish)

    def application_tissue_sample(self, show=None, publish=None):

        # RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.
        ihc_hed = rgb2hed(Plotting.bgr2rgb(cv.imread(self.picture)))

        # Create an RGB image for each of the stains
        unsplit = np.zeros_like(ihc_hed[:, :, 0])
        h_channel = hed2rgb(np.stack((ihc_hed[:, :, 0], unsplit, unsplit), axis=-1))
        e_channel = hed2rgb(np.stack((unsplit, ihc_hed[:, :, 1], unsplit), axis=-1))
        d_channel = hed2rgb(np.stack((unsplit, unsplit, ihc_hed[:, :, 2]), axis=-1))

        if show:
            titles = ["Original Image", "Hematoxylin", "Eosin", "DAB"]
            Plotting.image_subplot_style(2, 2,
                                         image_array=[Plotting.bgr2rgb(cv.imread(self.picture)), h_channel, e_channel,
                                                      d_channel],
                                         show=show,
                                         publish=publish,
                                         rgb=None,
                                         title=titles)

        return unsplit, h_channel, e_channel, d_channel

    def application_butterworth_filter(self):
        image = ImageProcess(self.picture).grayscale()

        # cutoff frequencies as a fraction of the maximum frequency
        cutoffs = [.02, .08, .38, 0.44]

        def get_filtered(image, cutoffs, squared_butterworth=True, order=3.0, npad=0):
            """Lowpass and highpass butterworth filtering at all specified cutoffs.

            Parameters
            ----------
            image : ndarray
                The image to be filtered.
            cutoffs : sequence of int
                Both lowpass and highpass filtering will be performed for each cutoff
                frequency in `cutoffs`.
            squared_butterworth : bool, optional
                Whether the traditional Butterworth filter or its square is used.
            order : float, optional
                The order of the Butterworth filter

            Returns
            -------
            lowpass_filtered : list of ndarray
                List of images lowpass filtered at the frequencies in `cutoffs`.
            highpass_filtered : list of ndarray
                List of images highpass filtered at the frequencies in `cutoffs`.
            """

            lowpass_filtered, highpass_filtered = [], []
            for cutoff in cutoffs:
                lowpass_filtered.append(
                    filters.butterworth(
                        image,
                        cutoff_frequency_ratio=cutoff,
                        order=order,
                        high_pass=False,
                        squared_butterworth=squared_butterworth,
                        npad=npad,
                    )
                )
                highpass_filtered.append(
                    filters.butterworth(
                        image,
                        cutoff_frequency_ratio=cutoff,
                        order=order,
                        high_pass=True,
                        squared_butterworth=squared_butterworth,
                        npad=npad,
                    )
                )
            return lowpass_filtered, highpass_filtered

        def plot_filtered(lowpass_filtered, highpass_filtered, cutoffs):
            """Generate plots for paired lists of lowpass and highpass images."""
            fig, axes = plt.subplots(2, 1 + len(cutoffs), figsize=(12, 8))
            fontdict = dict(fontsize=14, fontweight='bold')

            axes[0, 0].imshow(image.astype('uint8'), cmap='gray')
            axes[0, 0].set_title('original', fontdict=fontdict)
            axes[1, 0].set_axis_off()

            for i, c in enumerate(cutoffs):
                axes[0, i + 1].imshow(lowpass_filtered[i].astype('uint8'), cmap='gray')
                axes[0, i + 1].set_title(f'lowpass, c={c}', fontdict=fontdict)
                axes[1, i + 1].imshow(highpass_filtered[i].astype('uint8'), cmap='gray')
                axes[1, i + 1].set_title(f'highpass, c={c}', fontdict=fontdict)

            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            return fig, axes

        # Perform filtering with the (squared) Butterworth filter at a range of
        # cutoffs.
        lowpasses, highpasses = get_filtered(image, cutoffs, squared_butterworth=True)

        fig, axes = plot_filtered(lowpasses, highpasses, cutoffs)
        titledict = dict(fontsize=18, fontweight='bold')
        fig.text(0.5, 0.95, '(squared) Butterworth filtering (order=3.0, npad=0)',
                 fontdict=titledict, horizontalalignment='center')

        plt.show()


class Nyquist_Sampling(object):
    plt.rcParams.update({'axes.facecolor': '(0.98, 0.98, 0.98)'})
    plt.rcParams['axes.linewidth'] = 2

    def sample_signal(duration=1, f_sampling=100):
        """Generate example signal

        Args:
            duration: Duration (in seconds) of signal (Default value = 1)
            f_sampling: Sampling rate (in samples per second) (Default value = 100)

        Returns:
            x: Signal
            time_array: Time axis (in seconds)
        """
        signal_array = int(f_sampling * duration)
        time_array = np.arange(signal_array) / f_sampling
        sample = 1 * np.sin(2 * np.pi * (1.9 * time_array - 0.3)) \
                 + 0.5 * np.sin(2 * np.pi * (6.1 * time_array - 0.1)) \
                 + 0.1 * np.sin(2 * np.pi * (20 * time_array - 0.2))
        return sample, time_array

    def sampling_equidistant(signal, time, f_sampling, duration=None):
        """Equidistant sampling of interpolated signal

        Notebook: PCP_08_signal.ipynb

        Args:
            signal: Signal to be interpolated and sampled
            time: Time axis (in seconds) of x_1
            f_sampling: Sampling rate used for equidistant sampling
            duration: Duration (in seconds) of sampled signal (Default value = None)

        Returns:
            x_2: Sampled signal
            t_2: time axis (in seconds) of sampled signal
        """
        if duration is None:
            duration = len(time) * time[1]
        array = int(f_sampling * duration)

        sampled_time = np.arange(array) / f_sampling
        sampled_signal = np.interp(sampled_time, time, signal)
        return sampled_signal, sampled_time

    def signal_reconstruction(x, t, t_sinc):
        """Reconstruction from sampled signal using sinc-functions

        Args:
            x: Sampled signal
            t: Equidistant discrete time axis (in seconds) of x
            t_sinc: Equidistant discrete time axis (in seconds) of signal to be reconstructed

        Returns:
            x_sinc: Reconstructed signal having time axis t_sinc
        """
        f_s = 1 / t[1]
        x_sinc = np.zeros(len(t_sinc))
        for n in range(0, len(t)):
            x_sinc += x[n] * np.sinc(f_s * t_sinc - n)
        return x_sinc

    @staticmethod
    def showcase(show=None, publish=None):

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True)
        f_sampling_array = [64, 32, 16, 8, 4, 2]
        axes_array = [ax1, ax2, ax3, ax4, ax5, ax6]  # axes array

        Fs_1 = 100
        x_1, t_1 = Nyquist_Sampling.sample_signal(f_sampling=Fs_1)

        ind = 0  # indices for loop

        # Loop through the array
        for axes in axes_array:
            x, t = Nyquist_Sampling.sampling_equidistant(x_1, t_1, f_sampling_array[ind])
            t_sinc = t_1
            x_sinc = Nyquist_Sampling.signal_reconstruction(x, t, t_sinc)

            axes.plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
            axes.plot(t_sinc, x_sinc, color="orange", label='Reconstructed signal', linewidth=2)
            axes.stem(t, x, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')

            axes.set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t[1]))
            axes.set_xlabel('Time (seconds)')
            Plotting.plot_subplot_style(fig, axes)
            plt.ylim([-1.8, 1.8]), plt.xlim([t_1[0], t_1[-1]])

            ind += 1

        Plotting.printer(show, publish)


class KernelOperations(object):
    """Examples of Blur functions used in Digital Image Processing."""

    def __init__(self, picture, matrix=np.ones(5)):
        """Initialise the variables

        :param picture: input image
        :param matrix: kernel of a size of your choice
        """
        self.picture = picture
        self.kernel = matrix

    # Simple Averaging Function
    def blur_averaging(self, show=None, rgb=None, publish=None, k_size=5):
        averaged_blur = cv.blur(cv.imread(self.picture), (k_size, k_size))  # apply the averaging filter

        # Invoke the sub-plotting function
        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), averaged_blur],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), averaged_blur

    # Gaussian Blur Function
    def blur_gaussian(self, k_height=5, k_width=5, show=None, rgb=None, publish=None):
        gaussian_blur = cv.GaussianBlur(cv.imread(self.picture), (k_height, k_width), 0)  # apply the filter

        if show:
            # Invoke the sub-plotting function
            Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), gaussian_blur],
                                         show=show,
                                         publish=publish,
                                         rgb=rgb)

        return cv.imread(self.picture), gaussian_blur

    def blur_median(self, k_size=5, show=None, rgb=None, publish=None):
        median_blur = cv.medianBlur(cv.imread(self.picture), k_size)  # apply the filter

        if show:
            # Invoke the sub-plotting function
            Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), median_blur],
                                         show=show,
                                         publish=publish,
                                         rgb=rgb)

        return cv.imread(self.picture), median_blur

    def bilateral_filter(self, show=None, publish=None, rgb=None, pixel_d=5, sigma_colour=75, sigma_space=75):
        """Apply a bilateral filter to an image.

        :param show: Presents the comparative image to the user.
        :param publish: Prints the image with the name being the entered value (200 dpi, .eps)
        :param rgb: Converts images from bgr to rgb.
        :return: Returns original image and the filtered image.
        """
        original = cv.imread(self.picture)  # set original image
        bilateral = cv.bilateralFilter(original, pixel_d, sigma_colour, sigma_space)  # apply the filter

        # Invoke the sub-plotting function
        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), bilateral],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), bilateral  # returns the original and the filtered image

    def morph_erosion(self, iterations=1, show=None, publish=None, rgb=None):
        """ Apply the erosion morphology process to an image

        :param iterations: Number of iterations
        :param show: Presents the comparative image to the user.
        :param publish: Prints the image with the name being the entered value (200 dpi, .eps)
        :param rgb: Converts images from bgr to rgb.
        :return: Returns original image and the filtered image.
        """
        erosion = cv.erode(cv.imread(self.picture), self.kernel, iterations=iterations)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), erosion],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), erosion

    def morph_dilation(self, iterations=1, show=None, publish=None, rgb=None):
        dilation = cv.dilate(cv.imread(self.picture), self.kernel, iterations=iterations)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), dilation],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), dilation

    def morph_opening(self, show=None, publish=None, rgb=None):
        opening = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_OPEN, self.kernel)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), opening],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), opening

    def morph_closing(self, show=None, publish=None, rgb=None):
        closing = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_CLOSE, self.kernel)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), closing],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), closing

    def morph_gradient(self, show=None, publish=None, rgb=None):
        """It is the difference between dilation and erosion of an image. The result will look like the outline of
        the object."""
        gradient = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_GRADIENT, self.kernel)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), gradient],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), gradient

    def morph_tophat(self, show=None, publish=None, rgb=None):
        """It is the difference between dilation and erosion of an image. The result will look like the outline of
        the object."""

        hat = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_TOPHAT, self.kernel)

        Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), hat],
                                     show=show,
                                     publish=publish,
                                     rgb=rgb)

        return cv.imread(self.picture), hat

    def morph_blackhat(self, show=None, publish=None, rgb=None):
        """It is the difference between dilation and erosion of an image. The result will look like the outline of
        the object."""

        black_hat = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_BLACKHAT, self.kernel)

        if show:
            Plotting.image_subplot_style(1, 2, image_array=[cv.imread(self.picture), black_hat],
                                         show=show,
                                         publish=publish,
                                         rgb=rgb)

        return cv.imread(self.picture), black_hat


o, a200 = ImageProcess("Fruit.jpg").edge_detection_canny(lower=100, upper=500)
o, a300 = ImageProcess("Fruit.jpg").edge_detection_canny(lower=200, upper=500)
o, a400 = ImageProcess("Fruit.jpg").edge_detection_canny(lower=300, upper=500)
o, a500 = ImageProcess("Fruit.jpg").edge_detection_canny(lower=400, upper=500)

im_ar = [a200, a300, a400, a500]

Plotting.image_subplot_style(2,2, image_array=im_ar,show=True,publish="canny_b")