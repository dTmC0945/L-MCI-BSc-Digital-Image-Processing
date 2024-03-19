# PointPy ---------------------------------
# A Python package to assist with the Digital Image Processing Lectures taught at MCI.

# D.T. McGuiness
# -----------------------------------------

# Import specific packages

import cv2 as cv  # used in digital image analysis
import numpy as np  # everything related to data analysis
from matplotlib import pyplot as plt  # all related to plotting
import matplotlib.gridspec as gridspec  # for equal size grid for sub-plotting
from scipy import signal  # for use in signal analysis
import inspect
from numpy.fft import fft, ifft
import scipy as sp
import scipy.fftpack as fft


# Class to simplify matplotlib parameters and standardise some stuff.
class Plotting:
    # set the background colour of plots to metropolis theme
    plt.rcParams["figure.facecolor"] = "(0.98, 0.98, 0.98)"
    plt.rcParams.update({'axes.facecolor': '(0.98, 0.98, 0.98)'})
    plt.rcParams['axes.linewidth'] = 2

    @staticmethod
    def image_subplot_style(row, column, *args, publish=None, show=None, colour=None):
        """Presents a set of images in a grid of subplots.

        :param row: Number of rows in an image
        :param column: Number of columns in an image
        :param args: Write here the images you want to be in the plot in the order you want it to show.
        :param publish: Wite the name of the file you want to save it as (.eps, 200 dpi)
        :param show: Just activates plt.show()
        :param colour: Sets print colour to true
        """

        # Set the grid size
        gs = gridspec.GridSpec(row, column)

        ind = 0  # loop parameters

        for photo in args:
            ax = plt.subplot(gs[ind])  # set axis variable

            ax.imshow(photo)  # print image for subplot

            if colour:
                ax.imshow(Plotting.bgr2rgb(photo))  # convert image bgr to rgb

            # Determine where to put the title on the image
            if ind < column:
                ax.set_title("(" + str(chr(ord('a') + ind)) + ")", fontsize=12)
            else:
                ax.set_xlabel("(" + str(chr(ord('a') + ind)) + ")", fontsize=12)

            # Remove the ticks and labels on the figure
            plt.tick_params(left=False,
                            right=False,
                            labelleft=False,
                            labelbottom=False,
                            bottom=False)

            ind = ind + 1  # don't forget to increment the loop parameter

        gs.update(wspace=0.05, hspace=0)  # set the spacing between axes.

        # Condition if you want to print the figure
        if publish:
            plt.savefig(publish + ".eps", format='eps', dpi=200, bbox_inches='tight')

        # To lazy to write plt.show()
        if show:
            plt.show()

    @staticmethod
    def plot_subplot_style(axes):
        """Some standard aestetics for the matplotlib function

        :param axes: axes for the subplot.
        """

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.xaxis.set_tick_params(width=2)
        axes.yaxis.set_tick_params(width=2)

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


# Fundamental plots and operations related to signal / image processing
class Fundamentals:

    @classmethod
    def convolution(cls, publish=None, show=None):
        """A simple function to showcase the effect of convolution

        :param publish: Save figure in an .eps format.
        :param show: Show the plot.
        """
        sig = np.repeat([0., 1., 0.], 100)  # Original pulse
        win = signal.windows.hann(50)  # Filter impulse response
        filtered = signal.convolve(sig, win, mode='same') / sum(win)  # filtered convoluted response

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        axes_array = [ax1, ax2, ax3]  # axes array
        figure_array = [sig, win, filtered]  # figure array

        # Title array
        title = ["Original pulse $f\,(t)$",
                 "Filter impulse response $g\,(t)$",
                 "Filtered signal $(f*g)(t)$"]

        ind = 0  # indices for loop

        # Loop through the array
        for axes in axes_array:
            axes.plot(figure_array[ind], linewidth=4, color="orange")

            axes.set_title(title[ind])
            axes.margins(0, 0.1)
            Plotting.plot_subplot_style(axes)
            ind = ind + 1

        fig.tight_layout()

        if publish:
            fig.savefig(publish + ".eps", format='eps', dpi=200)

        if show:
            fig.show()  # show the plot


# Class to generate a variety of noise onto a signal / image
class Noise(object):

    def __init__(self, picture):
        self.picture = picture

    def gauss_noise(self, var, mean):
        """Generates gaussian noise onto an image with a given mean and variance

        :param var: variance
        :param mean: mean
        :return: noisy image and SNR
        """
        sigma = var ** 0.5
        original = cv.imread(self.picture)

        gauss = np.random.normal(mean, sigma, original.size)
        gauss = gauss.reshape(original.shape[0], original.shape[1], original.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv.add(original, gauss)

        img1 = original
        snr = cv.PSNR(img1, img_gauss)

        return img_gauss, snr

    @staticmethod
    def add_noise(SNR_dB):
        # frequency
        f1 = 0.1
        Fs = 80.0  # sampling frequency

        # amplitudes
        amp1 = 0.75

        # time
        n = np.arange(1024)

        # Desired linear SNR
        snr = 10.0 ** (SNR_dB / 10.0)
        print("Linear snr = ", snr)

        # Measure power of signal
        signal1 = amp1 * np.sin(2 * sp.pi * f1 / Fs * n) * np.cos(2 * sp.pi * f1 / Fs * n)
        p1 = signal1.var()
        print("Power of signal1 = ", p1)

        # Calculate required noise power for desired SNR
        n = p1 / snr
        print("Noise power = ", n)

        print("Calculated SNR =  %f dB" % (10 * np.log10(p1 / n)))

        # Generate noise with calculated power
        w = np.sqrt(n) * np.random.randn(1024)

        # Add noise to signal
        s1 = signal1 + w

        return signal1, s1, n


class ImageProcess(object):

    def __init__(self, picture):
        self.picture = picture

    def equalisation(self):
        original = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)
        assert self.picture is not None, "file could not be read, check with os.path.exists()"

        equalised = cv.equalizeHist(original)

        return original, equalised

    def pixellation(self, wid, hei):
        original = cv.imread(self.picture)  # Input image
        height, width = original.shape[:2]  # Get input size
        w, h = (wid, hei)  # Desired "pixelated" size

        # Resize input to "pixelated" size
        temp = cv.resize(original, (w, h), interpolation=cv.INTER_LINEAR)

        # Initialize output image
        pixellised = cv.resize(temp, (width, height), interpolation=cv.INTER_NEAREST)

        return original, pixellised

    def fourier_image(self):
        """Converts an image to its frequency components (i.e, fourier transform)

        :return: returns two images greyed out original image and frequency of the image
        """
        original = cv.imread(self.picture)  # retrieve the original image
        greyed = cv.cvtColor(original, cv.COLOR_BGR2GRAY)  # turn the image to grayscale
        phase = np.fft.fft2(greyed)  # convert the image to fft
        phase_shift = np.fft.fftshift(phase)  # Shift the zero-frequency component to the center of the spectrum
        phase_spectrum = np.angle(phase_shift)  # retrive the angle

        return greyed, phase_spectrum

    def edge_detection_canny(self, lower=100, upper=200):
        """Implement a canny edge detection

        :param lower: lower threshold
        :param upper: higher threshold
        :return: original image, altered image
        """
        original = cv.imread(self.picture)  # read original image file
        altered = cv.Canny(original, lower, upper)  # apply canny edge detection

        return original, altered

    def edge_detection_sobel(self):
        original = cv.imread(self.picture)
        gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        cv.imshow('grad X', grad_x)
        cv.imshow('grad Y', grad_y)
        cv.imshow('Sobel Image', grad)
        cv.waitKey()

    def edge_detection_prewitt(self):
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv.filter2D(img_gaussian, -1, kernely)

    def line_transform_hough(self, args):
        if args == "norm":
            img = cv.imread(self.picture)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 50, 150, apertureSize=3)
            lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv.imshow('Original Image', Fundamentals.original())
            cv.imshow("Output Image", img)
            # Wait and close all windows
            cv.waitKey(0)
            cv.destroyAllWindows()

        elif args == "prob":

            img = cv.imread(cv.samples.findFile(self.picture))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 50, 150, apertureSize=3)
            lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imwrite('houghlines5.jpg', img)

            cv.imshow("Output Image", img)
            # Wait and close all windows
            cv.waitKey(0)
            cv.destroyAllWindows()

        else:
            raise ValueError("Only acceptable options are: \"norm\" and \"prob\"")


class KernelOperations(object):
    """Examples of Blur functions used in Digital Image Processing."""

    def __init__(self, picture, matrix):
        """Initialise the variables

        :param picture: input image
        :param matrix: kernel of a size of your choice
        """
        self.picture = picture
        self.kernel = matrix

    # Simple Averaging Function
    def blur_averaging(self, publish=None):
        original = cv.imread(self.picture)  # set original image
        altered = cv.blur(original, self.kernel)  # apply the averaging filter

        return original, altered

    # Gaussian Blur Function
    def blur_gaussian(self, publish=None):
        original = cv.imread(self.picture)  # set original image
        altered = cv.GaussianBlur(original, (5, 5), 0)  # apply the filter

        return original, altered

    def blur_median(self, publish=None):
        original = cv.imread(self.picture)  # set original image
        altered = cv.medianBlur(original, 5)  # apply the filter

        return original, altered

    def bilateral_filter(self, publish=None):
        original = cv.imread(self.picture)  # set original image
        altered = cv.bilateralFilter(original, 9, 75, 75)  # apply the filter

        return original, altered

    def morph_erosion(self):
        erosion = cv.erode(cv.imread(self.picture), self.kernel, iterations=1)

        return cv.imread(self.picture), erosion

    def morph_dilation(self):
        dilation = cv.dilate(cv.imread(self.picture), self.kernel, iterations=1)

        return cv.imread(self.picture), dilation

    def morph_opening(self):
        opening = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_OPEN, self.kernel)

        return cv.imread(self.picture), opening

        # def morph_closing(self):
        closing = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_CLOSE, self.kernel)

        return cv.imread(self.picture), closing

    def morph_gradient(self):
        """It is the difference between dilation and erosion of an image. The result will look like the outline of
        the object."""
        gradient = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_GRADIENT, self.kernel)

        return cv.imread(self.picture), gradient

    def morph_tophat(self):
        """It is the difference between dilation and erosion of an image. The result will look like the outline of
        the object."""

        hat = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_TOPHAT, self.kernel)

        return cv.imread(self.picture), hat

    def morph_blackhat(self):
        """It is the difference between dilation and erosion of an image. The result will look like the outline of
        the object."""

        black_hat = cv.morphologyEx(cv.imread(self.picture), cv.MORPH_BLACKHAT, self.kernel)

        return cv.imread(self.picture), black_hat


class Thresholding:

    @staticmethod
    def simple():
        original = Fundamentals.original()
        ret, thresh1 = cv.threshold(original, 127, 255, cv.THRESH_BINARY)
        ret, thresh2 = cv.threshold(original, 127, 255, cv.THRESH_BINARY_INV)
        ret, thresh3 = cv.threshold(original, 127, 255, cv.THRESH_TRUNC)
        ret, thresh4 = cv.threshold(original, 127, 255, cv.THRESH_TOZERO)
        ret, thresh5 = cv.threshold(original, 127, 255, cv.THRESH_TOZERO_INV)
        titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        images = [original, thresh1, thresh2, thresh3, thresh4, thresh5]
        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def adaptive():
        original = Fundamentals.original()
        img = cv.medianBlur(original, 5)
        ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


class LineDetection(object):

    def __init__(self, picture):
        self.picture = picture


class ImageSegmentation:

    @classmethod
    def Watershed(cls):
        img = cv.imread('Coins.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        cv.imshow("Output Image", ret)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]

        cv.imshow("Output Image", markers)

        # Wait and close all windows
        cv.waitKey(0)
        cv.destroyAllWindows()


class CornerDetection:

    @classmethod
    def fun(cls):
        filename = 'Chess.jpg'
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv.imshow('dst', img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

    @classmethod
    def fun2(cls):
        img = cv.imread('Chess.jpg')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv.circle(img, (x, y), 3, 255, -1)
        plt.imshow(img), plt.show()


class HistogramOperations(object):

    # initialise class parameters
    def __init__(self, picture):
        self.picture = picture  # required image for analysis

    # method for plotting histogram
    def hist_plot(self, publish=None, name=None):

        input_hist = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)  # read image

        plt.hist(input_hist.ravel(), 256, [0, 256])  # plot histogram (using matplotlib)

        ax = plt.gca()  # access the axis information

        heights, positions, patches = ax.hist(input_hist.ravel(),
                                              40,
                                              color='skyblue',
                                              alpha=0.5,
                                              density=True,
                                              histtype='stepfilled',
                                              ec="k")

        Plotting.plot_style(ax, x="Bins", y="Number of Pixels")

        ax.set_xlim([0, 256]), ax.set_ylim([0, 7000])
        ax.set_xticks(np.arange(0, 257, step=32))

    # plot a red, green, blue distribution of the image
    def rgb_plot(self, publish=None, name=None):
        color = ('r', 'g', 'b')  # define the colours
        low = np.linspace(0, 256, 256, endpoint=True)
        # loop for each individual colour
        for i, col in enumerate(color):
            histr = cv.calcHist([cv.imread(self.picture)], [i], None, [256], [0, 256])
            fig = plt.plot(histr, color=col, linewidth=2)

            plt.xlim([0, 256])
            ax = plt.gca()

        Plotting.plot_style(ax, x="Bins", y="Number of Pixels")
        plt.legend(["r", "g", "b"], ncol=3, loc="upper left")

        ax.set_xticks(np.arange(0, 257, step=32))
        ax.set_xlim([0, 256])
        ax.set_ylim([0, 7000])

        if publish:
            text = name
            plt.savefig(text + ".eps", format='eps')
        plt.show()  # show the plot

    def masking(self):
        img = cv.imread(self.picture, cv.IMREAD_GRAYSCALE)
        # create a mask
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[100:300, 100:400] = 255
        masked_img = cv.bitwise_and(img, img, mask=mask)
        # Calculate histogram with mask and without mask
        # Check third argument for mask
        hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
        hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
        plt.subplot(221), plt.imshow(img, 'gray')
        plt.subplot(222), plt.imshow(mask, 'gray')
        plt.subplot(223), plt.imshow(masked_img, 'gray')
        plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
        plt.xlim([0, 256])
        plt.show()

    def contrast_stretching(self, publish=None):

        original = cv.imread(self.picture)
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        altered = cv.LUT(original, table)

        Fundamentals.output_print(cv.imread(self.picture), altered)

        if publish:
            Plotting.subplot_style(self.picture, altered, str(inspect.stack()[0][3]), 200)
        plt.show()


class Bitwise(object):

    def __init__(self, picture1, picture2):
        picture1.self = picture1
        picture2.self = picture2

    def AND(picture1, picture2):
        img1 = cv.imread(picture1)
        img2 = cv.imread(picture2)

        # cv2.bitwise_and is applied over the
        # image inputs with applied parameters
        dest_and = cv.bitwise_and(img2, img1, mask=None)

        # the window showing output image
        # with the Bitwise AND operation
        # on the input images
        cv.imshow('Bitwise And', dest_and)

        # De-allocate any associated memory usage
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

    def OR(picture1, picture2):
        img1 = cv.imread(picture1)
        img2 = cv.imread(picture2)

        # cv2.bitwise_or is applied over the
        # image inputs with applied parameters
        dest_or = cv.bitwise_or(img2, img1, mask=None)

        # the window showing output image
        # with the Bitwise OR operation
        # on the input images
        cv.imshow('Bitwise OR', dest_or)

        # De-allocate any associated memory usage
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()


image = "Fruit.jpg"
kernel = np.ones((5, 5), np.uint8)
#
# img=cv.imread(image)
# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# dft = np.fft.fft2(img)
# dft_shift = np.fft.fftshift(dft)
# phase_spectrum = np.angle(dft_shift)
#
# ax1 = plt.subplot(1,2,1)
# ax1.imshow(img, cmap='gray')
#
# ax2 = plt.subplot(1,2,2)
# ax2.imshow(phase_spectrum, cmap='gray')
#
# plt.savefig("fourier.eps", format='eps', dpi=200, bbox_inches='tight')
# plt.show()

# o, a = ImageProcess(image).fourier_image()
#
# Plotting.image_subplot_style(1, 2, o, a, show=True)
# class Sampling(object):
#     plt.rcParams.update({'axes.facecolor': '(0.98, 0.98, 0.98)'})
#     plt.rcParams['axes.linewidth'] = 2
#
#     def generate_signal(duration=1, f_sampling=100):
#         """Generate example signal
#
#         Args:
#             duration: Duration (in seconds) of signal (Default value = 1)
#             f_sampling: Sampling rate (in samples per second) (Default value = 100)
#
#         Returns:
#             x: Signal
#             t: Time axis (in seconds)
#         """
#         array = int(f_sampling * duration)
#         t = np.arange(array) / f_sampling
#         x = 1 * np.sin(2 * np.pi * (1.9 * t - 0.3))
#         x += 0.5 * np.sin(2 * np.pi * (6.1 * t - 0.1))
#         x += 0.1 * np.sin(2 * np.pi * (20 * t - 0.2))
#         return x, t
#
#     def sampling_equidistant(signal, time, f_sampling, dur=None):
#         """Equidistant sampling of interpolated signal
#
#         Notebook: PCP_08_signal.ipynb
#
#         Args:
#             signal: Signal to be interpolated and sampled
#             time: Time axis (in seconds) of x_1
#             f_sampling: Sampling rate used for equidistant sampling
#             dur: Duration (in seconds) of sampled signal (Default value = None)
#
#         Returns:
#             x_2: Sampled signal
#             t_2: time axis (in seconds) of sampled signal
#         """
#         if dur is None:
#             dur = len(time) * time[1]
#         array = int(f_sampling * dur)
#
#         sampled_time = np.arange(array) / f_sampling
#         sampled_signal = np.interp(sampled_time, time, signal)
#         return sampled_signal, sampled_time
#
#     Fs_1 = 100
#     x_1, t_1 = generate_signal(f_sampling=Fs_1)
#
#     Fs_2 = 20
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#
#     def reconstruction_sinc(x, t, t_sinc):
#         """Reconstruction from sampled signal using sinc-functions
#
#         Notebook: PCP_08_signal.ipynb
#
#         Args:
#             x: Sampled signal
#             t: Equidistant discrete time axis (in seconds) of x
#             t_sinc: Equidistant discrete time axis (in seconds) of signal to be reconstructed
#
#         Returns:
#             x_sinc: Reconstructed signal having time axis t_sinc
#         """
#         f_s = 1 / t[1]
#         x_sinc = np.zeros(len(t_sinc))
#         for n in range(0, len(t)):
#             x_sinc += x[n] * np.sinc(f_s * t_sinc - n)
#         return x_sinc
#
#     def plot_signal_reconstructed(vol, t_1, x_1, t_2, x_2, t_sinc, x_sinc, figsize=(8, 2.2)):
#         """Plotting three signals
#
#         Notebook: PCP_08_signal.ipynb
#
#         Args:
#             t_1: Time axis of original signal
#             x_1: Original signal
#             t_2: Time axis for sampled signal
#             x_2: Sampled signal
#             t_sinc: Time axis for reconstructed signal
#             x_sinc: Reconstructed signal
#             figsize: Figure size (Default value = (8, 2.2))
#         """
#         plt.figure(figsize=figsize)
#         axs[0].plt.plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#         axs[0].plt.stem(t_2, x_2, linefmt='r:', markerfmt='r.', basefmt='None', label='Samples',
#                         use_line_collection=True)
#         axs[0].plt.plot(t_sinc, x_sinc, color='orange', label='Reconstructed signal')
#         plt.title(r'Sampling rate $F_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#         plt.xlabel('Time (seconds)')
#         plt.ylim([-1.8, 1.8])
#         plt.xlim([t_1[0], t_1[-1]])
#         plt.legend(loc='upper right', framealpha=1)
#         plt.tight_layout()
#
#     fig, axs = plt.subplots(3, 2, sharex=True)
#     Fs_2 = 64
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#     t_sinc = t_1
#     x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
#
#     axs[0, 0].plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#     axs[0, 0].plot(t_sinc, x_sinc, color="orange", label='Reconstructed signal', linewidth=2)
#     axs[0, 0].stem(t_2, x_2, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')
#
#     axs[0, 0].set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#     axs[0, 0].set_xlabel('Time (seconds)')
#     axs[0, 0].spines['top'].set_visible(False)
#     axs[0, 0].spines['right'].set_visible(False)
#     axs[0, 0].xaxis.set_tick_params(width=2)
#     axs[0, 0].yaxis.set_tick_params(width=2)
#     plt.ylim([-1.8, 1.8])
#     plt.xlim([t_1[0], t_1[-1]])
#
#     Fs_2 = 32
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#     t_sinc = t_1
#     x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
#
#     axs[1, 0].plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#     axs[1, 0].plot(t_sinc, x_sinc, color="orange",  label='Reconstructed signal', linewidth=2)
#     axs[1, 0].stem(t_2, x_2, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')
#
#     axs[1, 0].set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#     axs[1, 0].set_xlabel('Time (seconds)')
#
#     axs[1, 0].spines['top'].set_visible(False)
#     axs[1, 0].spines['right'].set_visible(False)
#     axs[1, 0].xaxis.set_tick_params(width=2)
#     axs[1, 0].yaxis.set_tick_params(width=2)
#
#     plt.ylim([-1.8, 1.8])
#     plt.xlim([t_1[0], t_1[-1]])
#
#     Fs_2 = 16
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#     t_sinc = t_1
#     x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
#
#     axs[2, 0].plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#     axs[2, 0].plot(t_sinc, x_sinc, color="orange",  label='Reconstructed signal', linewidth=2)
#     axs[2, 0].stem(t_2, x_2, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')
#
#     axs[2, 0].set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#     axs[2, 0].set_xlabel('Time (seconds)')
#
#     axs[2, 0].spines['top'].set_visible(False)
#     axs[2, 0].spines['right'].set_visible(False)
#     axs[2, 0].xaxis.set_tick_params(width=2)
#     axs[2, 0].yaxis.set_tick_params(width=2)
#
#     plt.ylim([-1.8, 1.8])
#     plt.xlim([t_1[0], t_1[-1]])
#
#     Fs_2 = 8
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#     t_sinc = t_1
#     x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
#
#     axs[0, 1].plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#     axs[0, 1].plot(t_sinc, x_sinc, color="orange",  label='Reconstructed signal', linewidth=2)
#     axs[0, 1].stem(t_2, x_2, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')
#
#     axs[0, 1].set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#     axs[0, 1].set_xlabel('Time (seconds)')
#
#     axs[0, 1].spines['top'].set_visible(False)
#     axs[0, 1].spines['right'].set_visible(False)
#     axs[0, 1].xaxis.set_tick_params(width=2)
#     axs[0, 1].yaxis.set_tick_params(width=2)
#     plt.ylim([-1.8, 1.8])
#     plt.xlim([t_1[0], t_1[-1]])
#
#     Fs_2 = 4
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#     t_sinc = t_1
#     x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
#
#     axs[1, 1].plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#     axs[1, 1].plot(t_sinc, x_sinc, color="orange",  label='Reconstructed signal', linewidth=2)
#     axs[1, 1].stem(t_2, x_2, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')
#
#     axs[1, 1].set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#     axs[1, 1].set_xlabel('Time (seconds)')
#
#     axs[1, 1].spines['top'].set_visible(False)
#     axs[1, 1].spines['right'].set_visible(False)
#     axs[1, 1].xaxis.set_tick_params(width=2)
#     axs[1, 1].yaxis.set_tick_params(width=2)
#     plt.ylim([-1.8, 1.8])
#     plt.xlim([t_1[0], t_1[-1]])
#
#     Fs_2 = 2
#     x_2, t_2 = sampling_equidistant(x_1, t_1, Fs_2)
#     t_sinc = t_1
#     x_sinc = reconstruction_sinc(x_2, t_2, t_sinc)
#
#     axs[2, 1].plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
#     axs[2, 1].plot(t_sinc, x_sinc, color="orange", label='Reconstructed signal', linewidth=2)
#     axs[2, 1].stem(t_2, x_2, linefmt='k:', markerfmt='.', basefmt='None', label='Samples')
#
#     axs[2, 1].set_title(r'Sampling rate $f_\mathrm{s} = %.0f$' % (1 / t_2[1]))
#     axs[2, 1].set_xlabel('Time (seconds)')
#
#     axs[2, 1].spines['top'].set_visible(False)
#     axs[2, 1].spines['right'].set_visible(False)
#     axs[2, 1].xaxis.set_tick_params(width=2)
#     axs[2, 1].yaxis.set_tick_params(width=2)
#     plt.ylim([-1.8, 1.8])
#     plt.xlim([t_1[0], t_1[-1]])
#
#
#     plt.tight_layout()
#     plt.savefig("shannon.eps", format='eps', dpi=200, bbox_inches='tight')
#
#     plt.show()
#

# img = cv.imread(image, 0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
#
# plt.rcParams["figure.facecolor"] = "(0.98, 0.98, 0.98)"
# plt.rcParams.update({'axes.facecolor': '(0.98, 0.98, 0.98)'})
# plt.rcParams['axes.linewidth'] = 2
#
# # sampling rate
# sr = 2000
# # sampling interval
# ts = 1.0 / sr
# t = np.arange(0, 1, ts)
#
# freq = 1.
# x = 3 * np.sin(2 * np.pi * freq * t)
#
# freq = 4
# x += np.sin(2 * np.pi * freq * t)
#
# freq = 7
# x += 0.5 * np.sin(2 * np.pi * freq * t)
#
# # plt.plot(t, x, 'r')
# # plt.ylabel('Amplitude')
#
# X = fft(x)
# N = len(X)
# n = np.arange(N)
# T = N / sr
# freq = n / T
#
# ax = plt.subplot(211)
#
# plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
# # plt.xlabel('Freq (Hz)')
# # plt.ylabel('FFT Amplitude |X(freq)|')
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.xaxis.set_tick_params(width=2)
# # ax.yaxis.set_tick_params(width=2)
# Plotting().plot_subplot_style(ax, show=True)
#
# # plt.xlim(0, 10)
# #
# ax = plt.subplot(212)
# plt.plot(t, ifft(X), linewidth=4, color='orange')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude')
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # plt.tight_layout()
#
#
#
# i1, o1, n1 = Noise(image).add_noise(50)
# i2, o2, n2 = Noise(image).add_noise(40)
# i3, o3, n3 = Noise(image).add_noise(30)
# i4, o4, n4 = Noise(image).add_noise(20)
# i5, o5, n5 = Noise(image).add_noise(10)
# i6, o6, n6 = Noise(image).add_noise(1)
#
# o = [o1, o2, o3, o4, o5, o6]
# title = ["SNR = 50", "SNR = 40", "SNR = 30", "SNR = 20", "SNR = 10", "SNR = 1"]
# fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
#
# for i, ax in enumerate(fig.axes):
#     ax.plot(o[i], color="orange", linewidth=2)
#     Plotting.plot_subplot_style(ax)
#     ax.set_title(title[i])
#     ax.set_xlim(0, 1000)
#     ax.set_ylim(-1, 1)
#
# plt.savefig("snr" + ".eps", format='eps', dpi=200, bbox_inches='tight')
#
# plt.show()


# i, s = Noise(image).gauss_noise(1, 0)#
#
# i2, s = Noise(image).gauss_noise(5, 0)
# i3, s = Noise(image).gauss_noise(10, 0)
#
# i4, s = Noise(image).gauss_noise(20, 0)
# i5, s = Noise(image).gauss_noise(30, 0)
# i6, s = Noise(image).gauss_noise(40, 0)
#
# Plotting.image_subplot_style(2,3,Plotting.bgr2rgb(i),Plotting.bgr2rgb(i2),Plotting.bgr2rgb(i3),Plotting.bgr2rgb(i4),Plotting.bgr2rgb(i5),Plotting.bgr2rgb(i6),show=True,publish="snr_image")

#
# import numpy as np
# import cv2
# import math
#
# # read input as grayscale
# img = cv2.imread('Stripes.png', 0)
# hh, ww = img.shape
#
# # get min and max and mean values of img
# img_min = np.amin(img)
# img_max = np.amax(img)
# img_mean = int(np.mean(img))
#
# # pad the image to dimension a power of 2
# hhh = math.ceil(math.log2(hh))
# hhh = int(math.pow(2, hhh))
# www = math.ceil(math.log2(ww))
# www = int(math.pow(2, www))
# imgp = np.full((hhh, www), img_mean, dtype=np.uint8)
# imgp[0:hh, 0:ww] = img
#
# # convert image to floats and do dft saving as complex output
# dft = cv2.dft(np.float32(imgp), flags=cv2.DFT_COMPLEX_OUTPUT)
#
# # apply shift of origin from upper left corner to center of image
# dft_shift = np.fft.fftshift(dft)
#
# # extract magnitude and phase images
# mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
#
# # get spectrum
# spec = np.log(mag) / 20
# min, max = np.amin(spec, (0, 1)), np.amax(spec, (0, 1))
#
# # threshold the spectrum to find bright spots
# thresh = (255 * spec).astype(np.uint8)
# thresh = cv2.threshold(thresh, 155, 255, cv2.THRESH_BINARY)[1]
#
# # cover the center rows of thresh with black
# yc = hhh // 2
# cv2.line(thresh, (0, yc), (www - 1, yc), 0, 5)
#
# # get the y coordinates of the bright spots
# points = np.column_stack(np.nonzero(thresh))
# print(points)
#
# # create mask from spectrum drawing horizontal lines at bright spots
# mask = thresh.copy()
# for p in points:
#     y = p[0]
#     cv2.line(mask, (0, y), (www - 1, y), 255, 5)
#
# # apply mask to magnitude such that magnitude is made black where mask is white
# mag[mask != 0] = 0
#
# # convert new magnitude and old phase into cartesian real and imaginary components
# real, imag = cv2.polarToCart(mag, phase)
#
# # combine cartesian components into one complex image
# back = cv2.merge([real, imag])
#
# # shift origin from center to upper left corner
# back_ishift = np.fft.ifftshift(back)
#
# # do idft saving as complex output
# img_back = cv2.idft(back_ishift)
#
# # combine complex components into original image again
# img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
#
# # crop to original size
# img_back = img_back[0:hh, 0:ww]
#
# # re-normalize to 8-bits in range of original
# min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
# notched = cv2.normalize(img_back, None, alpha=img_min, beta=img_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
# # cv2.imshow("ORIGINAL", img)
# # cv2.imshow("PADDED", imgp)
# # cv2.imshow("MAG", mag)
# # cv2.imshow("PHASE", phase)
# # cv2.imshow("SPECTRUM", spec)
# # cv2.imshow("THRESH", thresh)
# # cv2.imshow("MASK", mask)
# # cv2.imshow("NOTCHED", notched)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # write result to disk
# cv2.imwrite("pattern_lines_spectrum.png", (255 * spec).clip(0, 255).astype(np.uint8))
# cv2.imwrite("pattern_lines_thresh.png", thresh)
# cv2.imwrite("pattern_lines_mask.png", mask)
# cv2.imwrite("pattern_lines_notched.png", notched)
#
# fig, axes = plt.subplots(nrows=4, ncols=1, constrained_layout=True, figsize=(6, 6))
#
# o = [img, (255 * spec).clip(0, 255).astype(np.uint8), mask, notched]
#
# for i, ax in enumerate(fig.axes):
#     ax.imshow(Plotting.bgr2rgb(o[i]))
#     Plotting.plot_subplot_style(ax)
#     # Remove the ticks and labels on the figure
#     ax.tick_params(left=False,
#                     right=False,
#                     labelleft=False,
#                     labelbottom=False,
#                     bottom=False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.set_ylabel("a")
# plt.savefig("fourier_example" + ".eps", format='eps', dpi=200, bbox_inches='tight')
# plt.show()
#

HistogramOperations('Fruit.jpg').hist_plot()