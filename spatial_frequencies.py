import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def get_amplitude_spectrum(img, mult = 255, img_type = 'uint8', show = False, type_convert=True):
	# first we get the fft of the image
	img_amp = np.fft.fft2(img)
	#then we turn it to the amplitude spectrum
	img_amp = np.fft.fftshift(np.abs(img_amp))
	#we ten take logarithms
	img_amp = np.log(img_amp + 1e-8)
	#we resscale to -1:+1 for displays
	img_amp = (((img_amp - np.min(img_amp))*2)/np.ptp(img_amp)) -1
	#we then multiply it out and cast it to type displayable in matplotlib
	if type_convert:
		img_amp = (img_amp * mult).astype(img_type)

	else:
		img_amp = img_amp * mult

	#we then show if we want to
	if show:
		plt.imshow(img_amp)
		plt.show()

	#and then return
	return img_amp

def get_fft(img):
	return np.fft.fft2(img)

def get_magnitude_spectrum(img, show=False, type_convert=True, img_type='uint8'):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	#print magnitude_spectrum
	if type_convert:
		magnitude_spectrum = magnitude_spectrum.astype(img_type)
	
	if show:
		#we plot the original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Original Image')
		plt.xticks([])
		plt.yticks([])
	
	#transformed image
		plt.subplot(131)
		plt.imshow(magnitude_spectrum, cmap='gray')
		plt.title('Magnitude Spectrum')
		plt.xticks([])
		plt.yticks([])
		plt.show()

	#we then return the magnitude spectrum
	return magnitude_spectrum

def get_fft_shift(img):
	f = np.fft.fft2(img)
	return np.fft.fftshift(f)
		


def high_pass_filter(img, filter_width = 10, show = False):

	fshift = get_fft_shift(img)

	rows, cols = img.shape
	crow, ccol = rows/2, cols/2
	#we remove low pass filters by simply dumping a masking window of 60 pixels width across the miage, fshift is the functiondefined to do tht
	fshift[crow-filter_width: crow+filter_width, ccol-filter_width: ccol+filter_width] = 0
	#we start to transform it back
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)

	if show:
		#get original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(img_back, cmap='gray')
		plt.title('Image after HPF')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()

	return img_back


# I can't realy claim credit for any of this. This is all copied from somewhere I forgot. But whoever they were, they did good

def butter2d_lp(shape, f, n, pxd=1):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = 1 / (1.0 + (radius / f)**(2*n))
    return filt
 
def butter2d_bp(shape, cutin, cutoff, n, pxd=1):
    """Designs an n-th order bandpass 2D Butterworth filter with cutin and
   cutoff frequencies. pxd defines the number of pixels per unit of frequency
   (e.g., degrees of visual angle)."""
    return butter2d_lp(shape,cutoff,n,pxd) - butter2d_lp(shape,cutin,n,pxd)
 
def butter2d_hp(shape, f, n, pxd=1):
    """Designs an n-th order highpass 2D Butterworth filter with cutin
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
    return 1. - butter2d_lp(shape, f, n, pxd)
 
def ideal2d_lp(shape, f, pxd=1):
    """Designs an ideal filter with cutoff frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = np.ones(shape)
    filt[radius>f] = 0
    return filt
 
def ideal2d_bp(shape, cutin, cutoff, pxd=1):
    """Designs an ideal filter with cutin and cutoff frequencies. pxd defines
   the number of pixels per unit of frequency (e.g., degrees of visual
   angle)."""
    return ideal2d_lp(shape,cutoff,pxd) - ideal2d_lp(shape,cutin,pxd)
 
def ideal2d_hp(shape, f, n, pxd=1):
    """Designs an ideal filter with cutin frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
    return 1. - ideal2d_lp(shape, f, n, pxd)
 
def bandpass(data, highpass, lowpass, n, pxd, eq='histogram'):
    """Designs then applies a 2D bandpass filter to the data array. If n is
   None, and ideal filter (with perfectly sharp transitions) is used
   instead."""
    fft = np.fft.fftshift(np.fft.fft2(data))
    if n:
        H = butter2d_bp(data.shape, highpass, lowpass, n, pxd)
    else:
        H = ideal2d_bp(data.shape, highpass, lowpass, pxd)
    fft_new = fft * H
    new_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))    
    if eq == 'histogram':
        new_image = exposure.equalize_hist(new_image)
    return new_image


def lowpass_filter(img, show = False):
	# we get the fast fourier transform (shifted) of the original image
	fft = np.fft.fftshift(np.fft.fft2(img))
	# weget the low pass filter
	filt = butter2d_lp(img.shape, 0.1, 1, pxd = 80) # these aprams we need to experiment with
	# we get the lowpass by multiplyingfilter with fft image
	fft_new = fft * filt
	# we then reconstruct the image
	new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
	# I don't know what this does, but it might be important
	new_img = exposure.equalize_hist(new_img)
	if show:
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Low pass filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return new_img

def highpass_filter(img, show = False):
	fft = np.fft.fftshift(np.fft.fft2(img))
	filt = butter2d_hp(img.shape, 0.2,2, pxd=43)
	fft_new = fft * filt
	new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
	new_img = exposure.equalize_hist(new_img)
	if show:
		
		#get original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Highpass filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return new_img

def bandpass_filter(img, show = False):
	fft = np.fft.fftshift(np.fft.fft2(img))
	filt = butter2d_bp(img.shape, 1.50001, 1.50002,2, pxd=43)
	fft_new = fft * filt
	new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
	new_img = exposure.equalize_hist(new_img)
	if show:
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Bandpass Filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return new_img

