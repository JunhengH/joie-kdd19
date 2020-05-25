# some useful tools
import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft

def circular_correlation(h, t):
    return tf.real(tf.spectral.ifft(tf.multiply(tf.conj(tf.spectral.fft(tf.complex(h, 0.))), tf.spectral.fft(tf.complex(t, 0.)))))

def np_ccorr(h, t):
    return ifft(np.conj(fft(h)) * fft(t)).real
    
def make_hparam_string(dim, onto_ratio, type_ratio, lr):
	# input params: dim, onto_ratio, type_ratio, lr, 
	return "dim_%s_onto_%s_type_%s_lr_%.0E" % (dim, onto_ratio, type_ratio,lr)