"""
Legacy RIR building routines (formerly the Cython extension ``build_rir.pyx``).

These functions are not used anywhere in pyroomacoustics anymore (the package
uses ``libroom.rir_builder``, ``libroom.fractional_delay`` and
``libroom.delay_sum``). They are kept for API compatibility and reproduce the
arithmetic of the Cython code (``cdef float`` variables are rounded to single
precision).
"""

import multiprocessing

import numpy as np

nthread = multiprocessing.cpu_count()


def fast_rir_builder(rir, time, alpha, visibility, fs, fdl, lut_gran=20):
    """
    Fast impulse response builder. This function takes the image source delays
    and amplitude and fills the impulse response. It uses a linear interpolation
    of the sinc function for speed.

    Parameters
    ----------
    rir: ndarray (double)
        The array to receive the impulse response. It should be filled with
        zero and of the correct size
    time: ndarray (double)
        The array of delays for the image sources
    alpha: ndarray (double)
        The array of attenuations for the image sources
    visibility: ndarray (int)
        Contains 1 if the image source is visible, 0 if not
    fs: int
        The sampling frequency
    fdl: int
        The length of the fractional delay filter (should be odd)
    lut_gran: int
        The number of point per unit in the sinc interpolation table
    """
    fs, fdl, lut_gran = int(fs), int(fdl), int(lut_gran)
    fdl2 = (fdl - 1) // 2
    n_times = time.shape[0]

    assert time.shape[0] == visibility.shape[0]
    assert time.shape[0] == alpha.shape[0]
    assert fdl % 2 == 1

    max_sample = np.ceil(fs * np.max(time)) + fdl2
    min_sample = np.floor(fs * np.min(time)) - fdl2
    assert min_sample >= 0
    assert max_sample < rir.shape[0]

    lut_size = (fdl + 1) * lut_gran + 1
    n = np.linspace(-fdl2 - 1, fdl2 + 1, lut_size)
    sinc_lut = np.sinc(n)
    hann = np.hanning(fdl)
    f32 = np.float32

    for i in range(n_times):
        if visibility[i] == 1:
            sample_frac = f32(fs * time[i])
            time_ip = int(np.floor(sample_frac))
            time_fp = f32(sample_frac - time_ip)
            x_off_frac = f32((1.0 - float(time_fp)) * lut_gran)
            lut_gran_off = int(np.floor(x_off_frac))
            x_off = f32(x_off_frac - lut_gran_off)
            lut_pos = lut_gran_off
            k = 0
            for f in range(-fdl2, fdl2 + 1):
                rir[time_ip + f] += (
                    alpha[i]
                    * hann[k]
                    * (
                        sinc_lut[lut_pos]
                        + float(x_off) * (sinc_lut[lut_pos + 1] - sinc_lut[lut_pos])
                    )
                )
                lut_pos += lut_gran
                k += 1


def fast_window_sinc_interpolator(
    vectorized_time_fp, window_length, vectorized_interpolated_sinc
):
    hann_wd = np.hanning(window_length)
    fdl2 = (window_length - 1) // 2
    lut_gran = 20
    lut_size = (window_length + 1) * lut_gran + 1
    n_ = np.linspace(-fdl2 - 1, fdl2 + 1, lut_size)
    sinc_lut = np.sinc(n_)

    for img_src, time_fp in enumerate(vectorized_time_fp):
        x_off_frac = (1 - time_fp) * lut_gran
        lut_gran_off = int(np.floor(x_off_frac))
        x_off = x_off_frac - lut_gran_off
        lut_pos = lut_gran_off
        for filter_sample in range(window_length):
            vectorized_interpolated_sinc[img_src, filter_sample] = hann_wd[
                filter_sample
            ] * (
                sinc_lut[lut_pos] + x_off * (sinc_lut[lut_pos + 1] - sinc_lut[lut_pos])
            )
            lut_pos += lut_gran

    return vectorized_interpolated_sinc


def fast_convolution_4(a, b, c, d, final_fir_IS_len):
    out = np.fft.fft(a) * np.fft.fft(b) * np.fft.fft(c) * np.fft.fft(d)
    return np.fft.ifft(out[:final_fir_IS_len])


def fast_convolution_3(a, b, c, final_fir_IS_len):
    out = np.fft.fft(a) * np.fft.fft(b) * np.fft.fft(c)
    return np.fft.ifft(out[:final_fir_IS_len])
