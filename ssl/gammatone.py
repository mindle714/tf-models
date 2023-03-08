# copied from nnAudio/librosa_functions.py
import numpy as np

def fft2gammatonemx(
    sr=20000, n_fft=2048, n_bins=64, width=1.0, fmin=0.0, fmax=11025, maxlen=1024
):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """

    wts = np.zeros([n_bins, n_fft], dtype=np.float32)

    # after Slaney's MakeERBFilters
    EarQ = 9.26449
    minBW = 24.7
    order = 1

    nFr = np.array(range(n_bins)) + 1
    em = EarQ * minBW
    cfreqs = (fmax + em) * np.exp(
        nFr * (-np.log(fmax + em) + np.log(fmin + em)) / n_bins
    ) - em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(n_fft / 2 + 1)))
    ucirc = np.exp(1j * 2 * np.pi * ucircArray / n_fft)
    # justpoles = 0 :taking out the 'if' corresponding to this.

    ERB = width * np.power(
        np.power(cfreqs / EarQ, order) + np.power(minBW, order), 1 / order
    )
    B = 1.019 * 2 * np.pi * ERB
    r = np.exp(-B / sr)
    theta = 2 * np.pi * cfreqs / sr
    pole = r * np.exp(1j * theta)
    T = 1 / sr
    ebt = np.exp(B * T)
    cpt = 2 * cfreqs * np.pi * T
    ccpt = 2 * T * np.cos(cpt)
    scpt = 2 * T * np.sin(cpt)
    A11 = -np.divide(
        np.divide(ccpt, ebt) + np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2
    )
    A12 = -np.divide(
        np.divide(ccpt, ebt) - np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2
    )
    A13 = -np.divide(
        np.divide(ccpt, ebt) + np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2
    )
    A14 = -np.divide(
        np.divide(ccpt, ebt) - np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2
    )
    zros = -np.array([A11, A12, A13, A14]) / T
    wIdx = range(int(n_fft / 2 + 1))
    gain = np.abs(
        (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                + np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        * (
            -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T
            + 2
            * np.exp(-(B * T) + 2 * 1j * cfreqs * np.pi * T)
            * T
            * (
                np.cos(2 * cfreqs * np.pi * T)
                + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T)
            )
        )
        / (
            -2 / np.exp(2 * B * T)
            - 2 * np.exp(4 * 1j * cfreqs * np.pi * T)
            + 2 * (1 + np.exp(4 * 1j * cfreqs * np.pi * T)) / np.exp(B * T)
        )
        ** 4
    )
    # in MATLAB, there used to be 64 where here it says n_bins:
    wts[:, wIdx] = (
        ((T ** 4) / np.reshape(gain, (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[0], (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[1], (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[2], (n_bins, 1)))
        * np.abs(ucirc - np.reshape(zros[3], (n_bins, 1)))
        * (
            np.abs(
                np.power(
                    np.multiply(
                        np.reshape(pole, (n_bins, 1)) - ucirc,
                        np.conj(np.reshape(pole, (n_bins, 1))) - ucirc,
                    ),
                    -GTord,
                )
            )
        )
    )
    wts = wts[:, range(maxlen)]

    return wts, cfreqs


def gammatone(
    sr, n_fft, n_bins=64, fmin=20.0, fmax=None, htk=False, norm=1, dtype=np.float32
):
    """Create a Filterbank matrix to combine FFT bins into Gammatone bins
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_bins    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    G         : np.ndarray [shape=(n_bins, 1 + n_fft/2)]
        Gammatone transform matrix
    """

    if fmax is None:
        fmax = float(sr) / 2
    n_bins = int(n_bins)

    weights, _ = fft2gammatonemx(
        sr=sr,
        n_fft=n_fft,
        n_bins=n_bins,
        fmin=fmin,
        fmax=fmax,
        maxlen=int(n_fft // 2 + 1),
    )

    return (1 / n_fft) * weights
