"""
Calculate R(Q) from raw USAXS data.
"""

import logging
import math

import numpy

logger = logging.getLogger(__name__)
logger.info(__file__)


RMAX_CUTOFF = 0.4  # when calculating the center, look at data above CUTOFF*R_max
ZINGER_THRESHOLD = 2
DEFAULT_SCALER_PULSES_PER_SECOND = 1e-7  # FIXME: why is this < 1?


def amplifier_corrections(signal, seconds, dark, gain):
    """
    correct for amplifier dark current and gain
    :math:`v = (s - t*d) / g`
    """
    # v = (signal - seconds*dark) / gain
    v = numpy.array(signal, dtype=float)
    if dark is not None:  # compatibility with older USAXS data
        v = numpy.ma.masked_less_equal(v - seconds * dark, 0)
    if gain is not None:  # compatibility with older USAXS data
        gain = numpy.ma.masked_less_equal(gain, 0)
        v /= gain
    return v


def calc_R_Q(
    wavelength,
    ar,
    seconds,
    pd,
    pd_bkg,
    pd_gain,
    I0,
    I0_bkg=None,
    I0_gain=None,
    ar_center=None,
    V_f_gain=None,
):
    """
    Calculate 1-D :math:`R(Q)` from raw USAXS data.

    :param float wavelength: :math:`lambda`, (:math:`\A`)
    :param float ar_center: center of rocking curve along AR axis
    :param numpy.ndarray([float]) ar: array of crystal analyzer angles
    :param numpy.ndarray([float]) seconds: array of counting time for each point
    :param numpy.ndarray([float]) pd: array of photodiode counts
    :param numpy.ndarray([float]) pd_bkg: array of photodiode amplifier backgrounds
    :param numpy.ndarray([float]) pd_gain: array of photodiode amplifier gains
    :param numpy.ndarray([float]) I0: array of incident monitor counts
    :param numpy.ndarray([float]) I0_bkg: array of I0 backgrounds
    :param numpy.ndarray([float]) I0_amp_gain: array of I0 amplifier gains
    :param numpy.ndarray([float]) V_f_gain: array of voltage-frequency converter gains
    :returns dictionary: Q, R
    :param numpy.ndarray([float]) qVec: :math:`Q`
    :param numpy.ndarray([float]) rVec: :math:`R = I/I_o`
    """
    r = amplifier_corrections(pd, seconds, pd_bkg, pd_gain)
    r0 = amplifier_corrections(I0, seconds, I0_bkg, I0_gain)

    rVec = r / r0
    if V_f_gain is not None:  # but why?
        rVec /= V_f_gain
    rVec = numpy.ma.masked_less_equal(rVec, 0)

    ar_r_peak = ar[numpy.argmax(rVec)]  # ar value at peak R
    rMax = rVec.max()
    if ar_center is None:  # compute ar_center from rVec and ar
        ar_center = centroid(ar, rVec)  # centroid of central peak

    d2r = math.pi / 180
    qVec = (4 * math.pi / wavelength) * numpy.sin(d2r * (ar_center - ar) / 2)

    # trim off masked points
    r0 = remove_masked_data(r0, rVec.mask)
    r = remove_masked_data(r, rVec.mask)
    ar = remove_masked_data(ar, rVec.mask)
    qVec = remove_masked_data(qVec, rVec.mask)
    rVec = remove_masked_data(rVec, rVec.mask)

    result = dict(
        Q=qVec,
        R=rVec,
        ar=ar,
        r=r,
        r0=r0,
        ar_0=ar_center,
        ar_r_peak=ar_r_peak,
        r_peak=rMax,
    )
    return result


def centroid(x, y):
    """Compute centroid of y(x)."""
    import scipy.integrate

    def zinger_test(u, v):
        m = max(v)
        p = numpy.where(v == m)[0][0]
        top = (v[p - 1] + v[p] + v[p + 1]) / 3
        bot = (v[p - 1] + v[p + 1]) / 2
        v_test = top / bot
        logger.debug("zinger test: %f", v_test)
        return v_test

    a = remove_masked_data(x, y.mask)
    b = remove_masked_data(y, y.mask)

    while zinger_test(a, b) > ZINGER_THRESHOLD:
        R_max = max(b)
        peak_index = numpy.where(b == R_max)[0][0]
        # delete or mask x[peak_index], and y[peak_index]
        logger.debug("removing zinger at ar = %f", a[peak_index])
        a = numpy.delete(a, peak_index)
        b = numpy.delete(b, peak_index)

    # gather the data nearest the peak (above the CUTOFF)
    R_max = max(b)
    cutoff = R_max * RMAX_CUTOFF
    peak_index = numpy.where(b == R_max)[0][0]
    n = len(a)

    # walk down each side from the peak
    pLo = peak_index
    while pLo >= 0 and b[pLo] > cutoff:
        pLo -= 1

    pHi = peak_index + 1
    while pHi < n and b[pHi] > cutoff:
        pHi += 1

    # enforce boundaries
    pLo = max(0, pLo + 1)  # the lowest ar above the cutoff
    pHi = min(n - 1, pHi)  # the highest ar (+1) above the cutoff

    if pHi - pLo == 0:
        emsg = "not enough data to find peak center - not expected"
        logger.debug(emsg)
        raise KeyError(emsg)
    elif pHi - pLo == 1:
        # trivial answer
        emsg = "peak is 1 point, picking peak position as center"
        logger.debug(emsg)
        return x[peak_index]

    a = a[pLo:pHi]
    b = b[pLo:pHi]

    weight = b * b
    top = scipy.integrate.simps(a * weight, a)
    bottom = scipy.integrate.simps(weight, a)
    center = top / bottom

    emsg = "computed peak center: " + str(center)
    logger.debug(emsg)
    return center


def reduce_uascan(root):
    "1-D data reduction, from livedata."

    entry = root["/entry"]
    baseline = entry["instrument/bluesky/streams/baseline"]
    primary = entry["instrument/bluesky/streams/primary"]

    # Must copy from h5py into local data to keep once h5py file is closed.
    wavelength = entry["instrument/monochromator/wavelength"][()]
    ar = primary["a_stage_r/value"][()]
    # sds is SPEC Data File Scan: pps = sds.MD.get("scaler_pulses_per_second", 1e-7)
    pps = DEFAULT_SCALER_PULSES_PER_SECOND
    seconds = pps * primary["seconds/value"][()]  # convert from counts
    pd = primary["PD_USAXS/value"][()]
    I0 = primary["I0_USAXS/value"][()]
    I0_amplifier_gain = primary["I0_autorange_controls_gain/value"][()]
    ar_center = baseline["terms_USAXS_center_AR/value_start"][()]

    pd_gain = primary["upd_autorange_controls_gain/value"][()]

    pd_range = primary["upd_autorange_controls_reqrange/value"][()]
    bkg = []
    for ch in range(5):
        addr = "upd_autorange_controls_ranges_gain%d_background" % ch
        bkg.append(baseline[addr + "/value_start"][()])
    pd_dark = [bkg[i] for i in pd_range]

    # compute the R(Q) profile
    usaxs = calc_R_Q(
        wavelength,
        ar,
        seconds,
        pd,
        pd_dark,
        pd_gain,
        I0,
        I0_gain=I0_amplifier_gain,
        ar_center=ar_center,
    )

    return usaxs


def remove_masked_data(data, mask):
    """Remove all masked data, convenience routine."""
    arr = numpy.ma.masked_array(data=data, mask=mask)
    return arr.compressed()