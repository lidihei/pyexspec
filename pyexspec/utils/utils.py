import numpy as np

def sigma_clip(yobs, y_predict, indselect, iiter, num_sigclip=3, min_select=100, verbose=False):
    '''
    yobs[array]
    y_predict [array]
    indselect [bool array]
    iiter [int] the ith iteration

    example:
    -----------------
    >>>import numpy as np
    >>>x = np.arange(100)
    >>>z = 10*x+5 + np.random.random(100)
    >>>indselect = np.ones_like(x, dtype=bool)
    >>>def getfunc(x, y, deg, rcond=None, full=False, w=None, cov=False, **argkeys):
    >>>    zz = np.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)
    >>>    pfunc1 = np.poly1d(zz)
    >>>    return pfunc1
    >>>iiter = 0
    >>>while True:
    >>>    pf1 = getfunc(x[indselect], z[indselect], deg)
    >>>    z_pred = pf1(x)
    >>>    indselect, iiter, break_bool = self.sigma_clip(z, z_pred, indselect, iiter,num_sigclip=num_sigclip, min_select=min_select, verbose=verbose)
    >>>    if break_bool: break
    >>>yfit = pf1(waveobs)
    '''
    y_res = yobs - y_predict
    sigma = np.std(y_res[indselect])
    indreject = np.abs(y_res[indselect]) > num_sigclip * sigma
    n_reject = np.sum(indreject)
    if n_reject == 0:
        # no lines to kick
        break_bool = True
    elif isinstance(min_select, int) and min_select >= 0 and np.sum(indselect) <= min_select:
        # selected lines reach the threshold
        break_bool = True
    else:
        # continue to reject lines
        indselect &= np.abs(y_res) < num_sigclip * sigma
        iiter += 1
        break_bool = False
    if verbose:
        print("  |-@grating_equation: iter-{} \t{} lines kicked, {} lines left, rms={:.5f} A".format(
            iiter, n_reject, np.sum(indselect), sigma))
    return indselect, iiter, break_bool


def estimate_gain(flat1,flat2, bias1, bias2, section = None):
    '''
    Calculate gain and readnoise by using flat and bais image
    ref. https://astro.uni-bonn.de/~sysstw/lfa_html/iraf/noao.nproto.findgain.html
    flat1 [2D array] image of flat 1
    flat2 [2D array] image of flat 2
    bias1 [2D array] image of flat 1
    bias2 [2D array] image of flat 2
    section [list] the section is used
           e.g. section = [row_start, row_end, col_start, col_end] 
                        =  [10, 200, 10, 50]
    returns:
    ----------------
    gain [float]
    readnoise [float]
    '''
    if section is None:
       row_start, col_start = 0, 0
       row_end, col_end = flat1.shape
    else:
       row_start, row_end, col_start, col_end = section
    flat1 = flat1[row_start:row_end, col_start:col_end]
    flat2 = flat2[row_start:row_end, col_start:col_end]
    bias1 = bias1[row_start:row_end, col_start:col_end]
    bias2 = bias2[row_start:row_end, col_start:col_end]
    flatdif = flat1 - flat2
    biasdif = bias1 - bias2
    gain = ((np.mean(flat1) + np.mean(flat2)) - (np.mean(bias1) + np.mean(bias2)))/(np.var(flatdif) - np.var(biasdif) )
    readnoise = gain * np.var(biasdif) / sqrt(2)
    return gain, readnoise

