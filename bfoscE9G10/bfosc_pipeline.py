#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% imports
import collections
import glob
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from astropy.io import fits
from twodspec import thar
from twodspec.aperture import Aperture
from twodspec.polynomial import Poly1DFitter
from twodspec.trace import trace_naive_max

plt.rcParams.update({"font.size": 20})
warnings.simplefilter("ignore")

# %% Set your data parameters
fp_fmt = "/Users/cham/projects/bfosc/20200917_bfosc/20200917-{:04d}.fit"
bias_num = [1, 2, 3, 4, 5, ]
flat_num = [11, 12, 13, 14, 15]
fear_num = [17, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 59, 61][:5]
star_num = [28, 30]

# %% Set your data parameters
# fp_fmt = "/Users/cham/projects/bfosc/20200916_bfosc/20200916-{:04d}.fit"
# bias_num = [1,2,3,4,5]
# flat_num = [6,7,8,9,10]
# fear_num = [11,12,27,33]
# star_num = [25,26]

# %% Set parameters of the pipeline
""" fear templates """
dp_fear_temp = "/Users/cham/projects/bfosc/bfosc/E9G10/template"
EXPTIME_THRESHOLD = 299
FEAR_VERBOSE = False

# %%
print(
    """
    |------------------------------------------------|
    | This pipeline is designed for the Xinglong     |
    | 2.16m BFOSC E9+G10 (R~10000) data.             |
    | It uses routines in the pipeline for the       |
    | SONG-China project -- the *songcn*.            |
    | To install *songcn*, type `pip install         |
    | songcn` in your terminal.                      |
    | All rights reserved.                           |
    | Current Version: v20201124                     |
    | Github: https://github.com/hypergravity/bfosc  |
    |                                       Bo Zhang |
    |                                     2020-11-24 |
    |                             bozhang@bnu.edu.cn |
    |------------------------------------------------|
    """)

# %%
print("""[1.0] get the working directory """)
dp = os.path.dirname(fp_fmt)
print("  |- setting working directory to {} ".format(dp))
os.chdir(dp)

print("""[1.1] generate file paths [bias, flat, fear, star] """)


def genfp(fp_fmt, num):
    if isinstance(num, collections.Iterable):
        return [fp_fmt.format(num_) for num_ in num]
    else:
        return fp_fmt.format(num)


fps_bias = genfp(fp_fmt, bias_num)
fps_flat = genfp(fp_fmt, flat_num)
fps_star = genfp(fp_fmt, star_num)
fps_fear = genfp(fp_fmt, fear_num)

print("""[1.2] check file existence """)


def check_existence(fps, infostr=""):
    print("[{}]".format(infostr))
    status = 0
    for fp in fps:
        if os.path.exists(fp):
            print("  |- {}".format(fp))
        else:
            status += 1
            print("  |- {}  DOESN'T EXIST!!!".format(fp))
    if status > 0:
        raise RuntimeError("Check file path PLEASE!")
    return


check_existence(fps_bias, "BIAS")
check_existence(fps_flat, "FLAT")
check_existence(fps_star, "STAR")
check_existence(fps_fear, "FEAR")

print("[1.3] check FEAR EXPTIME")
status_exptime = 0
for fp in fps_fear:
    _exptime = fits.getheader(fp)["EXPTIME"]
    if _exptime >= EXPTIME_THRESHOLD:
        print("  |- {}  {:.0f}s".format(fp, _exptime))
    else:
        status_exptime += 1
        print("·{}  {:.0f}s ?????".format(fp, _exptime))
if status_exptime > 0:
    raise RuntimeError("Check FEAR EXPTIME PLEASE!")

print("""[2.1] get ready to read & trim images """)


def trim(im):
    return im[850:-30, :]


def comb_bias(fps_bias):
    return np.rot90(trim(np.median(np.array([fits.getdata(fp) for fp in fps_bias]), axis=0)))


def comb_flat(fps_flat):
    return np.rot90(trim(np.median(np.array([fits.getdata(fp) for fp in fps_flat]), axis=0))) - master_bias


def read_star(fp_star):
    return np.rot90(trim(fits.getdata(fp_star))) - master_bias


print("""[2.2] process BIAS & FLAT """)
master_bias = comb_bias(fps_bias)
master_flat = comb_flat(fps_flat)
nrow, ncol = master_flat.shape
print("  |- the shape of trimmed images are ", master_flat.shape)

print("""[3.1] trace apertures """)
# trace using local maximum
ap_trace = trace_naive_max(master_flat, sigma=7, maxdev=50, irow_start=1300)
print("  |- initial aperture shape", ap_trace.shape)
# mask invalid aperture sections
ap_diff = np.abs(np.diff(ap_trace, axis=1))
for i, j in zip(*np.where(ap_diff > 20)):
    # print(i, j)
    ap_trace[i, :j + 1] = np.nan
# interpolate with poly2
ap_coord = np.arange(nrow)
ap_illusion = np.zeros_like(ap_trace)
for i in range(ap_trace.shape[0]):
    ind_finite = np.isfinite(ap_trace[i])
    ap_illusion[i] = np.polyval(np.polyfit(ap_coord[ind_finite], ap_trace[i, ind_finite], 2), ap_coord)
# exclude invalid apertures
ind_valid_ap = np.nanmax(np.abs(ap_illusion - ap_trace), axis=1) < 20
ap_illusion = ap_illusion[ind_valid_ap]
# exclude duplicated apertures using intersects
intersect = ap_illusion[:, 1200]
ind_valid_ap = np.ones_like(intersect, bool)
for i in range(len(intersect)):
    if np.any(np.abs(intersect[i] - intersect[i + 1:]) < 5):
        ind_valid_ap[i] = False
ap_interp = ap_illusion[ind_valid_ap]
ap_interp = ap_interp[ap_interp[:, -1] < 1070]
try:
    assert ap_interp.shape[0] == 12
except:
    print("invalid number of apertures, please check images!")
    print("ap_interp", ap_interp.shape)
    print("Note that the FEAT template has 11 orders!")
    raise RuntimeError()
ap_interp = ap_interp[1:]
print("  |- final aperture shape", ap_interp.shape)

print("""[3.2] remove background of flat, compute blaze & sensitivity """)
# initiate Aperture instance
ap = Aperture(ap_center=ap_interp, ap_width=22)
ap.get_image_info(master_flat)
ap.polyfit(2)
# compute blaze & sensitivity
flat_bg = ap.background(master_flat, q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(21, 21))
master_flat -= flat_bg
blaze, sensitivity = ap.make_normflat(master_flat, )
# figure(); imshow(sensitivity)
# figure(); plot(blaze.T)


def debug_ap():
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].imshow(np.log10(master_flat + 100), cmap=plt.cm.plasma)
    axs[0].plot(ap_interp[:, :].T, np.arange(nrow), 'k')
    axs[0].set_title("FLAT")
    axs[1].imshow(flat_bg, cmap=plt.cm.plasma)
    axs[1].plot(ap_interp[:, :].T, np.arange(nrow), 'k')
    axs[0].set_title("FLAT background")
    return fig


print("""[4.1] extracting star1d (~5s/star) """)
# loop over stars
n_star = len(fps_star)
for i_star, fp in enumerate(fps_star):
    print("  |- ({}/{}) processing STAR ... ".format(i_star, n_star), end="")
    fp_out = "{}/star-{}.dump".format(os.path.dirname(fp), os.path.basename(fp))
    star = read_star(fp)
    star -= ap.background(star, q=(10, 10), npix_inter=5, sigma=(20, 20), kernel_size=(21, 21))
    star /= sensitivity
    star1d = ap.extract_all(star, n_jobs=1, verbose=False)
    print("writing to {}".format(fp_out))
    star1d["blaze"] = blaze
    star1d["JD"] = fits.getheader(fp)["JD"]
    star1d["EXPTIME"] = fits.getheader(fp)["EXPTIME"]
    joblib.dump(star1d, fp_out)
    # figure(); imshow(star)
    # figure(); plot(star1d["spec_extr"].T,c="k"); plot(star1d["spec_sum"].T)

print("[5.1] load FEAR template & FEAR line list")
# import  glob
# from astropy.table import Table
# fps_temp = glob.glob(fp_fear_temp+"/*.txt")
# fps_temp.sort()
# wave_temp = np.flipud(np.array([np.loadtxt(fp)[:,0] for fp in fps_temp]))[0:-1]
# fear_temp = np.flipud(np.array([np.loadtxt(fp)[:,1] for fp in fps_temp]))[0:-1]
# fear_list = Table.read(fp_fear_list,format="ascii.no_header")["col1"].data
# joblib.dump(fear_list, "fear_list.dump")
# joblib.dump((wave_temp, fear_temp), "fear_temp.dump")
# figure(); plot(wave_temp.T, fear_temp.T)
""" use dump files """
fear_list = joblib.load("{}/fear_list.dump".format(dp_fear_temp))
wave_temp, fear_temp = joblib.load("{}/fear_temp.dump".format(dp_fear_temp))

print("[5.2] processing FEAR ")


def proc_fear(fp, nsigma=2.5, verbose=False):
    """ read fear """
    fear = read_star(fp)
    fear /= sensitivity
    # unnecessary to remove background
    # fear -= apbackground(fear, ap_interp, q=(10, 10), npix_inter=5,sigma=(20, 20),kernel_size=(21,21))
    # extract 1d fear
    fear1d = ap.extract_all(fear, n_jobs=1)["spec_sum"]
    # remove baseline
    # fear1d -= np.median(fear1d)

    """ corr2d to get initial estimate of wavelength """
    wave_init = thar.corr_thar(wave_temp, fear_temp, fear1d, maxshift=50)
    # figure(figsize=(15, 5));
    # plot(wave_temp[:].T, -1000-fear1d.T, c="m", lw=2,)
    # plot(wave_temp.T, fear_temp.T, c="darkcyan", lw=2)
    # text(6000, 10000, "REN")
    # text(6000, -10000, "LUO")

    """ find thar lines """
    tlines = thar.find_lines(wave_init, fear1d, fear_list, npix_chunk=20, ccf_kernel_width=1.5)
    ind_good = np.isfinite(tlines["line_x_ccf"]) & (np.abs(tlines["line_x_ccf"] - tlines["line_x_init"]) < 10) & (
            (tlines["line_peakflux"] - tlines["line_base"]) > 100) & (
                       np.abs(tlines["line_wave_init_ccf"] - tlines["line"]) < 3)
    tlines.add_column(table.Column(ind_good, "ind_good"))
    # tlines.show_in_browser()

    """ clean each order """
    def clean(pw=1, deg=2, threshold=0.1, min_select=20):
        order = tlines["order"].data
        ind_good = tlines["ind_good"].data
        linex = tlines["line_x_ccf"].data
        z = tlines["line"].data

        u_order = np.unique(order)
        for _u_order in u_order:
            ind = (order == _u_order) & ind_good
            if np.sum(ind) > 10:
                # in case some orders have only a few lines
                p1f = Poly1DFitter(linex[ind], z[ind], deg=deg, pw=pw)
                res = z[ind] - p1f.predict(linex[ind])
                ind_good[ind] &= np.abs(res) < threshold
        tlines["ind_good"] = ind_good
        return

    print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
    clean(pw=1, deg=2, threshold=0.8, min_select=20)
    clean(pw=1, deg=2, threshold=0.4, min_select=20)
    clean(pw=1, deg=2, threshold=0.2, min_select=20)
    print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
    tlines = tlines[tlines["ind_good"]]

    """ fitting grating equation """
    x = tlines["line_x_ccf"]  # line_x_ccf/line_x_gf
    y = tlines["order"]
    z = tlines["line"]
    pf1, pf2, indselect = thar.grating_equation(
        x, y, z, deg=(3, 7), nsigma=nsigma, min_select=210, verbose=FEAR_VERBOSE)
    tlines.add_column(table.Column(indselect, "indselect"))
    if 0.01 < pf2.rms < 0.1:
        # reasonable
        nlines = np.sum(indselect)
        # mpflux
        mpflux = np.median(tlines["line_peakflux"][tlines["indselect"]])
        # rms
        rms = np.std((pf2.predict(x, y) - z)[indselect])
        print("  |- nlines={}  rms={:.4f}A  mpflux={:.1f}".format(nlines, rms, mpflux))
        # predict wavelength solution
        nx, norder = fear1d.shape
        mx, morder = np.meshgrid(np.arange(norder), np.arange(nx))
        wave_solu = pf2.predict(mx, morder)  # polynomial fitter
        # result
        calibration_dict = collections.OrderedDict(
            fp=fp,
            jd=fits.getheader(fp)["JD"],
            exptime=fits.getheader(fp)["EXPTIME"],
            wave_init=wave_init,
            wave_solu=wave_solu,
            tlines=tlines,
            nlines=nlines,
            rms=rms,
            pf1=pf1,
            pf2=pf2,
            mpflux=mpflux,
            # fear=fear,
            fear1d=fear1d
        )
        return calibration_dict
    else:
        print("!!! result is not acceptable, this FEAR is skipped")
        return None


""" loop over fear """
n_fear = len(fps_fear)
for i_fear, fp in enumerate(fps_fear):
    print("  |- ({}/{}) processing FEAR {} ... ".format(i_fear, n_fear, fp))
    fp_out = "{}/fear-{}.dump".format(os.path.dirname(fp), os.path.basename(fp))
    res = proc_fear(fp, 2.5, True)
    if res is not None:
        print("  |- writing to {}".format(fp_out))
        joblib.dump(res, fp_out)

print("""[6.0] make stats for the FEAR solutions """)
fps_fear_res = glob.glob("{}/fear-*".format(dp))
fps_fear_res.sort()
tfear = table.Table([joblib.load(_) for _ in fps_fear_res])

# %%
""" a statistic figure of reduced fear """
fig = plt.figure(figsize=(9, 7))
ax = plt.gca()
ax.plot(tfear['jd'], tfear["rms"] / 4500 * 3e5, 's-', ms=10, label="RMS")
ax.set_xlabel("JD")
ax.set_ylabel("RMS [km s$^{-1}$]")
ax.set_title("The precision of FEAR calibration @4500A")
ax.legend(loc="upper left")

axt = ax.twinx()
axt.plot(tfear['jd'], tfear["nlines"], 'o-', ms=10, color="gray", label="nlines");
axt.set_ylabel("N(Lines)")
axt.legend(loc="upper right")

fig.tight_layout()
fig.savefig("{}/fear_stats.pdf".format(dp))

print("""[7.0] Finished!""")
print("")
print("")
print("")

# %%
# #%%
# figure()
# plot(wave_init.T, fear1d.T)
# vlines(fear_list, 0, 1000,color="gray")
# vlines(tlines["line"], 0, 1000)

# #%%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z,'o')

# #%%

# figure();plot(tlines["line"],tlines["line_wave_init_ccf"]-tlines["line"],'o')
# #%%
# figure();
# plot(wave_solu.T, fear1d.T/10)
# plot(wave_temp.T, -fear_temp.T)

# #%%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z,'mo')
# ax.plot_surface(mx, morder, wave_solu,alpha=0.5,cmap=cm.jet)
# ax.set_ylabel("Order")
# ax.set_xlabel("#pixel @ dispersion axis")
# ax.set_zlabel("wavelength /A")

# #%%
# figure()
# plot(z, pf2.predict(x, y)-z,'o',mec='k',label="all")
# plot(z[indselect], (pf2.predict(x, y)-z)[indselect],'o',mec='k',label="used")

# # plot(tlines["line"][indselect], pf2.predict(x[indselect], y[indselect])-tlines["line"][indselect],'o',mec='k',label="reserved")
# legend()
# xlabel("$\\lambda_{true}$ / A")
# ylabel("$\\lambda_{pred}-\\lambda_{true}$ / A")
