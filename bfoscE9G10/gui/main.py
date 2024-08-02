import sys, os, glob
from PyQt5 import QtCore, QtGui, QtWidgets
from bfosc import Ui_MainWindow
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from astropy import table
from astropy.io import fits
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
import joblib

matplotlib.use('Qt5Agg')
matplotlib.rcParams["font.size"] = 5


class UiBfosc(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(UiBfosc, self).__init__(parent)
        # data
        self._wd = ""
        self.datatable = None
        self.pos = []
        self.pos_temp = [0, 0]
        self.master_bias = None
        self.master_flat = None
        self.trace_handle = []
        self.ap_trace = np.zeros((0, 2048), dtype=int)
        self._fear = None

        # UI
        self.setupUi(self)
        self.add_canvas()
        self.initUi()

        # debug
        self.assumption()

    def add_canvas(self):
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(710, 20, 700, 500))
        self.widget2.setObjectName("widget")
        self.verticalLayout2 = QtWidgets.QVBoxLayout(self.widget2)
        self.verticalLayout2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout2.setObjectName("verticalLayout")

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        # self.canvas.setGeometry(QtCore.QRect(350, 110, 371, 311))
        # self.canvas.setObjectName("canvas")

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        # self.toolbar.setGeometry(QtCore.QRect(370, 70, 371, 41))

        # Just some button connected to `plot` method
        # self.pushButton_showimage.clicked.connect(self.plot)

        # set the layout
        # layout = QtWidgets.QVBoxLayout()
        self.verticalLayout2.addWidget(self.toolbar)
        self.verticalLayout2.addWidget(self.canvas)
        # layout.addWidget(self.button)
        # self.setLayout(layout)

    def assumption(self):
        test_dir = "/Users/lijiao/Documents/works/Feige89/data/216cm/20240314_bfosc_liuzhicun/E9G10"
        self._wd = test_dir
        self.lineEdit_wd.setText(test_dir)
        #self.ap = joblib.load("/home/sdb216/sdOBdata/216BFOSC/sdOB_216/20220522_bfosc/ap.dump")
        self._fear = joblib.load("../template/fear_model_order2_12.dump")
        #self.ap = joblib.load(os.path.join(test_dir, "ap.dump"))
        #self.ap = joblib.load(f"/home/lcq/media/backup/216BFOSC/sdOB_216/20210603_bfosc/ap.dump")
        #self.ap_trace = self.ap.ap_center_interp[:,::-1]

    def initUi(self):
        self.toolButton.clicked.connect(self._select_wd)
        self.toolButton_load_fear.clicked.connect(self._select_fear)
        self.lineEdit_wd.textChanged.connect(self._get_file_list)
        self.tableWidget_files.itemSelectionChanged.connect(self._show_img)
        self.pushButton_update_table.clicked.connect(self._update_datatable)
        self.pushButton_proc_bias.clicked.connect(self._proc_bias)
        self.pushButton_proc_flat.clicked.connect(self._proc_flat)
        self.pushButton_clear_aperture.clicked.connect(self._clear_aperture)
        self.pushButton_add_aperture.clicked.connect(self._add_aperture)
        self.pushButton_del_aperture.clicked.connect(self._del_aperture)
        self.pushButton_save_aperture.clicked.connect(self._save_aperture)
        self.pushButton_proc_all.clicked.connect(self._proc_all)
        # self.listWidget_files.currentItemChanged.connect(self._show_img)

    def _select_wd(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        self.lineEdit_wd.setText(directory)
        self._wd = directory
        print("WD set to ", self._wd)

    def _select_fear(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Fe/Ar", "dump files (*.dump)")
        print(fileName)
        print('abc')
        self.lineEdit_fear.setText(fileName)
        self._fear = joblib.load(fileName)
        print("FEAR loaded!")

    def _set_wd(self):
        self._wd = self.lineEdit_wd.text()

    def _make_datatable(self):
        # get file list
        fps_full = glob.glob(self.lineEdit_wd.text() + "/*.fit")
        fps_full.sort()
        self.fps_full = fps_full
        fps = [os.path.basename(_) for _ in fps_full]
        self.nfp = len(fps)

        imgtype = np.asarray([fits.getheader(fp)["IMAGETYP"] for fp in fps_full])
        exptime = np.asarray([fits.getheader(fp)["EXPTIME"] for fp in fps_full])
        types = np.zeros_like(imgtype)
        self.type_dict = OrderedDict(drop=0, bias=1, flat=2, fear=3, star=4)
        self.type_list = list(self.type_dict.keys())
        self.color_list = [[255, 255, 255],
                           [211, 211, 211],
                           [255, 182, 193],
                           [255, 228, 181],
                           [173, 216, 230], ]
        # initial guess for types:
        for i in range(self.nfp):
            if "bias" in imgtype[i].lower() or "bias" in fps_full[i].lower():
                types[i] = "bias"
            elif "flat" in imgtype[i].lower() or "flat" in fps_full[i].lower():
                types[i] = "flat"
            elif "light" in imgtype[i].lower() and (exptime[i] == 300 or "fear" in fps_full[i].lower()):
                types[i] = "fear"
            elif "light" in imgtype[i].lower() and (exptime[i] != 300 or "target" in fps_full[i].lower()):
                types[i] = "star"
            else:
                types[i] = "drop"

        self.datatable = table.Table(
            data=[fps, imgtype, exptime, types],
            names=["filename", "imagetype", "exptime", "type"])
        # print(self.datatable["type"])

    def _update_datatable(self):
        # print(self.datatable["type"])
        self.datatable["type"] = [self.type_list[self.tableWidget_files.cellWidget(irow, 3).currentIndex()] for irow in range(self.nfp)]
        self._refresh_datatable()
        self.datatable.write(self._wd+"/catalog.fits", overwrite=True)

    def _get_file_list(self):
        self._make_datatable()
        self._refresh_datatable()

    def _refresh_datatable(self):
        if self.datatable is None:
            return
        # change to Table Widget
        self.tableWidget_files.clear()
        self.tableWidget_files.verticalHeader().setVisible(False)
        self.tableWidget_files.setRowCount(self.nfp)
        self.tableWidget_files.setColumnCount(4)
        self.tableWidget_files.setHorizontalHeaderLabels(self.datatable.colnames)
        for irow in range(self.nfp):
            self.tableWidget_files.setItem(irow, 0, QtWidgets.QTableWidgetItem(str(self.datatable["filename"][irow])))
            self.tableWidget_files.setItem(irow, 1, QtWidgets.QTableWidgetItem(str(self.datatable["imagetype"][irow])))
            self.tableWidget_files.setItem(irow, 2, QtWidgets.QTableWidgetItem("{:.0f}".format(self.datatable["exptime"][irow])))

            comboBoxItem = QtWidgets.QComboBox()
            comboBoxItem.addItems(self.type_dict.keys())
            # print(self.type_dict[self.datatable["type"][irow]])
            this_type_index = self.type_dict[self.datatable["type"][irow]]
            comboBoxItem.setCurrentIndex(this_type_index)
            self.tableWidget_files.setCellWidget(irow, 3, comboBoxItem)

            for icol in range(3):
                self.tableWidget_files.item(irow, icol).setBackground(
                    QtGui.QBrush(QtGui.QColor(*self.color_list[this_type_index])))

        self.tableWidget_files.resizeColumnsToContents()
        self.tableWidget_files.resizeRowsToContents()

    def _show_img(self):
        ind_elected = self.tableWidget_files.currentRow()
        fp_selected = self.fps_full[ind_elected]
        print("Show file {}: {}".format(ind_elected, fp_selected))
        # try to draw it
        try:
            img = fits.getdata(fp_selected)
        except IsADirectoryError:
            print("Not sure about what you are doing ...")
            return
        self._draw_img(img)

    def _draw_img(self, img):
        # draw
        self.figure.clear()
        self.ax = self.figure.add_axes([0, 0, 1, 1])
        self.ax.imshow(img, cmap=plt.cm.jet, origin="lower", vmin=np.percentile(img, 5), vmax=np.percentile(img, 90),
                       aspect="auto")
        self.pos_handle, = self.ax.plot([], [], "+", ms=10, color="tab:cyan", mew=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_ylim(850, 2048)
        self.ax.set_xlim(0, 2048)
        self.ax.plot()
        # refresh canvas
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def onclick(self, event):
        # capture cursor position ===============
        # ref: https://matplotlib.org/stable/users/event_handling.html
        self.pos_temp = event.xdata, event.ydata
        self._draw_updated_pos()

    def _draw_updated_pos(self):
        self.pos_handle.set_data(*[np.array([_]) for _ in self.pos_temp])
        self.canvas.draw()
        # and trace this aperture
        print(self.pos_temp)

    def _trace_one_aperture(self):
        print("trace one aperture")
        pass

    def _gather_files(self, filetype="bias"):
        fps_bias = []
        for i in range(self.nfp):
            if self.datatable["type"][i] == filetype:
                fps_bias.append(self.fps_full[i])
                print("appending {}: {}".format(filetype, self.fps_full[i]))
        return fps_bias

    def _proc_bias(self):
        if self.datatable is None:
            pass
        fps_bias = []
        for i in range(self.nfp):
            if self.datatable["type"][i]=="bias":
                fps_bias.append(self.fps_full[i])
                print("appending BIAS: {}".format(self.fps_full[i]))
        self.master_bias = np.median(np.array([fits.getdata(fp) for fp in fps_bias]), axis=0)
        self._draw_img(self.master_bias)
        print(">>> BIAS processed!")

    def _proc_flat(self):
        if self.datatable is None:
            pass
        fps_flat = []
        for i in range(self.nfp):
            if self.datatable["type"][i] == "flat":
                fps_flat.append(self.fps_full[i])
                print("appending FLAT: {}".format(self.fps_full[i]))
        self.master_flat = np.median(np.array([fits.getdata(fp) for fp in fps_flat]), axis=0)
        self.master_flat -= self.master_bias
        self._draw_img(gaussian_filter(self.master_flat, sigma=2))
        # import joblib
        # joblib.dump(self.master_flat, "/Users/cham/projects/bfosc/20200915_bfosc/master_flat.dump")
        print(">>> FLAT processed!")

    def _clear_aperture(self):
        # from twodspec.trace import trace_naive_max
        # self.ap_trace = trace_naive_max(self.master_flat.T, sigma=7, maxdev=10, irow_start=1300)
        self.ap_trace = np.zeros((0, 2048), dtype=int)
        self._update_nap()
        # print(self.ap_trace.shape)
        self._draw_aperture()

    def _draw_aperture(self):
        if len(self.ax.lines) > 1:
            for line in self.ax.get_lines()[1:]:  # ax.lines:
                line.remove()
        for _trace in self.ap_trace:
            ind_plot = _trace > 0
            self.ax.plot(np.arange(2048)[ind_plot], _trace[ind_plot], "w-", lw=1)
        self.canvas.draw()

    def _add_aperture(self):
        try:
            from twodspec.trace2 import trace_local_max
            _trace = trace_local_max(
                gaussian_filter(self.master_flat, sigma=2),
                *np.asarray(self.pos_temp[::-1], dtype=int), maxdev=10, fov=20, ntol=5)
            if np.sum(_trace>0)>100:
                self.ap_trace = np.vstack((self.ap_trace, _trace.reshape(1, -1)))
                self._draw_aperture()
                self._update_nap()
        except Exception as _e:
            print("Error occurred, aperture not added!")
            print(_e)

    def _del_aperture(self):
        if self.ap_trace.shape[0] == 0:
            pass
        dx = np.arange(2048) - self.pos_temp[0]
        dy = self.ap_trace - self.pos_temp[1]
        d = np.abs(dx ** 2 + dy ** 2)
        ind_min = np.argmin(d)
        ind_min_ap, ind_min_pix = np.unravel_index(ind_min, self.ap_trace.shape)
        self.ap_trace = self.ap_trace[np.arange(self.ap_trace.shape[0])!=ind_min_ap]
        self._update_nap()
        self._draw_aperture()

    def _save_aperture(self):
        from twodspec.aperture import Aperture
        # print(self.ap_trace[:,0])
        self.ap_trace = self.ap_trace[sort_apertures(self.ap_trace)]
        # fit
        self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=15)
        self.ap.get_image_info(self.master_flat)
        self.ap.polyfit(2)
        # replace old traces
        self.ap_trace = self.ap.ap_center_interp[:, ::-1]
        # fit again
        self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=15)
        self.ap.get_image_info(self.master_flat)
        self.ap.polyfit(2)
        self._draw_aperture()
        import joblib
        joblib.dump(self.ap, self._wd+"/ap.dump")
        print("Aperture saved to ", self._wd+"/ap.dump")

    def _update_nap(self):
        self.lineEdit_nap.setText("N(ap)={}".format(self.ap_trace.shape[0]))
    def _proc_all(self):
        if self._fear is None:
            print("FEAR not loaded!")
        nrow, ncol = self.master_flat.shape

        # compute blaze & sensitivity
        flat_bg = self.ap.background(np.rot90(self.master_flat), q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(21, 21))
        self.blaze, self.sensitivity = self.ap.make_normflat(np.rot90(self.master_flat)-flat_bg, )

        print("""[4.1] extracting star1d (~5s/star) """)
        # loop over stars
        fps_star = self._gather_files("star")
        n_star = len(fps_star)
        for i_star, fp in enumerate(fps_star):
            print("  |- ({}/{}) processing STAR ... ".format(i_star, n_star), end="")
            fp_out = "{}/star-{}.dump".format(os.path.dirname(fp), os.path.basename(fp))
            star = self.read_star(fp)
            star -= self.ap.background(star, q=(10, 10), npix_inter=5, sigma=(20, 20), kernel_size=(21, 21))
            star /= self.sensitivity
            star1d = self.ap.extract_all(star, n_jobs=1, verbose=False)
            print("writing to {}".format(fp_out))
            star1d["blaze"] = self.blaze
            star1d["JD"] = fits.getheader(fp)["JD"]
            star1d["EXPTIME"] = fits.getheader(fp)["EXPTIME"]
            joblib.dump(star1d, fp_out)

        print("[5.1] load FEAR template & FEAR line list")
        """ loop over fear """
        fps_fear = self._gather_files("fear")
        n_fear = len(fps_fear)
        for i_fear, fp in enumerate(fps_fear):
            print("  |- ({}/{}) processing FEAR {} ... ".format(i_fear, n_fear, fp))
            fp_out = "{}/fear-{}.dump".format(os.path.dirname(fp), os.path.basename(fp))
            res = self._proc_fear(fp, 2.5, True)
            if res is not None:
                print("  |- writing to {}".format(fp_out))
                joblib.dump(res, fp_out)

        print("""[6.0] make stats for the ARCS solutions """)
        fps_fear_res = glob.glob("{}/fear-*".format(self._wd))
        fps_fear_res.sort()
        tfear = table.Table([joblib.load(_) for _ in fps_fear_res])

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
        fig.savefig("{}/fear_stats.pdf".format(self._wd))
        pass

    def _proc_fear(self, fp, nsigma=2.5, verbose=False, deg=(4,7)):
        """ read fear """
        fear = self.read_star(fp)
        fear /= self.sensitivity
        # unnecessary to remove background
        # fear -= apbackground(fear, ap_interp, q=(10, 10), npix_inter=5,sigma=(20, 20),kernel_size=(21,21))
        # extract 1d fear
        fear1d = self.ap.extract_all(fear, n_jobs=1)["spec_sum"]
        # remove baseline
        # fear1d -= np.median(fear1d)

        """ corr2d to get initial estimate of wavelength """
        from twodspec import thar
        wave_init = thar.corr_thar(self._fear["wave"], self._fear["flux"], fear1d, maxshift=50)
        # figure(figsize=(15, 5));
        # plot(wave_temp[:].T, -1000-fear1d.T, c="m", lw=2,)
        # plot(wave_temp.T, fear_temp.T, c="darkcyan", lw=2)
        # text(6000, 10000, "REN")
        # text(6000, -10000, "LUO")

        """ find thar lines """
        tlines = thar.find_lines(wave_init, fear1d, self._fear["linelist"], npix_chunk=20, ccf_kernel_width=1.5)
        ind_good = np.isfinite(tlines["line_x_ccf"]) & (np.abs(tlines["line_x_ccf"] - tlines["line_x_init"]) < 10) & (
                (tlines["line_peakflux"] - tlines["line_base"]) > 100) & (
                           np.abs(tlines["line_wave_init_ccf"] - tlines["line"]) < 3)
        tlines.add_column(table.Column(ind_good, "ind_good"))
        # tlines.show_in_browser()

        """ clean each order """
        from twodspec.polynomial import Poly1DFitter

        def clean(pw=1, deg=2, threshold=0.1, min_select=10):
            order = tlines["order"].data
            ind_good = tlines["ind_good"].data
            linex = tlines["line_x_ccf"].data
            z = tlines["line"].data

            u_order = np.unique(order)
            for _u_order in u_order:
                ind = (order == _u_order) & ind_good
                if np.sum(ind) > min_select:
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
            x, y, z, deg=deg, nsigma=nsigma, min_select=210, verbose=10)
        tlines.add_column(table.Column(indselect, "indselect"))
        print(f'###----------------\npf2.rms = {pf2.rms}\n###------------------')
        if 0.01 < pf2.rms < 1:
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
            calibration_dict = OrderedDict(
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
                deg=(deg, 'The degree of the 2D polynomial'),
                mpflux=mpflux,
                # fear=fear,
                fear1d=fear1d
            )
            return calibration_dict
        else:
            print("!!! result is not acceptable, this FEAR is skipped")
            return None

    def read_star(self, fp_star):
        return np.rot90(fits.getdata(fp_star) - self.master_bias)


def sort_apertures(ap_trace: np.ndarray):
    """ sort ascend """
    nap = ap_trace.shape[0]
    ind_sort = np.arange(nap, dtype=int)
    for i in range(nap - 1):
        for j in range(i + 1, nap):
            ind_common = (ap_trace[i] >= 0) & (ap_trace[j] >= 0)
            if np.median(ap_trace[i][ind_common]) > np.median(ap_trace[j][ind_common]) and ind_sort[i] < ind_sort[j]:
                ind_sort[i], ind_sort[j] = ind_sort[j], ind_sort[i]
                # print(ind_sort)
    return ind_sort


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # mainWindow = QtWidgets.QMainWindow()
    bfosc = UiBfosc()
    # ui.setupUi(mainWindow)
    # ui.initUi(mainWindow)

    bfosc.show()
    sys.exit(app.exec_())


