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
from astropy.time import Time
from skimage.filters import gaussian
from scipy.signal import medfilt2d

matplotlib.use('Qt5Agg')
matplotlib.rcParams["font.size"] = 5


class UiBfosc(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, nwv=None,  wavecalibrate=True):
        super(UiBfosc, self).__init__(parent)
        # data
        #print('The index of hdulist: ihdu = (must be integer e.g. 1)')
        ihdu = 0
        ihdu = int(ihdu)
        self.ihdu = ihdu
        self._wd = ""
        self.datatable = None
        self.pos = []
        self.pos_temp = [0, 0]
        self.master_bias = None
        self.master_flat = None
        self.trace_handle = []
        self.strimy = [0, 2834] #strim pixal on y axis (ystart, yend) 
        self.strimx = [51, 410] #strim pixal on x axis (xstart, xend)
        self.strimdy = self.strimy[1] - self.strimy[0]
        self.strimdx = self.strimx[1] - self.strimx[0]
        # the number of pixel of in the direction of dispersion (the number of pixel of wavelength)
        if nwv is None: self.nwv = self.strimdy 
        self.ap_trace = np.zeros((0, self.nwv), dtype=int)
        self._lamp = None
        self.aperture_image = None
        self. wavecalibrate =  wavecalibrate
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
        test_dir = "/home/sdb216/sdOBdata/Documents/Feige64/data/spec/feige64_spec_all_20230210/P200feige64DATA-tolijiao/"
        self._wd = test_dir
        self.lineEdit_wd.setText(test_dir)
        self._lamp = joblib.load("../template/fear_template_blue.z")
        apfname = f'{self._wd}/ap.dump'
        self.ap = joblib.load(apfname)
        self.ap_trace = self.ap.ap_center_interp

    def initUi(self):
        self.toolButton.clicked.connect(self._select_wd)
        self.toolButton_load_lamp.clicked.connect(self._select_lamp)
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

    def _select_lamp(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Open LAMP", "dump files (*.dump)")
        print(fileName)
        self.lineEdit_lamp.setText(fileName)
        self._lamp = joblib.load(fileName)
        print("Lamp loaded!")

    def _set_wd(self):
        self._wd = self.lineEdit_wd.text()

    def _make_datatable(self):
        # get file list
        fps_full = glob.glob(self.lineEdit_wd.text() + "/*.fits")
        fps_full.sort()
        self.fps_full = fps_full
        fps = [os.path.basename(_) for _ in fps_full]
        self.nfp = len(fps)

        imgtype = np.asarray([fits.getheader(fp)["OBJECT"] for fp in fps_full])
        imgtype1 = np.asarray([fits.getheader(fp)["IMGTYPE"] for fp in fps_full])
        exptime = np.asarray([fits.getheader(fp)["EXPTIME"] for fp in fps_full])
        types = np.zeros_like(imgtype)
        self.type_dict = OrderedDict(drop=0, bias=1, flat=2, lamp=3, star=4)
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
            elif ("arcs" in imgtype[i].lower()) or ('fear' in imgtype[i].lower())\
                or ('cal' in imgtype1[i].lower()): #and (exptime[i] == 300 or "arcs" in fps_full[i].lower()):
                types[i] = "lamp"
            #elif "light" in imgtype[i].lower() and (exptime[i] != 300 or "target" in fps_full[i].lower()):
            #    types[i] = "star"
            else:
                types[i] = "star"

        self.datatable = table.Table(
            data=[fps, imgtype, exptime, types],
            names=["filename", "imagetype", "exptime", "type"])
        # print(self.datatable["type"])

    def _update_datatable(self):
        # print(self.datatable["type"])
        self.datatable["type"] = [self.type_list[self.tableWidget_files.cellWidget(irow, 3).currentIndex()] for irow in range(self.nfp)]
        self._refresh_datatable()
        self.datatable.write(self._wd+"/catalog.csv", overwrite=True)

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
            #img = fits.getdata(fp_selected)
            img = self._read_img(fp_selected)
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
        #self.ax.set_ylim(850, 2048)
        #self.ax.set_ylim(self.strimy)
        #self.ax.set_xlim(0, 2048)
        #self.ax.set_xlim(self.strimx)
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

    def _read_img(self, fp_star):
        hdu = fits.open(fp_star)
        ihdu = self.ihdu
        data = hdu[ihdu].data
        xs, xe = self.strimx
        ys, ye = self.strimy
        data = data[ys:ye, xs:xe]
        hdu.close()
        #return np.rot90(data - self.master_bias)
        return data

    def _proc_bias(self):
        if self.datatable is None:
            pass
        fps_bias = []
        for i in range(self.nfp):
            if self.datatable["type"][i]=="bias":
                fps_bias.append(self.fps_full[i])
                print("appending BIAS: {}".format(self.fps_full[i]))
        master_bias = np.median(np.array([self._read_img(fp) for fp in fps_bias]), axis=0)
        self.master_bias = master_bias
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
        master_flat = np.median(np.array([self._read_img(fp) for fp in fps_flat]), axis=0)
        self.master_flat = master_flat
        self.master_flat -= self.master_bias
        self._draw_img(gaussian_filter(self.master_flat, sigma=2))
        # import joblib
        # joblib.dump(self.master_flat, "/Users/cham/projects/bfosc/20200915_bfosc/master_flat.dump")
        print(">>> FLAT processed!")

    def _clear_aperture(self):
        # from twodspec.trace import trace_naive_max
        # self.ap_trace = trace_naive_max(self.master_flat.T, sigma=7, maxdev=10, irow_start=1300) 
        self.ap_trace = np.zeros((0, self.nwv), dtype=int)
        self._update_nap()
        # print(self.ap_trace.shape)
        self._draw_aperture()

    def _draw_aperture(self):
        if len(self.ax.lines) > 1:
            for line in self.ax.get_lines()[1:]:  # ax.lines:
                line.remove()
        for _trace in self.ap_trace:
            ind_plot = _trace > 0
            #self.ax.plot(np.arange(self.strimdx)[ind_plot], _trace[ind_plot], "w-", lw=1)
            self.ax.plot(_trace[ind_plot], np.arange(self.nwv)[ind_plot], "w-", lw=1) # lijiao
        self.canvas.draw()

    def _choose_aperture_image(self):
        '''
        choose proper image to trace the aperture
        '''
        print('#----Print the fits (or fit) name for tracing aperture, e.g. blue0059.fits')
        print('#----should be star fits for HIRES of None ')
        ind_elected = self.tableWidget_files.currentRow()
        fname = self.fps_full[ind_elected]
        print(fname)
        #fname = input()
        if fname[-5:] == '.fits':
            #fname = os.path.basename(fname)
            #fname = os.path.join(self._wd, fname)
            self.fname_aperture = fname
            image_aperture = self.read_star(fname)
            self.aperture_image = image_aperture
        else:
            self.aperture_image = self.master_flat

    def trace_local_max(self, img, starting_row, starting_col, maxdev=10, fov=20, ntol=5):
        """ trace aperture from a starting point (row, col) """
        nrow, ncol = img.shape
        assert ntol < starting_row < nrow - ntol
        # find the local max for this column
        this_ind_row = -np.ones(nrow, dtype=int)
        # this_ind_col = np.arange(ncol, dtype=int)
        # search this column
        this_ind_row[starting_row] = self.find_local_max(
            img[starting_row, :], starting_col, maxdev=maxdev, fov=fov)
        # search up
        for irow in range(starting_row - 1, -1, -1):
            # find the closest finite value
            irow_up = np.min((irow + 1 + ntol, starting_row + 1))
            irow_closest_valid = np.where(this_ind_row[irow + 1:irow_up] > 0)[0]
            if len(irow_closest_valid) > 0:
                this_ind_row[irow] = self.find_local_max(
                    img[irow, :], this_ind_row[irow + 1:irow_up][irow_closest_valid[0]], maxdev=maxdev, fov=fov)
            else:
                break
        # search down
        for irow in range(starting_row + 1, nrow, 1):
            # find the closest finite value
            irow_down = np.max((irow - ntol, starting_row))
            irow_closest_valid = np.where(this_ind_row[irow_down:irow] > 0)[0]
            if len(irow_closest_valid) > 0:
                this_ind_row[irow] = self.find_local_max(
                    img[irow, :], this_ind_row[irow_down:irow][irow_closest_valid[-1]], maxdev=maxdev, fov=fov)
            else:
                break
        return this_ind_row


    def find_local_max(self, rowdata, icol, maxdev=10, fov=20):
        if icol < 0:
            return -1
        ind0 = icol - fov
        if ind0 < 0:
           ind0 = 0
        indmax = np.argmax(rowdata[ind0:icol + fov + 1])
        if np.abs(indmax - fov) > maxdev:
            return -1
        else:
            return indmax + icol - fov


    def _add_aperture(self):
        #try:
        #if self.aperture_image is None:
        self._choose_aperture_image()
        _trace = self.trace_local_max(
            gaussian_filter(self.aperture_image, sigma=5),
            *np.asarray(self.pos_temp[::-1], dtype=int), maxdev=10, fov=10, ntol=10)
        self._trace = _trace
        if np.sum(_trace>0)>100:
            self.ap_trace = np.vstack((self.ap_trace, _trace.reshape(1, -1)))
            self._draw_aperture()
            self._update_nap()
        print("An aperture is added")
        #except Exception as _e:
        #    print("Error occurred, aperture not added!")

    def _del_aperture(self):
        if self.ap_trace.shape[0] == 0:
            pass
        dx = np.arange(self.strimdx) - self.pos_temp[0]#lijiao
        dy = np.arange(self.strimdy) - self.pos_temp[1] #lijiao
        dxx, dyy = np.meshgrid(dx, dy) #lijiao
        #d = np.abs(dx ** 2 + dy ** 2)
        d = np.abs(dxx**2 + dyy**2)
        ind_min = np.argmin(d)
        #ind_min_ap, ind_min_pix = np.unravel_index(ind_min, self.ap_trace.shape)
        #print(f'self.ap_trace.shape= {self.ap_trace.shape}, d.shape ={d.shape}')
        _, ind_min_pix = np.unravel_index(ind_min, d.shape)
        ind_min_ap = 0
        self.ap_trace = self.ap_trace[np.arange(self.ap_trace.shape[0])!=ind_min_ap]
        self._update_nap()
        self._draw_aperture()

    def _save_aperture(self):
        #from twodspec.aperture import Aperture
        from aperture import Aperturenew as Aperture
        # print(self.ap_trace[:,0])
        self._choose_aperture_image()
        self.ap_trace = self.ap_trace[sort_apertures(self.ap_trace)]
        # fit
        #self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=12)
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=12) # lijiao
        #self.ap.get_image_info(np.rot90(self.aperture_image))#lijiao
        self.ap.get_image_info(self.aperture_image)#lijiao
        self.ap.polyfitnew(deg = 2, ystart=700)
        # replace old traces
        #self.ap_trace = self.ap.ap_center_interp[:, ::-1]
        self.ap_trace = self.ap.ap_center_interp # lijiao
        # fit again
        #self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=10)
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=10)
        #self.ap.get_image_info(np.rot90(self.master_flat)) # lijiao
        self.ap.get_image_info(self.master_flat) # lijiao
        self.ap.polyfitnew(deg=2, ystart=700)
        self._draw_aperture()
        import joblib
        joblib.dump(self.ap, os.path.join(self._wd+"ap.dump"))
        print("Aperture saved to ", os.path.join(self._wd+"ap.dump"))

    def _update_nap(self):
        self.lineEdit_nap.setText("N(ap)={}".format(self.ap_trace.shape[0]))

    def _modify_header(self, fname):
        header = fits.getheader(fname, 0)
        header['TRIMSEC'] = ('[{}:{}, {}:{}]'.format(*self.strimx, *self.strimy), 'PYEXSPEC: Section of useful data')
        ap_upper = self.ap.ap_upper_interp
        ap_lower = self.ap.ap_lower_interp
        nap, nrow = ap_upper.shape
        ind = int(nrow/2)
        for i in np.arange(1, nap+1):
            _key =f'APNUM{i}'
            _value = f'{i} {i} {ap_lower[i-1][ind]:2f} {ap_upper[i-1][ind]:.2}'
            header[_key] = _value
        header['WCSDIM']  = 1
        header['CDELT1'] = 1.
        header['LTM1_1'] = 1.
        header['WAT0_001']= 'system=equispec'
        header['WAT1_001']= 'wtype=linear label=Pixel'
        header['WAT2_001']= 'wtype=linear'
        return header

    def _proc_all(self):
        if self._lamp is None:
            print("LAMP not loaded!")
        nrow, ncol = self.master_flat.shape
        wavecalibrate = self.wavecalibrate
        # compute blaze & sensitivity
        #flat_bg = self.ap.background(np.rot90(self.master_flat), q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(21, 21))
        #self.blaze, self.sensitivity = self.ap.make_normflat(np.rot90(self.master_flat)-flat_bg, )
        #flat_bg = self.ap.background(self.master_flat, q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(15, 15))
        #from skimage.filters import gaussian
        #from scipy.signal import medfilt2d
        flat_bg = medfilt2d(self.master_flat, kernel_size=(21, 21))
        #flat_bg = gaussian(flat_bg, sigma=(20, 20))
        #flat_bg= self.ap.backgroundnew(self.master_flat, q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(21, 21))
        self.flat_bg = flat_bg
        self.blaze, self.sensitivity = self.ap.make_normflat(self.master_flat)

        print("""[4.1] extracting star1d (~5s/star) """)
        # loop over stars
        fps_star = self._gather_files("star")
        n_star = len(fps_star)
        for i_star, fp in enumerate(fps_star):
            print("  |- ({}/{}) processing STAR ... ".format(i_star, n_star), end="")
            _dir = os.path.dirname(fp)
            dirdump = os.path.join(_dir, 'dump')
            self.dirdump = dirdump
            if not os.path.exists(dirdump): os.makedirs(dirdump)
            fp_out = "{}/star-{}.dump".format(dirdump, os.path.basename(fp))
            star = self.read_star(fp)
            header = self._modify_header(fp)
            isot =  f'{header["UTSHUT"]}'
            jd = Time(isot, format='isot').jd
            star -= self.ap.backgroundnew(star, q=(10, 10), npix_inter=5, sigma=(20, 20), kernel_size=(21, 21))
            star /= self.sensitivity
            gain = 1/header['GAIN']
            ron = header['RON']
            star1d = self.ap.extract_all(star, gain=gain, ron=ron, n_jobs=1, verbose=False)
            print("writing to {}".format(fp_out))
            star1d["blaze"] = self.blaze
            star1d["UTC-OBS"] = isot
            star1d["jd"] = jd
            star1d["EXPTIME"] = header["EXPTIME"]
            star1d["header"] = header
            star1d['STRIM'] = ('[{}:{}, {}:{}]'.format(*self.strimx, *self.strimy), 'PYEXSPEC data reduce')
            #star1d["header"] = header
            joblib.dump(star1d, fp_out)

        print("[5.1] load LAMP template & LAMP line list")
        """ loop over LAMP """
        fps_lamp = self._gather_files("lamp")
        n_lamp = len(fps_lamp)
        for i_lamp, fp in enumerate(fps_lamp):
            print("  |- ({}/{}) processing LAMP {} ... ".format(i_lamp, n_lamp, fp))
            _dir = os.path.dirname(fp)
            dirdump = os.path.join(_dir, 'dump')
            if not os.path.exists(dirdump): os.makedirs(dirdump)
            fp_out = "{}/lamp-{}.dump".format(dirdump, os.path.basename(fp))
            res = self._proc_lamp(fp, nsigma=4, verbose=False, wavecalibrate=True, deg=3)
            if res is not None:
                print("  |- writing to {}".format(fp_out))
                joblib.dump(res, fp_out)

        if wavecalibrate:
            print("""[6.0] make stats for the ARCS solutions """)
            fps_lamp_res = glob.glob(os.path.join(self._wd, 'dump', "lamp-*"))
            fps_lamp_res.sort()
            tlamp = table.Table([joblib.load(_) for _ in fps_lamp_res])

            """ a statistic figure of reduced lamp """
            fig = plt.figure(figsize=(9, 7))
            ax = plt.gca()
            ax.plot(tlamp['jd'], tlamp["rms"] / 4500 * 3e5, 's-', ms=10, label="RMS")
            ax.set_xlabel("JD")
            ax.set_ylabel("RMS [km s$^{-1}$]")
            ax.set_title("The precision of LAMP calibration @4500A")
            ax.legend(loc="upper left")

            axt = ax.twinx()
            axt.plot(tlamp['jd'], tlamp["nlines"], 'o-', ms=10, color="gray", label="nlines");
            axt.set_ylabel("N(Lines)")
            axt.legend(loc="upper right")

            fig.tight_layout()
            fig.savefig("{}/lamp_stats.pdf".format(self._wd))
        pass


    def _extract_sum(self, im,ap_center_interp, ap_width=15, gain=1., ron=0):
        """ extract an aperture by sum the values given the aperture center

        Parameters
        ----------
        im : ndarray
            The target image.
        ap_center_interp : ndarray
            The ap_center_interp.
        ap_width : float, optional
            The width of aperture / pix. The default is 15.
        gain : float, optional
            The gain of CCD. The default is 1..
        ron : flaot, optional
            The readout noise of CCD. The default is 0.

        Returns
        -------
        a dict consisting of many results.

        """
        from twodspec.extract import get_aperture_section
        # 1. get aperture section
        ap_im, ap_im_xx, ap_im_yy, ap_im_xx_cor = get_aperture_section(
            im, ap_center_interp, ap_width=ap_width)

        # set negative values to 0
        ap_im = np.where(ap_im > 0, ap_im, 0)

        # error image
        ap_im_errerr = ap_im / gain + ron ** 2.

        return dict(
            # ----- simple extraction -----
            spec_sum=ap_im.sum(axis=1),
            err_sum=np.sqrt(ap_im_errerr.sum(axis=1)),
            # ----- reconstructed profile -----
            # ----- reconstructed aperture -----
            ap_im=ap_im,
            # ----- aperture coordinates -----
            ap_im_xx=ap_im_xx,
            ap_im_xx_cor=ap_im_xx_cor,
            ap_im_yy=ap_im_yy,
        )


    def extract_sum(self, im, gain=1., n_jobs=-1, ron=0,
                    verbose=False, backend="multiprocessing"):
        """ extract all apertures with simple summary
        Parameters
        ----------
        im : ndarray
            The target image.
        gain : float, optional
            The gain of CCD. The default is 1..
        ron : flaot, optional
            The readout noise of CCD. The default is 0.
        n_jobs : int, optional
            The number of processes launched. The default is -1.
        verbose :
            joblib verbose
        backend :
            joblib backend

        Returns
        -------
        dict
            a dict sconsisting of many results.

        """
        # extract all apertures in parallel
        rs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(joblib.delayed(self._extract_sum)(
            im, self.ap.ap_center_interp[i],ap_width=self.ap.ap_width, gain=gain, ron=ron) for i in range(self.ap.nap))
        # reconstruct results
        result = dict(
            # simple extraction
            spec_sum=np.array([rs[i]["spec_sum"] for i in range(self.ap.nap)]),
            err_sum=np.array([rs[i]["err_sum"] for i in range(self.ap.nap)]),
        )
        return result

    def grating_equation(self, x, z, deg=4, nsigma=3, min_select=None, verbose=True):
        """
        Fit a grating equation (1D polynomial function) to data 
        Parameters
        ----------
        x : array
            x coordinates of emission lines
        z : array
            The true wavelengths of lines.
        deg : tuple, optional
            The degree of the 1D polynomial. The default is (4, 10).
        nsigma : float, optional
            The data outside of the nsigma*sigma radius is rejected iteratively. The default is 3.
        min_select : int or None, optional
            The minimal number of selected lines. The default is None.
        verbose :
            if True, print info

        Returns
        -------
        pf1, indselect

        """
        from twodspec.polynomial import Poly1DFitter
        indselect = np.ones_like(x, dtype=bool)
        iiter = 0
        # pf1
        while True:
            pf1 = Poly1DFitter(x[indselect], z[indselect], deg=deg, pw=1, robust=False)
            z_pred = pf1.predict(x)
            z_res = z_pred - z
            sigma = np.std(z_res[indselect])
            indreject = np.abs(z_res[indselect]) > nsigma * sigma
            n_reject = np.sum(indreject)
            if n_reject == 0:
                # no lines to kick
                break
            elif isinstance(min_select, int) and min_select >= 0 and np.sum(indselect) <= min_select:
                # selected lines reach the threshold
                break
            else:
                # continue to reject lines
                indselect &= np.abs(z_res) < nsigma * sigma
                iiter += 1
            if verbose:
                print("@grating_equation: iter-{} \t{} lines kicked, {} lines left, rms={:.5f} A".format(
                    iiter, n_reject, np.sum(indselect), sigma))
        pf1.rms = sigma

        if verbose:
            print("@grating_equation: {} iterations, rms = {:.5f} A".format(iiter, pf1.rms))
        return pf1, indselect

    def _proc_lamp(self, fp, nsigma=2.5, verbose=False, wavecalibrate=True, deg=4):
        """ read lamp """
        lamp = self.read_star(fp)
        lamp /= self.sensitivity
        header = fits.getheader(fp)
        gain = 1/header['GAIN']
        ron = header['RON']
        # unnecessary to remove background
        # fear -= apbackground(fear, ap_interp, q=(10, 10), npix_inter=5,sigma=(20, 20),kernel_size=(21,21))
        # extract 1d fear
        lamp1d = self.extract_sum(lamp,gain=gain, n_jobs=1, ron=ron)["spec_sum"]
        # remove baseline
        # fear1d -= np.median(fear1d)

        """ corr2d to get initial estimate of wavelength """
        from twodspec import thar
        if wavecalibrate:
            wave_init = thar.corr_thar(self._lamp["wave"][:, ::-1], self._lamp["flux"][:, ::-1], lamp1d[:, ::-1], maxshift=50) 
            """ find thar lines """
            tlines = thar.find_lines(wave_init, lamp1d[:, ::-1], self._lamp["linelist"], npix_chunk=8, ccf_kernel_width=1.5)
            ind_good = np.isfinite(tlines["line_x_ccf"]) & (np.abs(tlines["line_x_ccf"] - tlines["line_x_init"]) < 10) & (
                    (tlines["line_peakflux"] - tlines["line_base"]) > 100) & (
                               np.abs(tlines["line_wave_init_ccf"] - tlines["line"]) < 3)
            tlines.add_column(table.Column(ind_good, "ind_good"))
            # tlines.show_in_browser()
            wave_init1 = wave_init[:, ::-1] 
        else:
            wave_init1 = None; tlines=None

        """ clean each order """
        from twodspec.polynomial import Poly1DFitter

        def clean(pw=1, deg=3, threshold=0.1, min_select=10):
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

        header = self._modify_header(fp)
        isot =  f'{header["UTSHUT"]}'
        jd = Time(isot, format='isot').jd
        if wavecalibrate: 
            print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
            clean(pw=1, deg=deg, threshold=0.8, min_select=20)
            clean(pw=1, deg=deg, threshold=0.4, min_select=20)
            clean(pw=1, deg=deg, threshold=0.2, min_select=20)
            print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
            tlines = tlines[tlines["ind_good"]]

            """ fitting grating equation """
            x = tlines["line_x_ccf"]  # line_x_ccf/line_x_gf
            y = tlines["order"]
            z = tlines["line"]
            pf1, indselect = self.grating_equation(
                   x, z, deg=deg, nsigma=nsigma, min_select=210, verbose=True)
            tlines.add_column(table.Column(indselect, "indselect"))
            mpflux = np.median(tlines["line_peakflux"][tlines["indselect"]])
            rms = np.std((pf1.predict(x) - z)[indselect])
            nlines = np.sum(indselect)
            mx = np.array([np.arange(self.nwv)])[:,::-1]
            wave_solu = pf1.predict(mx)  # polynomial fitter
            print("  |- nlines={}  rms={:.4f}A  mpflux={:.1f}".format(nlines, rms, mpflux))
            print(f'###----------------\npf1.rms = {pf1.rms}\n###------------------')
        else:
            pf1 = None; rms =np.nan; mpflux=None; nlines=None
            wave_solu = None
        if (0.01 < rms < 1) or (wavecalibrate is False):
            if wavecalibrate is False:
               print("!!! The wavelenth of this LAMP is not calibrated")
            # reasonable
            # mpflux
            # rms
            # predict wavelength solution
            nx, norder = lamp1d.shape
            mx, morder = np.meshgrid(np.arange(norder), np.arange(nx))
            # result
            calibration_dict = OrderedDict(
                fp=fp,
                jd=jd,
                exptime=header['EXPTIME'],
                header = header,
                STRIM = '[{}:{}, {}:{}]'.format(*self.strimx, *self.strimy),
                wave_init=wave_init1,
                wave_solu=wave_solu,
                tlines=tlines,
                nlines=nlines,
                rms=rms,
                pf1=pf1,
                deg=(deg, 'The degree of the 1D polynomial'),
                mpflux=mpflux,
                # lamp=lamp,
                lamp1d=lamp1d,
                blaze = self.blaze,
            )
            return calibration_dict
        else:
            print("!!! result is not acceptable, this LAMP is skipped")
            return None

    def read_star(self, fp_star):
        hdu = fits.open(fp_star)
        ihdu = self.ihdu
        data = hdu[ihdu].data
        xs, xe = self.strimx
        ys, ye = self.strimy
        data = data[ys:ye, xs:xe]
        hdu.close()
        #return np.rot90(data - self.master_bias)
        return data - self.master_bias

    def contourf(self, img):
        rows, _cols = img.shape
        _xx, _yy = np.meshgrid(np.arange(_cols), np.arange(_rows))
        plt.contourf(_xx, _yy, _img)

    def writefits(self, header, data, fout):
        hdu = fits.HDUList([fits.PrimaryHDU(header=_header, data=_data)])
        hdu.writeto(fout)


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
