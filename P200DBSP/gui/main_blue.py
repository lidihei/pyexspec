import sys, os, glob
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except:
    from PyQt6 import QtCore, QtGui, QtWidgets
from P200DBPS import Ui_MainWindow
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
from astropy.time import Time,TimeDelta
from skimage.filters import gaussian
from scipy.signal import medfilt2d
from astropy.nddata import CCDData
from astropy.nddata import block_replicate
import ccdproc as ccdp
from photutils.segmentation import detect_sources
from convenience_functions import show_image, display_cosmic_rays
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Qt5Agg')
matplotlib.rcParams["font.size"] = 5


class UiBfosc(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None, nwv=None,  wavecalibrate=True, site=None):
        '''
        site: the site of observatory
              if None: site = EarthLocation.of_site('Palomar') ## Palomar Observotory
              e.g.
              from astropy import coordinates as EarthLocation
              site = EarthLocation.from_geodetic(lat=26.6951*u.deg, lon=100.03*u.deg, height=3200*u.m) Lijiang observation
        '''
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
        self.strimx = [75, 392] #strim pixal on x axis (xstart, xend)
        self.strimdy = self.strimy[1] - self.strimy[0]
        self.strimdx = self.strimx[1] - self.strimx[0]
        self.rot90 = False
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
        self.site = EarthLocation.of_site('Palomar') if site is None else site
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
        test_dir = "/share/data/lijiao/Documents/sdOB/example/lan11/data/spec/P200_DBPS/20220219_rawdata/blue"
        self._wd = test_dir
        self.lineEdit_wd.setText(test_dir)
        self._lamp = joblib.load("../template/fear_template_blue.z")
        #apfname = f'{self._wd}/ap.dump'
        #print(apfname)
        #self.ap = joblib.load(apfname)
        #self.ap_trace = self.ap.ap_center_interp

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
        ra = np.asarray([fits.getheader(fp)["RA"] for fp in fps_full])
        dec = np.asarray([fits.getheader(fp)["DEC"] for fp in fps_full])
        UTSHUT = np.asarray([fits.getheader(fp)["UTSHUT"] for fp in fps_full])
        types = np.zeros_like(imgtype)
        self.type_dict = OrderedDict(drop=0, bias=1, flat=2, arc=3, star=4)
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
                types[i] = "arc"
            #elif "light" in imgtype[i].lower() and (exptime[i] != 300 or "target" in fps_full[i].lower()):
            #    types[i] = "star"
            else:
                types[i] = "star"

        self.datatable = table.Table(
            data=[fps, imgtype, exptime, types, ra, dec, UTSHUT],
            names=["filename", "imagetype", "exptime", "type", 'ra', 'dec', 'UTSHUT'])
        self.datatable_arc = self.datatable[self.datatable['type']=='arc']
        # print(self.datatable["type"])

    def _update_datatable(self):
        # print(self.datatable["type"])
        self.datatable["type"] = [self.type_list[self.tableWidget_files.cellWidget(irow, 3).currentIndex()] for irow in range(self.nfp)]
        self._refresh_datatable()
        #self.datatable.write(self._wd+"/catalog.csv", overwrite=True)
        dire_table = os.path.join(self._wd, 'TABLE')
        if not os.path.exists(dire_table): os.makedirs(dire_table)
        self.datatable.write(os.path.join(dire_table, 'filelist_table.csv'), overwrite=True)

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
        n_col = 7
        self.tableWidget_files.setColumnCount(n_col)
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
            self.tableWidget_files.setItem(irow, 4, QtWidgets.QTableWidgetItem(str(self.datatable["ra"][irow])))
            self.tableWidget_files.setItem(irow, 5, QtWidgets.QTableWidgetItem(str(self.datatable["dec"][irow])))
            self.tableWidget_files.setItem(irow, 6, QtWidgets.QTableWidgetItem(str(self.datatable["UTSHUT"][irow])))

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
        if self.rot90: data = np.rot90(data)
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
        self.master_bias_err_squared = np.nanvar(master_bias)
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
        flats = []
        flats_err_squared = []
        for fp in fps_flat:
            flats.append(self.read_star(fp, remove_cosmic_ray=False))
            flats_err_squared.append(self.image_err_squared)
        self.flats = flats
        #master_flat = np.median(np.array([self._read_img(fp) for fp in fps_flat]), axis=0)
        master_flat = np.median(np.array(flats), axis=0)
        master_flat_err_squared = np.median(np.array(flats_err_squared), axis=0)/len(fps_flat)
        self.master_flat = master_flat
        self.master_flat_err_squared = master_flat_err_squared
        #self.master_flat -= self.master_bias
        #self._draw_img(gaussian_filter(self.master_flat, sigma=2))
        self._draw_img(self.master_flat)
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
        ap_width = 12
        #from twodspec.aperture import Aperture
        from pyexspec.aperture import Aperturenew as Aperture
        # print(self.ap_trace[:,0])
        self._choose_aperture_image()
        self.ap_trace = self.ap_trace[sort_apertures(self.ap_trace)]
        # fit
        #self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=12)
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=12) # lijiao
        #self.ap.get_image_info(np.rot90(self.aperture_image))#lijiao
        self.ap.get_image_info(self.aperture_image)#lijiao
        self.ap.polyfitnew(deg = 2, ystart=400)
        # replace old traces
        #self.ap_trace = self.ap.ap_center_interp[:, ::-1]
        self.ap_trace = self.ap.ap_center_interp # lijiao
        # fit again
        #self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=10)
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=ap_width)
        #self.ap.get_image_info(np.rot90(self.master_flat)) # lijiao
        self.ap.get_image_info(self.master_flat) # lijiao
        self.ap.polyfitnew(deg=2, ystart=400)
        self.ap.ap_width = ap_width
        self._draw_aperture()
        import joblib
        joblib.dump(self.ap, os.path.join(self._wd, "ap.dump"))
        print("Aperture saved to ", os.path.join(self._wd, "ap.dump"))

    def estimte_ap_width(self, flux, A, mu, sigma, k, c, Nsigma=3, show=False):
        '''
        Fiting apeture with a Gaussian linear function
        F(X) = A /(sqrt(2*np.pi)*sigma)exp((x-mu)**2/sigma**2)+kx+c
        ap_width = Nsimga * simga
        returns:
        ----------------
        ap_width
        '''
        from fit_function import gaussian_linear_func
        from scipy.optimize import curve_fit
        x = np.arange(len(flux))
        p0 = [A, mu, sigma, k, c]
        popt, pcov = curve_fit(gaussian_linear_func, x, flux, p0=p0)
        sigma = np.abs(popt[2])
        ap_width = Nsigma*sigma
        if show:
           xdens = np.arange(0, len(flux), 0.01)
           yfit = gaussian_linear_func(xdens, *popt)
           fig, ax = plt.subplots(1,1)
           plt.plot(x, flux)
           plt.plot(xdens, yfit, label='Gaussian fit')
           plt.legend()
           plt.xlabel('x coords (pixel)')
           plt.ylabel('counts')
           plt.axvline(x=popt[1] - ap_width, ls='--')
           plt.axvline(x=popt[1] + ap_width, ls='--')
        return ap_width


    def _adjust_aperture_star(self, image_star, ap_width = 12, Nsigma=3, strim_l=50, strim_r=300):
        '''
        adjust aperture for a specific image
        ap_width is half width of the aperture
        '''
        #try:
        #from aperture import Aperturenew as Aperture
        from pyexspec.aperture import Aperturenew as Aperture
        ap_init = self.ap
        for i, ap_center in enumerate(ap_init.ap_center):
            starting_row = int(len(ap_center)/2)
            starting_col = int(ap_center[starting_row])
            _trace = self.trace_local_max(
                gaussian_filter(image_star, sigma=5),
                starting_row, starting_col, maxdev=10, fov=10, ntol=10)
            self._trace = _trace
            if np.sum(_trace>0)>100:
                self.ap_trace[i] = _trace
        self.ap_trace = self.ap_trace[sort_apertures(self.ap_trace)]
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=12)
        self.ap.get_image_info(image_star)
        self.ap.polyfitnew(deg = 2, ystart=400)
        self.ap_trace = self.ap.ap_center_interp
        if ap_width is None:
            print(f'estimate ap_width using Gaussian function, ap_width = {Nsigma}sigma')
            ap_widths = []
            for i, ap_center in enumerate(ap_init.ap_center):
                N = 5
                _row = int(len(ap_center)/2)
                rows = np.arange(_row-N, _row+N)
                for j in rows:
                    mu = ap_center[j]
                    self._mu = mu
                    #print(f'mu = {mu}')
                    flux = image_star[j]
                    self._flux = flux
                    A = flux[int(mu)] - flux[int(mu)-10]
                    flux = flux[strim_l:strim_r]
                    _ap_width = self.estimte_ap_width(flux, A, mu-strim_l, 5, 0, 0, Nsigma=Nsigma, show=False)
                    ap_widths.append(_ap_width)
            ap_width = int(np.ceil(np.median(ap_widths)))
        print(f'  |-ap_width = {ap_width}')
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=ap_width)
        self.ap.get_image_info(image_star)
        self.ap.polyfitnew(deg=2, ystart=starting_row)
        joblib.dump(self.ap, os.path.join(self._wd, "ap.dump"))
        print("  |-Adjust_star aperture")

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

    def remove_cosmic_ray(self, image, readnosize=2, sigclip=7, verbose=True):
        '''
        image [2D array]:
        '''
        ccd = CCDData(image, unit='adu')
        new_ccd = ccdp.cosmicray_lacosmic(ccd, readnoise=readnoise, sigclip=sigclip, verbose=verbose)
        return new_ccd


    def _extract_single_star(self, fp, master_flat=None, master_flat_err_squared=None,
                             adjust_aperture=True, irow=1000, show=False,
                             site=None, kind='barycentric'):
        '''
        site: the site of observatory
              e.g.
              from astropy import coordinates as EarthLocation
              site = EarthLocation.from_geodetic(lat=26.6951*u.deg, lon=100.03*u.deg, height=3200*u.m) Lijiang observation
              site = EarthLocation.of_site('Palomar') ## Palomar Observotory
        '''
        from twodspec.extract import extract_aperture
        _dir = os.path.dirname(fp)
        bg_deg = 2
        dirdump = os.path.join(_dir, f'dump_bgdeg{bg_deg}')
        self.dirdump = dirdump
        if not os.path.exists(dirdump): os.makedirs(dirdump)
        fp_out = "{}/star-{}.dump".format(dirdump, os.path.basename(fp))
        star = self.read_star(fp).copy()
        if adjust_aperture:
           self._adjust_aperture_star(star, ap_width = None, Nsigma=3.5)
        self.blaze, self.sensitivity = self.ap.make_normflat(self.master_flat)
        ####------------ fit back ground --------------------
        ap_star = self.ap
        star_err_squared = self.image_err_squared
        bg = ap_star.backgroundnew(star, longslit = True, Napw_bg=3, deg=bg_deg, num_sigclip=5, Napw=4, verbose=False)
        star_withbg = star.copy()
        star_withbg_err_squared =star_err_squared.copy()
        #### ---  image /divided flat
        star_withbg_divide_flat = star_withbg/master_flat
        star_withbg_divide_flat_err_squared = star_err_squared/master_flat**2 + star_withbg**2*master_flat_err_squared/master_flat**4
        #### ---  image - bg /divided flat
        star -= bg
        star_divide_flat = star/master_flat
        star_divide_flat_err_squared = star_err_squared/master_flat**2 + star**2*master_flat_err_squared/master_flat**4
        self.star_image_divide_flat = star_divide_flat
        self.image_star = star
        self.image_star_err_squared = star_withbg_err_squared
        header = self._modify_header(fp)
        isot =  f'{header["UTSHUT"]}'
        time_shut = Time(isot, format='isot', location=site)
        jd = time_shut.jd
        if show:
           fig, ax = plt.subplots(1,1)
           plt.plot(star[irow])
           plt.plot(bg[irow])
        gain = 1/header['GAIN']
        ron = header['RON']
        #star1d = self.ap.extract_all(star, gain=gain, ron=ron, n_jobs=1, verbose=False)
        ##### extract with background
        star1d_withbg = extract_aperture(star_withbg, ap_star.ap_center, im_err_squared = star_withbg_err_squared, n_chunks=8,
                ap_width=ap_star.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron)
        star1d_withbg_divide_flat = extract_aperture(star_withbg_divide_flat, ap_star.ap_center,
                im_err_squared = star_withbg_divide_flat_err_squared, n_chunks=8,
                ap_width=ap_star.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron)
        ##### extract image - background
        star1d = extract_aperture(star, ap_star.ap_center, im_err_squared = star_err_squared, n_chunks=8,
                ap_width=ap_star.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron)
        star1d_divide_flat = extract_aperture(star_divide_flat, ap_star.ap_center,
                im_err_squared = star_divide_flat_err_squared, n_chunks=8,
                ap_width=ap_star.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron)
        #star1d = self._extract_sum(star, ap_star.ap_center, im_err_sqaured=star_err_squared, ap_width=ap_star.ap_width)
        #star1d_divide_flat = self._extract_sum(star_divide_flat, ap_star.ap_center, 
        #                     im_err_sqaured=star_divide_flat_err_squared, ap_width=ap_star.ap_width)
        #######--------find arc file and extract-------------------------------------
        c_star = SkyCoord(header['ra'], header['dec'], unit=(units.hourangle, units.deg))
        filename_arc = self.find_arc_for_star(c_star.ra, c_star.dec, jd, radius=5)
        if filename_arc is not None:
           print(f"  |-processing arc {os.path.basename(filename_arc)} for sience {os.path.basename(fp)}")
           fp_arc = os.path.join(self._wd, filename_arc)
           if not os.path.exists(dirdump): os.makedirs(dirdump)
           fp_out_arc = f"{dirdump}/lamp-{filename_arc}_for_{os.path.basename(fp)}.dump"
           arcdic = self._proc_lamp(fp_arc, ap_star, num_sigclip=3, verbose=False,
                                 suffix=f'_science_{os.path.basename(fp)}',
                                  wavecalibrate=True, deg=4, show=True)
           if arcdic is not None:
               print("  |- writing to {}".format(fp_out_arc))
               joblib.dump(arcdic, fp_out_arc)
           arcfile = os.path.basename(filename_arc)
        star1d["blaze"] = self.blaze
        star1d["UTC-OBS"] = isot
        star1d["jd"] = jd
        star1d["EXPTIME"] = header["EXPTIME"]
        star1d['STRIM'] = ('[{}:{}, {}:{}]'.format(*self.strimx, *self.strimy), 'PYEXSPEC data reduce')
        star1d['filename_arc'] = filename_arc
        star1d['wave_init'] = arcdic['wave_init']
        star1d['wave_solu'] = arcdic['wave_solu']
        star1d['rms_wave_calibrate'] =arcdic['rms']
        star1d['median_flat'] =self.median_master_flat
        #star1d["header"] = header
        #self.star1d_divide_flat = star1d_divide_flat
        #self.star1d = star1d
        for _key in star1d_divide_flat.keys():
            key_ = f'{_key}_divide_flat'
            star1d[key_] = star1d_divide_flat[_key]
        for _key in star1d_withbg.keys():
            key_ = f'{_key}_withbg'
            star1d[key_] = star1d_withbg[_key]
        for _key in star1d_withbg_divide_flat.keys():
            key_ = f'{_key}_withbg_divide_flat'
            star1d[key_] = star1d_withbg_divide_flat[_key]
        ###------ barycentric correction
        from pyexspec.utils import rvcorr_spec
        kind = 'barycentric'
        ltt = time_shut.light_travel_time(c_star,kind)
        borhjd = time_shut.tdb + ltt.tdb
        time_exp = TimeDelta(header['exptime'], format='sec')
        vcorr = c_star.radial_velocity_correction(obstime=time_shut+time_exp, kind=kind).to('km/s').value #baryrv =rv+barycorr(1+rv/c) (km/s) 
        star1d['bjd_shut'] = borhjd.jd
        star1d['barycorr'] = vcorr
        header['barycorr'] = (vcorr, 'rv + vcorr + rv * vcorr / c')
        star1d['wave_init_barycorr'] = rvcorr_spec(arcdic['wave_init'], vcorr, returnwvl=True)
        star1d['wave_solu_barycorr'] = rvcorr_spec(arcdic['wave_solu'], vcorr, returnwvl=True)
        ###------'heliocentric  correction
        kind = 'heliocentric'
        ltt = time_shut.light_travel_time(c_star,kind)
        borhjd = time_shut.jd + ltt.jd
        vcorr = c_star.radial_velocity_correction(obstime=time_shut, kind=kind).to('km/s').value #baryrv =rv+barycorr(1+rv/c) (km/s)
        star1d['hjd_shut'] = borhjd
        star1d['heliocorr'] = vcorr
        header['heliocorr'] = (vcorr, 'rv + vcorr + rv * vcorr / c')
        star1d['wave_init_heliocorr'] = rvcorr_spec(arcdic['wave_init'], vcorr, returnwvl=True)
        star1d['wave_solu_heliocorr'] = rvcorr_spec(arcdic['wave_solu'], vcorr, returnwvl=True)
        header['arcfile'] = (arcfile, 'arc file name')
        star1d["header"] = header
        print("  |- writing to {}".format(fp_out))
        joblib.dump(star1d, fp_out)
        return



    def _proc_all(self, show=False, irow=1000):
        if self._lamp is None:
            print("LAMP not loaded!")
        nrow, ncol = self.master_flat.shape
        wavecalibrate = self.wavecalibrate
        
        #flat_bg = self.ap.background(self.master_flat, q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(15, 15))
        #from skimage.filters import gaussian
        #from scipy.signal import medfilt2d
        from twodspec.extract import extract_aperture
        flat_bg = medfilt2d(self.master_flat, kernel_size=(21, 21))
        #flat_bg = gaussian(flat_bg, sigma=(20, 20))
        #flat_bg= self.ap.backgroundnew(self.master_flat, q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(21, 21))
        self.flat_bg = flat_bg
        master_flat = self.master_flat.copy()
        m_master_flat = np.nanmedian(master_flat)
        self.median_master_flat= m_master_flat
        master_flat /=  m_master_flat
        master_flat_err_squared = self.master_flat_err_squared.copy()
        master_flat_err_squared /= m_master_flat**2
        print("""[4.1] extracting star1d (~5s/star) """)
        # loop over stars
        fps_star = self._gather_files("star")
        n_star = len(fps_star)
        if show: fig, ax = plt.subplots(1,1)
        for i_star, fp in enumerate(fps_star):
            print("|--({}/{}) processing STAR ... ".format(i_star+1, n_star), end="")
            self._extract_single_star(fp, master_flat=master_flat,
                                master_flat_err_squared=master_flat_err_squared, site=self.site)

        #print("[5.1] load LAMP template & LAMP line list")
        #""" loop over LAMP """
        #fps_lamp = self._gather_files("arc")
        #n_lamp = len(fps_lamp)
        #for i_lamp, fp in enumerate(fps_lamp):
        #    print("  |- ({}/{}) processing LAMP {} ... ".format(i_lamp, n_lamp, fp))
        #    _dir = os.path.dirname(fp)
        #    dirdump = os.path.join(_dir, 'dump')
        #    if not os.path.exists(dirdump): os.makedirs(dirdump)
        #    fp_out = "{}/lamp-{}.dump".format(dirdump, os.path.basename(fp))
        #    res = self._proc_lamp(fp, ap_star, num_sigclip=3, verbose=False, wavecalibrate=True, deg=4, show=True)
        #    if res is not None:
        #        print("  |- writing to {}".format(fp_out))
        #        joblib.dump(res, fp_out)

        return


    def _extract_sum(self, im, ap_center_interp,im_err_sqaured=None, ap_width=15, gain=1., ron=0):
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
        if im_err_sqaured is None:
            ap_im_err_squared = ap_im / gain + ron ** 2.
        else:
            ap_im_err_squared, _, _, _ = get_aperture_section(
            im_err_sqaured, ap_center_interp, ap_width=ap_width)

        return dict(
            # ----- simple extraction -----
            spec_sum=ap_im.sum(axis=1),
            err_sum=np.sqrt(ap_im_err_squared.sum(axis=1)),
            # ----- reconstructed profile -----
            # ----- reconstructed aperture -----
            ap_im=ap_im,
            # ----- aperture coordinates -----
            ap_im_xx=ap_im_xx,
            ap_im_xx_cor=ap_im_xx_cor,
            ap_im_yy=ap_im_yy,
        )


    def extract_sum(self, im, ap_star=None, gain=1., n_jobs=-1, ron=0,
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
        if ap_star is None: ap_star = self.ap
        rs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(joblib.delayed(self._extract_sum)(
            im, ap_star.ap_center_interp[i],ap_width=ap_star.ap_width, gain=gain, ron=ron) for i in range(ap_star.nap))
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

    def _proc_lamp(self, fp, ap_star, num_sigclip=2.5,
                  line_peakflux=50,line_type = 'line_x_ccf',
                  verbose=False, wavecalibrate=True, deg=4, show=False,
                  min_select_lines = 10,
                  QA_fig_path = None, suffix=None):
        """ read lamp
        parameters:
        --------------------
        suffix [str]  the suffix of file name of QA for calibrating wave
        """
        lamp = self.read_star(fp)
        lampbasename = os.path.basename(fp)
        lamp /= self.sensitivity
        header = fits.getheader(fp)
        gain = 1/header['GAIN']
        ron = header['RON']
        # unnecessary to remove background
        # fear -= apbackground(fear, ap_interp, q=(10, 10), npix_inter=5,sigma=(20, 20),kernel_size=(21,21))
        # extract 1d fear
        lamp1d = self.extract_sum(lamp, ap_star=ap_star, gain=gain, n_jobs=1, ron=ron)["spec_sum"]
        # remove baseline
        # fear1d -= np.median(fear1d)
        
        """ corr2d to get initial estimate of wavelength """
        #from wave_calib import wavecalibrate_longslit
        from pyexspec.wavecalibrate import wclongslit
        from astropy.table import vstack
        if wavecalibrate:
            wave_calibrate = wclongslit.longslit(wave_template =None, flux_template=None, linelist =  self._lamp["linelist"])
            xshifts = []
            wave_init = []
            wave_solu = []
            pf1 = []
            rms = []
            tab_lines = []
            mpflux = []
            if QA_fig_path is None:
               QA_fig_path = os.path.join(self._wd, 'QA', 'wave_calibrate')
            if show and (not os.path.exists(QA_fig_path)):
               os.makedirs(QA_fig_path)
            suffix = '' if suffix is None else suffix
            for _i, _flux in enumerate(lamp1d):
                _flux = _flux[::-1]
                xcoord = np.arange(len(_flux))
                wave_template = self._lamp["wave"][_i][::-1]
                flux_template = self._lamp["flux"][_i][::-1]
                x_template = np.arange(len(wave_template))
                #dic = {'lamp1d': lamp1d,
                #   'xcoord': xcoord,
                #   '_flux': _flux,
                #   'wave_template': wave_template,
                #   'flux_template': flux_template,
                #   'x_template':x_template,
                #   'linelist':self._lamp["linelist"]
                #  }
                #joblib.dump(dic, 'dic.z')
                _xshift = wave_calibrate.get_xshift(xcoord, _flux, x_template=x_template, flux_template = flux_template,show=show)
                print(f'  |-xshift = {_xshift}  pixel')
                _wave_init = wave_calibrate.estimate_wave_init(xcoord, xshift=_xshift, x_template=x_template, wave_template = wave_template, 
                                                            deg=deg, nsigma=num_sigclip, min_select=min_select_lines, verbose=False)
                _tab = wave_calibrate.find_lines(wave_init=_wave_init, flux=_flux, npix_chunk=8, ccf_kernel_width=2)
                _wave_solu = wave_calibrate.calibrate(xcoord, _tab, flux=_flux, deg=deg, num_sigclip=num_sigclip,
                                              line_peakflux=line_peakflux, line_type=line_type ,min_select_lines=min_select_lines, show=show)
                _tab_lines = wave_calibrate.tab_lines.copy()
                _tab_lines['order'] = _i
                xshifts.append(-_xshift)
                wave_init.append(_wave_init[::-1])
                wave_solu.append(_wave_solu[::-1])
                #pf1.append(wave_calibrate.pf1)
                rms.append(wave_calibrate.rms)
                tab_lines.append(_tab_lines)
                _mpflux = np.median(_tab_lines["line_peakflux"][_tab_lines['indselect']])
                mpflux.append(_mpflux)
                fig_init_solu, axs = plt.subplots(1, 1, figsize=(7, 4))
                #plt.subplots_adjust(hspace=0)
                plt.plot(wave_template, flux_template/np.median(flux_template),lw=1, color='b', label='template')
                plt.plot(_wave_init, _flux/np.median( _flux), lw=1, color='r', label='init')
                plt.plot(_wave_solu, _flux/np.median( _flux), lw=1, label='solution', color='k')
                plt.legend()
                plt.xlabel(r'Wavelength ${\rm \AA}$')
                plt.ylabel(r'Counts')
                ####------------ save figure
                fname_fig_QA_ccf= os.path.join(QA_fig_path, f'{lampbasename}_QA_ccf{_i:03d}{suffix}.pdf')
                wave_calibrate.fig_QA_ccf.savefig(fname_fig_QA_ccf)
                ##--------------
                fname_fig_wave_calibrate= os.path.join(QA_fig_path, f'{lampbasename}_wave_calibrate{_i:03d}{suffix}.pdf')
                wave_calibrate.fig_QA_wave_calibrate.savefig(fname_fig_wave_calibrate)
                ##-------------
                fname_fig_init_solu= os.path.join(QA_fig_path, f'{lampbasename}_init_solu{_i:03d}{suffix}.pdf')
                fig_init_solu.savefig(fname_fig_init_solu)
            _tab = vstack(tab_lines)
            nlines = np.sum(_tab['indselect'])
        else:
            pf1 = None; rms =np.nan; mpflux=None; nlines=None
            wave_solu = np.nan; wave_init = np.nan; tab_lines= None

        header = self._modify_header(fp)
        isot =  f'{header["UTSHUT"]}'
        jd = Time(isot, format='isot').jd
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
            wave_init=wave_init,
            wave_solu=wave_solu,
            tab_lines=tab_lines,
            nlines=nlines,
            rms=rms,
            #pf1=pf1,
            deg=(deg, 'The degree of the 1D polynomial'),
            meidan_peak_flux=mpflux,
            # lamp=lamp,
            flux_arc=lamp1d,
            blaze = self.blaze,
        )
        return calibration_dict


    def find_arc_for_star(self, ra_star, dec_star, jd_star, radius=5):
        '''
        find the arc file name from the datatable
        parameters:
        ra_star: in unit of  degree, e.g. 10*units.deg
        dec_star: in unit of  degree, e.g. 20*units.deg
        jd_star: the jd of shut open
        radius [float] in unit of arcsec
        returns:
        ------------------
        filename [str]
        '''
        from astropy.coordinates import angular_separation
        from astropy.time import Time
        tab_arc = self.datatable_arc.copy()
        c = SkyCoord(tab_arc['ra'], tab_arc['dec'], unit=(units.hourangle, units.deg))
        angdis = angular_separation(ra_star, dec_star, c.ra, c.dec)
        angdis = angdis.to('deg').value
        _ind = angdis <= (radius/3600)
        if np.any(_ind):
           tab_arc = tab_arc[_ind]
        jds = Time(tab_arc['UTSHUT'], format='isot')
        delta_jd = np.abs(jds.jd - jd_star)
        ind = np.argmin(delta_jd)
        if isinstance(ind, np.int64):
           filename = tab_arc[ind]['filename']
        else:
           print(f'the number of arc for the star = {np.sum(ind)}')
           filename = None
        return filename

    def cal_image_error_square(self, image, bias_err_squared, gain, readnoise):
        '''
        calculate error**2 of each pixel of image
        paramters:
        -------------------------
        image [2D array]
        bias_err_squared [float], np.std(master_biase)**2
        gain [float]  in unit of e-/ADU
        readnoise [float]
        returns:
        image_err_squared
        '''
        image_err_squared = image*gain
        _ind = image_err_squared < 0
        image_err_squared[_ind] = 0
        image_err_squared = (image_err_squared + readnoise**2)/gain**2+bias_err_squared
        return image_err_squared

    def read_star(self, fp_star, master_bias=None, master_bias_err_squared=None, gain=None, readnoise=None, remove_cosmic_ray=False):
        '''
        fp_star [str] file name including full path
        gain [float] in units of e-/ADU
        readnoise [float] system noise in units of e-
        '''
        master_bias = self.master_bias if master_bias is None else master_bias
        hdu = fits.open(fp_star)
        ihdu = self.ihdu
        data = self._read_img(fp_star)
        #data = hdu[ihdu].data
        #xs, xe = self.strimx
        #ys, ye = self.strimy
        #data = data[ys:ye, xs:xe]
        if gain is None: gain = hdu[ihdu].header['gain']
        if readnoise is None: readnoise = hdu[ihdu].header['RON']
        hdu.close()
        #return np.rot90(data - self.master_bias)
        image = data- master_bias
        bias_err_squared = np.nanvar(master_bias)
        bias_err_squared = self.master_bias_err_squared if  master_bias_err_squared is None else master_bias_err_squared
        image_err_squared = self.cal_image_error_square(image, bias_err_squared, gain, readnoise).copy()
        self.image = image
        self.image_err_squared = image_err_squared
        if remove_cosmic_ray:
           pass
        return image

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
    mainprocess = UiBfosc()
    # ui.setupUi(mainWindow)
    # ui.initUi(mainWindow)
    mainprocess.show()
    try:
        sys.exit(app.exec_())# pyqt5
        #sys.exit(app.exec())# pyqt5
    except:
        sys.exit(app.exec())# pyqt6
