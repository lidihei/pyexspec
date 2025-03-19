import sys, os, glob
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    pyqt5_bool = True
except:
    from PyQt6 import QtCore, QtGui, QtWidgets
    pyqt5_bool = False
from pyexspec.gui import ExtractWindow
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
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units
from pyexspec.io import iospec2d
from pyexspec.aperture import sort_apertures
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Qt5Agg')
matplotlib.rcParams["font.size"] = 5


class UiExtract(QtWidgets.QMainWindow, ExtractWindow, iospec2d):

    def __init__(self, parent=None, instrument=None,
                 rot90=False, trimx = [830, 2148], trimy = [1368, 4000],
                 nwv=None,  wavecalibrate=True, site=None, Napw=2, Napw_bg=2, find_arc_for_spec=False):
        '''
        rot90 [bool] if True rotate the image with 90 degree
        trimx: list, or tuple]: Trim image on x axis, e.g. trimx = [0, 100]
        trimy: [list, or tuple]: Trim image on y axis, e.g. trimx = [0, 100]
        Napw: float
           N times ap_width away from the aperture center which is the right (and left) edge of the background
        Napw_bg: float
           N times ap_width area used to fitting background, |Napw_bg*ap_width-|Napw*ap_width -|center|+ Napw*ap_width| + Napw_bg*ap_width|
        site: the site of observatory
              if None: site = EarthLocation.of_site('Palomar') ## Palomar Observotory
              e.g.
              from astropy import coordinates as EarthLocation
              site = EarthLocation.from_geodetic(lat=26.6951*u.deg, lon=100.03*u.deg, height=3200*u.m) Lijiang observation
        find_arc_for_spec: [bool]
              if True, finding the arc file for each of exposure spectrum.
        '''
        #super(UiExtract, self).__init__(parent)
        # data
        #print('The index of hdulist: ihdu = (must be integer e.g. 1)')
        QtWidgets.QMainWindow.__init__(self, parent)
        ExtractWindow.__init__(self, instrument_name=instrument)
        iospec2d.__init__(self, rot90=rot90, trimx=trimx, trimy=trimy)
        ihdu = 1
        ihdu = int(ihdu)
        self.ihdu0 = ihdu
        self.ihdu = ihdu
        self.trimx0 = trimx
        self.trimy0 = trimy
        self._wd = ""
        self.datatable = None
        self.pos = []
        self.pos_temp = [0, 0]
        self.master_bias = None
        self.master_flat = None
        self.trace_handle = []
        self.trimdy = self.trimy[1] - self.trimy[0]
        self.trimdx = self.trimx[1] - self.trimx[0]
        self.rot90 = rot90
        self.Napw = Napw
        self.Napw_bg = Napw_bg
        self.find_arc_for_spec = find_arc_for_spec
        # the number of pixel of in the direction of dispersion (the number of pixel of wavelength)
        if nwv is None: self.nwv = self.trimdy
        self.ap_trace = np.zeros((0, self.nwv), dtype=int)
        self._lamp_template = None
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
        test_dir = "/Users/lijiao/Documents/works/sdb_BHB_240cm_observation/data/reduce_data/20241125/E9G10"
        self._wd = test_dir
        self.lineEdit_wd.setText(test_dir)
        lamp_template = "../template/feartemplate_order0to10_from202410280096.z"
        self._lamp_template = joblib.load(lamp_template)
        self.lineEdit_lamp.setText(lamp_template)
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
        self._lamp_template = joblib.load(fileName)
        print("The Template Spectrum of Arc Lamp is loaded!")

    def _set_wd(self):
        self._wd = self.lineEdit_wd.text()

    def _make_datatable(self):
        # get file list
        fps_full = glob.glob(self.lineEdit_wd.text() + "/*.fit")
        if len(fps_full) == 0:
            fps_full = glob.glob(self.lineEdit_wd.text() + "/*.fits.fz")
        if len(fps_full) == 0:
            fps_full = glob.glob(self.lineEdit_wd.text() + "/*.fits") 
        fps_full.sort()
        self.fps_full = fps_full
        fps = [os.path.basename(_) for _ in fps_full]
        self.nfp = len(fps)
        headers = [fits.getheader(fp, 1) if '.fz' in fp else fits.getheader(fp) for fp in fps_full]

        imgtype = np.asarray([hd["OBSTYPE"] for hd in headers])
        imgtype1 = np.asarray([hd["FILTER7"] for hd in headers])
        exptime = np.asarray([hd["EXPTIME"] for hd in headers])
        try:
            UTSHUT = np.asarray([hd["UTSHUT"] for hd in headers])
        except:
            UTSHUT = np.asarray([hd["DATE-OBS"] for hd in headers])
        ra = [] #np.zeros(len(fps_full))
        dec = [] #np.zeros(len(fps_full))
        for i, fp in enumerate( fps_full):
            #print(fp)
            header = headers[i]
            try: ra.append(header["RA"])
            except: ra.append(np.nan)
            try: dec.append(header["DEC"])
            except: dec.append(np.nan)
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
            elif "lamp_halogen" in imgtype1[i].lower() or "flat" in fps_full[i].lower():
                types[i] = "flat"
            elif ("arcs" in imgtype1[i].lower()) or ('lamp_fe_argon' in imgtype1[i].lower())\
                or ('lamp_neon_helium' in imgtype1[i].lower()): #and (exptime[i] == 300 or "arcs" in fps_full[i].lower()):
                types[i] = "arc"
            #elif "light" in imgtype[i].lower() and (exptime[i] != 300 or "target" in fps_full[i].lower()):
            #    types[i] = "star"
            else:
                types[i] = "star"

        self.datatable = table.Table(
            data=[fps, imgtype, exptime, types, ra, dec, UTSHUT],
            names=["filename", "imagetype", "exptime", "type", 'ra', 'dec', 'UTSHUT'])
        # print(self.datatable["type"])

    def _update_datatable(self):
        # print(self.datatable["type"])
        self.datatable["type"] = [self.type_list[self.tableWidget_files.cellWidget(irow, 3).currentIndex()] for irow in range(self.nfp)]
        self._refresh_datatable()
        #self.datatable.write(self._wd+"/catalog.csv", overwrite=True)
        dire_table = os.path.join(self._wd, 'TABLE')
        if not os.path.exists(dire_table): os.makedirs(dire_table)
        self.datatable.write(os.path.join(dire_table, 'filelist_table.csv'), overwrite=True)
        self.datatable_arc = self.datatable[self.datatable['type']=='arc']

    def _get_file_list(self):
        self._make_datatable()
        self._refresh_datatable()

    def _refresh_datatable(self):
        if self.datatable is None:
            return
        # change to Table Widget
        self.tableWidget_files.clear()
        self.tableWidget_files.verticalHeader().setVisible(True)
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
        print("  |-Show file {}: {}".format(ind_elected+1, fp_selected))
        # try to draw it
        try:
            #img = fits.getdata(fp_selected)
            img = self._read_img(fp_selected)
        except IsADirectoryError:
            print("Not sure about what you are doing ...")
            return
        self._draw_img(img, canvas=True)

    def _draw_img(self, img, canvas=True):
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
        if canvas:
            self.canvas.mpl_connect('button_press_event', self.onclick)
            self.canvas.draw()

    def _draw_img_show(self, img):
        # draw
        fig, ax = plt.subplots()
        self.ax = ax
        self.ax.imshow(img, cmap=plt.cm.jet, origin="lower", vmin=np.percentile(img, 5), vmax=np.percentile(img, 90),
                       aspect="auto")
        self.pos_handle, = self.ax.plot([], [], "+", ms=10, color="tab:cyan", mew=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        #self.ax.plot()

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
        ### trim
        def get_trim(lineEdit_trim_start, lineEdit_trim_end, trim):
            try: trim_start = int(lineEdit_trim_start.text())
            except: trim_start = trim[0]
            try: trim_end = int(lineEdit_trim_end.text())
            except: trim_end = trim[1]
            trim= [trim_start, trim_end]
            return trim
        self.trimx = get_trim(self.lineEdit_trimx_start, self.lineEdit_trimx_end, self.trimx0)
        self.trimy = get_trim(self.lineEdit_trimy_start, self.lineEdit_trimy_end, self.trimy0)
        self.trimdx = self.trimx[1] - self.trimx[0]
        self.trimdy = self.trimy[1] - self.trimy[0]
        # the number of pixel of in the direction of dispersion (the number of pixel of wavelength)
        self.nwv = self.trimdy
        ### hdu
        try: self.ihdu = int(self.lineEdit_hdu.text())
        except: self.ihdu = self.ihdu0
        ### rot90 image
        if pyqt5_bool:
            checkBox_rot90_state = self.checkBox_rot90.checkState()
        else:
            checkBox_rot90_state = self.checkBox_rot90.checkState().value
        if not self.rot90:
            self.rot90 = False if checkBox_rot90_state == 0 else True
        print(f'checkBox_rot90.checkState() = {self.checkBox_rot90.checkState()}')
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
            readnoise = fits.getheader(fp, self.ihdu)['RDNOISE']
            flats.append(self.read_star(fp, remove_cosmic_ray=False, readnoise=readnoise))
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

    def _draw_aperture(self, canvas=True):
        if len(self.ax.lines) > 1:
            for line in self.ax.get_lines()[1:]:  # ax.lines:
                line.remove()
        for _trace in self.ap_trace:
            ind_plot = _trace > 0
            #self.ax.plot(np.arange(self.trimdx)[ind_plot], _trace[ind_plot], "w-", lw=1)
            self.ax.plot(_trace[ind_plot], np.arange(self.nwv)[ind_plot], "w-", lw=1) # lijiao
            try:
                ap_width = self.ap.ap_width
                self.ax.plot(_trace[ind_plot]+ap_width, np.arange(self.nwv)[ind_plot], "w-.", lw=1)
                self.ax.plot(_trace[ind_plot]-ap_width, np.arange(self.nwv)[ind_plot], "w-.", lw=1)
            except:
                print('Appeture width is not difined')
            #try:
            #    ap_width = self.ap.ap_width
            #    _Napw = self.ap.Napw*ap_width; _Napw_bg = self.ap.Napw_bg*ap_width
            #    self.ax.plot(_trace[ind_plot]+_Napw, np.arange(self.nwv)[ind_plot], "k--", lw=0.8)
            #    self.ax.plot(_trace[ind_plot]+_Napw+_Napw_bg, np.arange(self.nwv)[ind_plot], "k--", lw=0.8)
            #    self.ax.plot(_trace[ind_plot]-_Napw, np.arange(self.nwv)[ind_plot], "k--", lw=0.8)
            #    self.ax.plot(_trace[ind_plot]-_Napw-_Napw_bg, np.arange(self.nwv)[ind_plot], "k--", lw=0.8)
            #except:
            #    print('Background Appeture width is not difined')
        if canvas:
            self.canvas.draw()

    def _choose_aperture_image(self, use_master_flat=False):
        '''
        choose proper image to trace the aperture
        '''
        print('#----Print the fits (or fit) name for tracing aperture, e.g. blue0059.fits')
        print('#----should be star fits for HIRES of None ')
        ind_elected = self.tableWidget_files.currentRow()
        fname = self.fps_full[ind_elected]
        print(fname)
        #fname = input()
        file_extension = os.path.splitext(fname)[-1]
        self.aperture_image = self.master_flat
        if (('.fit' in file_extension) or ('.fz' in file_extension)) and ( not use_master_flat):
            #fname = os.path.basename(fname)
            #fname = os.path.join(self._wd, fname)
            self.fname_aperture = fname
            readnoise = fits.getheader(fname, self.ihdu)['RDNOISE']
            image_aperture = self.read_star(fname, readnoise = readnoise)
            self.aperture_image = image_aperture
            print(f'#----Finding aperture from {fname}')
        else:
            print(f'#----Finding aperture from master flat')
        aperture_image_smooth = gaussian_filter(self.aperture_image,sigma=4)
        self._aperture_image_uncertainty = np.max(
            np.diff(np.percentile(self.aperture_image - aperture_image_smooth, [16, 50, 84]))
                                            )

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
            gaussian_filter(self.aperture_image, sigma=1.5),
            *np.asarray(self.pos_temp[::-1], dtype=int), maxdev=7, fov=10, ntol=5)
        self._trace = _trace
        print(_trace)
        if np.sum(_trace>0)>100:
            self.ap_trace = np.vstack((self.ap_trace, _trace.reshape(1, -1)))
            self._draw_aperture()
            self._update_nap()
            print("An aperture is added")
        else:
            print(f"An aperture is not added as sum(_trace>0) = {np.sum(_trace>0)} is less than threshold")
        #except Exception as _e:
        #    print("Error occurred, aperture not added!")

    def _del_aperture(self):
        if self.ap_trace.shape[0] == 0:
            pass
        dx = np.arange(self.trimdx) - self.pos_temp[0]#lijiao
        dy = np.arange(self.trimdy) - self.pos_temp[1] #lijiao
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
        deg =2
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
        self.ap.polyfitnew(deg = deg, ystart=400)
        # replace old traces
        #self.ap_trace = self.ap.ap_center_interp[:, ::-1]
        self.ap_trace = self.ap.ap_center_interp # lijiao
        # fit again
        #self.ap = Aperture(ap_center=self.ap_trace[:, ::-1], ap_width=10)
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=ap_width)
        #self.ap.get_image_info(np.rot90(self.master_flat)) # lijiao
        self.ap.get_image_info(self.master_flat) # lijiao
        self.ap.polyfitnew(deg=deg, ystart=400)
        try:self.Napw = int(self.lineEdit_Napw.text())
        except: pass
        try: self.Napw_bg = int(self.lineEdit_Napw_bg.text())
        except: pass
        try: self.ap_width = int(self.lineEdit_ap_width.text())
        except: self.ap_width =ap_width
        self.ap.ap_width = self.ap_width
        self.ap.Napw = self.Napw
        self.ap.Napw_bg = self.Napw_bg
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
        from pyexspec.fitfunc.function import gaussian_linear_func
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
            starting_row = int(len(ap_center)/3)
            starting_col = int(ap_center[starting_row])
            _trace = self.trace_local_max(
                gaussian_filter(image_star, sigma=1.2),
                starting_row, starting_col, maxdev=7, fov=10, ntol=10)
            self._trace = _trace
            if np.sum(_trace>0)>100:
                self.ap_trace[i] = _trace
        self.ap_trace = self.ap_trace[sort_apertures(self.ap_trace)]
        self.ap = Aperture(ap_center=self.ap_trace, ap_width=12)
        self.ap.get_image_info(image_star)
        self.ap.polyfitnew(deg = 2, ystart=400)
        self.ap_trace = self.ap.ap_center_interp
        if ap_width is None:
            print(f'  |-estimate ap_width using Gaussian function, ap_width = {Nsigma}sigma')
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

    def _modify_header(self, fname, ihdu=0):
        header = fits.getheader(fname, ext=ihdu)
        header['TRIMSEC'] = ('[{}:{}, {}:{}]'.format(*self.trimx, *self.trimy), 'PYEXSPEC: Section of useful data')
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
        return header[:]

    def remove_cosmic_ray(self, image, readnosize=2, sigclip=7, verbose=True):
        '''
        image [2D array]:
        '''
        ccd = CCDData(image, unit='adu')
        new_ccd = ccdp.cosmicray_lacosmic(ccd, readnoise=readnoise, sigclip=sigclip, verbose=verbose)
        return new_ccd


    def _extract_single_star(self, fp, master_flat=None, master_flat_err_squared=None,
                             adjust_aperture=True, irow=1000, show=False,
                             site=None, kind='barycentric', find_arc_for_spec=False):
        '''
        site: the site of observatory
              e.g.
              from astropy import coordinates as EarthLocation
              site = EarthLocation.from_geodetic(lat=26.6951*u.deg, lon=100.03*u.deg, height=3200*u.m) Lijiang observation
              site = EarthLocation.of_site('Palomar') ## Palomar Observotory
        '''
        from twodspec.extract import extract_all
        _dir = os.path.dirname(fp)
        basename = os.path.basename(fp)
        rowID = np.where(self.datatable["filename"]== basename)[0][0]
        self.tableWidget_files.selectRow(rowID)
        dirdump = os.path.join(_dir, f'dump')
        self.dirdump = dirdump
        if not os.path.exists(dirdump): os.makedirs(dirdump)
        fp_out = "{}/star-{}.dump".format(dirdump, os.path.basename(fp))
        readnoise = fits.getheader(fp, self.ihdu)['RDNOISE']
        star = self.read_star(fp, readnoise=readnoise).copy()
        #self.tableWidget_files.itemSelectionChanged.connect(self._show_img)
        self._show_img()
        if adjust_aperture:
           self._adjust_aperture_star(star, ap_width = None, Nsigma=3.5)
        try: self.ap.ap_width = int(self.lineEdit_ap_width.text())
        except: pass
        self.blaze, self.sensitivity = self.ap.make_normflat(self.master_flat)
        ####------------ fit back ground --------------------
        #ap_star = self.ap
        star_err_squared = self.image_err_squared
        bg = self.ap.backgroundnew(star, longslit = False,q=(40, 40), npix_inter=7, sigma=(20, 20), kernel_size=(21, 21),
                                   Napw_bg=self.Napw_bg, num_sigclip=5, Napw=self.Napw, verbose=False)
        self._draw_aperture()
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
        header = self._modify_header(fp, ihdu=self.ihdu)
        try:
            isot =  f'{header["UTSHUT"]}'
        except:isot =  f'{header["DATE-OBS"]}'
        time_shut = Time(isot, format='isot', location=site)
        jd = time_shut.jd
        if show:
           fig, ax = plt.subplots(1,1)
           plt.plot(star[irow])
           plt.plot(bg[irow])
        gain = 1/np.float64(header['GAIN'])
        try:ron = header['RON']
        except: ron = header['RDNOISE']
        ron = np.float64(ron)
        #star1d = self.ap.extract_all(star, gain=gain, ron=ron, n_jobs=1, verbose=False)
        ##### extract with background
        n_jobs =1
        star1d_withbg = extract_all(star_withbg/self.sensitivity, self.ap, im_err_squared = star_withbg_err_squared, n_chunks=8,
                ap_width=self.ap.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron, n_jobs=n_jobs)
        star1d_withbg_divide_flat = extract_all(star_withbg_divide_flat, self.ap,
                im_err_squared = star_withbg_divide_flat_err_squared, n_chunks=8,
                ap_width=self.ap.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron, n_jobs=n_jobs)
        ##### extract image - background
        star1d = extract_all(star/self.sensitivity, self.ap, im_err_squared = star_err_squared, n_chunks=8,
                ap_width=self.ap.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron, n_jobs=n_jobs)
        star1d_divide_flat = extract_all(star_divide_flat, self.ap,
                im_err_squared = star_divide_flat_err_squared, n_chunks=8,
                ap_width=self.ap.ap_width, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=5., gain=gain, ron=ron, n_jobs=n_jobs)
        #star1d = self._extract_sum(star, self.ap.ap_center, im_err_sqaured=star_err_squared, ap_width=self.ap.ap_width)
        #star1d_divide_flat = self._extract_sum(star_divide_flat, self.ap.ap_center, 
        #                     im_err_sqaured=star_divide_flat_err_squared, ap_width=self.ap.ap_width)
        c_star = SkyCoord(header['ra'], header['dec'], unit=(units.hourangle, units.deg))
        #######--------find arc file and extract-------------------------------------
        if find_arc_for_spec:
            filename_arc = self.find_arc_for_star(c_star.ra, c_star.dec, jd, radius=5)
        else:
            filename_arc = None
        if filename_arc is not None:
           print(f"  |- processing arc {os.path.basename(filename_arc)} for sience {os.path.basename(fp)}")
           fp_arc = os.path.join(self._wd, filename_arc)
           if not os.path.exists(dirdump): os.makedirs(dirdump)
           fp_out_arc = f"{dirdump}/lamp-{filename_arc}_for_{os.path.basename(fp)}.dump"
           arcdic = self._proc_lamp(fp_arc, self.ap, num_sigclip=3, verbose=False,
                                 suffix=f'_science_{os.path.basename(fp)}',
                                  wavecalibrate=True, deg=4, show=True)
           if arcdic is not None:
               print("  |- writing to {}".format(fp_out_arc))
               joblib.dump(arcdic, fp_out_arc)
           arcfile = os.path.basename(filename_arc)
        else:
           arcdic = dict(wave_init = np.nan, wave_solu = np.nan, rms=np.nan, flux_arc=np.nan)
           arcfile = None
        star1d["blaze"] = self.blaze
        star1d["UTC-OBS"] = isot
        star1d["jd"] = jd
        star1d["EXPTIME"] = header["EXPTIME"]
        star1d['STRIM'] = ('[{}:{}, {}:{}]'.format(*self.trimx, *self.trimy), 'PYEXSPEC data reduce')
        star1d['filename_arc'] = filename_arc
        star1d['wave_init'] = arcdic['wave_init']
        star1d['wave_solu'] = arcdic['wave_solu']
        star1d['rms_wave_calibrate'] =arcdic['rms']
        star1d['flux_arc'] =arcdic['flux_arc']
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
        if self._lamp_template is None:
            print("  |-- LAMP teamplate is not loaded!")
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
        if len(fps_star) > 0:
            if show: fig, ax = plt.subplots(1,1)
            for i_star, fp in enumerate(fps_star):
                print("|--({}/{}) processing STAR ... ".format(i_star+1, n_star), end="")
                self._extract_single_star(fp, master_flat=master_flat, adjust_aperture=False,
                                    master_flat_err_squared=master_flat_err_squared, site=self.site, find_arc_for_spec=self.find_arc_for_spec)
        if self.find_arc_for_spec is False:
            self.blaze, self.sensitivity = self.ap.make_normflat(self.master_flat)
            print("""[5.1] extracting lamp1d (~5s/lamp) """)
            fps_lamp = self._gather_files("arc")
            n_lamp = len(fps_lamp)
            for i_lamp, fp in enumerate(fps_lamp):
                print("|--({}/{}) processing lamp ... ".format(i_lamp+1, n_lamp))
                _dir = os.path.dirname(fp)
                filename_arc = os.path.basename(fp)
                dirdump = os.path.join(_dir, f'dump')
                if filename_arc is not None:
                   print(f"  |-processing arc {os.path.basename(filename_arc)}")
                   fp_arc = os.path.join(self._wd, filename_arc)
                   if not os.path.exists(dirdump): os.makedirs(dirdump)
                   fp_out_arc = os.path.join(dirdump, f"lamp-{filename_arc}.z")
                   arcdic = self._proc_lamp(fp_arc, self.ap, num_sigclip=3, verbose=False,
                                         suffix=None,
                                          wavecalibrate=True, deg=4, show=False)
                   self.arcdic = arcdic
                   if arcdic is not None:
                       print("  |- writing to {}".format(fp_out_arc))
                       joblib.dump(arcdic, fp_out_arc)
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

    def _proc_lamp(self, fp, ap_star, num_sigclip=2.5,
                  line_peakflux=50,line_type = 'line_x_ccf',
                  verbose=False, wavecalibrate=True, deg=4, show=False,
                  degxy = (4, 6),
                  min_select_lines = 10,
                  QA_fig_path = None, suffix=None):
        """ read lamp
        parameters:
        --------------------
        suffix [str]  the suffix of file name of QA for calibrating wave
        """
        readnoise = fits.getheader(fp, self.ihdu)['RDNOISE']
        lamp = self.read_star(fp, readnoise=readnoise)
        lampbasename = os.path.basename(fp)
        lamp /= self.sensitivity
        header = fits.getheader(fp, self.ihdu)
        gain = 1/np.float64(header['GAIN'])
        try:ron = header['RON']
        except: ron = header['RDNOISE']
        ron = np.float64(ron)
        # unnecessary to remove background
        # fear -= apbackground(fear, ap_interp, q=(10, 10), npix_inter=5,sigma=(20, 20),kernel_size=(21,21))
        # extract 1d fear
        lamp1d = self.extract_sum(lamp, ap_star=ap_star, gain=gain, n_jobs=1, ron=ron)["spec_sum"]
        # remove baseline
        # fear1d -= np.median(fear1d)

        """ corr2d to get initial estimate of wavelength """
        #from wave_calib import wavecalibrate_longslit
        from astropy.table import vstack
        if pyqt5_bool:
            checkBox_autowvcalib_state = self.checkBox_autowvcalib.checkState()
        else:
            checkBox_autowvcalib_state = self.checkBox_autowvcalib.checkState().value
        wavecalibrate = False if checkBox_autowvcalib_state == 0 else True
        if wavecalibrate:
            from twodspec import thar
            from pyexspec.wavecalibrate import wclongslit
            from pyexspec.wavecalibrate import findlines
            wave_calibrate = wclongslit.longslit(wave_template =None, flux_template=None, linelist =  self._lamp_template["linelist"])
            xshifts = []
            waves_init = np.zeros_like(lamp1d)
            waves_solu = []
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
                xcoord = np.arange(len(_flux))
                wave_template = self._lamp_template["wave"][_i]
                flux_template = self._lamp_template["flux"][_i]
                x_template = np.arange(len(wave_template))
                _xshift = wave_calibrate.get_xshift(xcoord, _flux, x_template=x_template, flux_template = flux_template,show=show)
                if np.abs(_xshift) > 200:
                    xshift_old = _xshift
                    _flux = wave_calibrate.filter_spec_outlier(xcoord, _flux, window = 5, num_sigma_clip = 30)
                    _xshift = wave_calibrate.get_xshift(xcoord, _flux, x_template=x_template, flux_template = flux_template,show=show)
                    print(f"   |-@ccf with filtering outliers order = {_i}; ccf shift = {_xshift}, the outlier points have been removed when calculating CCF since original shift = {xshift_old} (> 100 pixels)")
                    if np.abs(_xshift) > 200:
                       cut = slice(200, -200)
                       _xshift = wave_calibrate.get_xshift(xcoord[cut], _flux[cut], x_template=x_template, flux_template = flux_template,show=show)
                       print(f"   |-@ccf with slice(200,-200) order = {_i}; ccf shift = {_xshift}, the outlier points have been removed when calculating CCF since original shift = {xshift_old} (> 100 pixels)")
                else:
                    print(f'    |--order {_i} xshift = {_xshift}  pixel')
                _wave_init = wave_calibrate.estimate_wave_init(xcoord, xshift=_xshift, x_template=x_template, wave_template = wave_template,
                                                          deg=deg, nsigma=num_sigclip, min_select=min_select_lines, verbose=False)
                waves_init[_i] = _wave_init
            _npix_chunk = 6
            tab_lines = findlines.find_lines(waves_init, lamp1d, self._lamp_template["linelist"], npix_chunk=_npix_chunk, ccf_kernel_width=1.5, num_sigma_clip=3)
            ind_good = np.isfinite(tab_lines["line_x_ccf"]) & (np.abs(tab_lines["line_x_ccf"] - tab_lines["line_x_gf"]) < _npix_chunk)
            tab_lines.add_column(table.Column(ind_good, "ind_good"))
            from pyexspec.fitfunc.polynomial import Poly1DFitter
            from pyexspec.wavecalibrate.findlines import grating_equation2D
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

            tlines = tab_lines.copy()
            print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
            clean(pw=1, deg=2, threshold=10, min_select=20)
            clean(pw=1, deg=2, threshold=5, min_select=20)
            clean(pw=1, deg=2, threshold=1, min_select=20)
            print("  |- {} lines left".format(np.sum(tlines["ind_good"])))
            tab_lines["ind_good"] = tlines["ind_good"].copy()
            tlines = tlines[tlines["ind_good"]]
            ### fitting grating equation
            x = tlines["line_x_ccf"]
            y = tlines["order"]
            z = tlines["line"]
            pf1, pf2, indselect = grating_equation2D(
                                  x, y, z, deg=degxy, nsigma=num_sigclip, min_select=100, verbose=10)
            tlines.add_column(table.Column(indselect, "indselect"))
            rms = pf2.rms
            nx, norder = lamp1d.shape
            mx, morder = np.meshgrid(np.arange(norder), np.arange(nx))
            wave_solu = pf2.predict(mx, morder)
            _tab = vstack(tab_lines)
            tab_lines = tlines
            nlines = np.sum(tlines['indselect'])
            wave_init = waves_init
        else:
            pf1 = None; rms =np.nan; nlines=None
            wave_solu = np.nan; wave_init = np.nan; tab_lines= None
        header = self._modify_header(fp, ihdu=self.ihdu)
        try:
            isot =  f'{header["UTSHUT"]}'
        except:isot =  f'{header["DATE-OBS"]}'
        jd = Time(isot, format='isot').jd
        if wavecalibrate is False:
           print("  |- !!! The wavelenth of this ARC LAMP is not calibrated")
        nx, norder = lamp1d.shape
        mx, morder = np.meshgrid(np.arange(norder), np.arange(nx))
        # result
        calibration_dict = OrderedDict(
            fp=fp,
            jd=jd,
            exptime=header['EXPTIME'],
            header = header,
            STRIM = '[{}:{}, {}:{}]'.format(*self.trimx, *self.trimy),
            wave_init=wave_init,
            wave_solu=wave_solu,
            tab_lines=tab_lines,
            nlines=nlines,
            rms=rms,
            #pf1=pf1,
            deg=(deg, 'The degree of the 1D polynomial'),
            degxy=(degxy, 'The degree of the 2D polynomial'),
            #meidan_peak_flux=mpflux,
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


    def contourf(self, img):
        rows, _cols = img.shape
        _xx, _yy = np.meshgrid(np.arange(_cols), np.arange(_rows))
        plt.contourf(_xx, _yy, _img)

    def writefits(self, header, data, fout):
        hdu = fits.HDUList([fits.PrimaryHDU(header=_header, data=_data)])
        hdu.writeto(fout)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # mainWindow = QtWidgets.QMainWindow()
    site = EarthLocation.from_geodetic(lat=26.6951*units.deg, lon=100.03*units.deg, height=3200*units.m)#lijiang
    mainprocess = UiExtract(instrument='E9G10', site=site)
    # ui.setupUi(mainWindow)
    # ui.initUi(mainWindow)
    mainprocess.show()
    if pyqt5_bool: 
        sys.exit(app.exec_())# pyqt5
    else:
        sys.exit(app.exec())# pyqt6
