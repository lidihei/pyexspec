import sys, os, glob
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except:
    from PyQt6 import QtCore, QtGui, QtWidgets
from pyexspec.gui import WvClibWindow
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from astropy.table import Table, vstack
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
import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Qt5Agg')
matplotlib.rcParams["font.size"] = 5


class UiWvcalib(QtWidgets.QMainWindow, WvClibWindow):

    def __init__(self, parent=None, wd=None):
        '''
        wd [str] the working directory
        '''
        super(UiWvcalib, self).__init__(parent)
        # data
        #print('The index of hdulist: ihdu = (must be integer e.g. 1)')
        QtWidgets.QMainWindow.__init__(self, parent)
        self.tablename_lines = None # the found emission lines table
        self.pos = []
        self.pos_line = [0, 0]
        self.master_bias = None
        self.master_flat = None
        self.trace_handle = []
        # the number of pixel of in the direction of dispersion (the number of pixel of wavelength)
        self._lamp = None
        self.aperture_image = None
        # UI
        self.setupUi(self)
        self.add_canvas()
        self.assumption() # debug
        self._get_tab_lin_order()
        self.initUi()
        self.order = 1
        # debug
        self._wd = os.path.dirname(self._arcfile) if wd is None else wd

    def add_canvas(self):
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(600, 10, 670, 670))
        self.widget2.setObjectName("widget")
        self.verticalLayout2 = QtWidgets.QVBoxLayout(self.widget2)
        #self.verticalLayout2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout2.setObjectName("verticalLayout")

        # a figure instance to plot on
        self.figure, self.axs = plt.subplots(3,1, sharex=True,
                    gridspec_kw={'height_ratios': [2, 2, 1]})
        plt.subplots_adjust(hspace=0, top=0.95, left=0.06, bottom=0.05, right=0.99 )
        #self.figure, self.ax = plt.subplots(1,1)

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
        test_dir = "test"
        self._wd = test_dir
        self._arcfile = os.path.join(self._wd, "lamp-blue0062.fits_for_blue0060.fits.dump")
        self._linelistfile = '../arc_linelist/FeAr.dat'
        self.lineEdit_arc.setText(self._arcfile)
        self.lineEdit_linelist.setText(self._linelistfile)
        self._read_linelist(self._linelistfile)
        self._read_arc(self._arcfile, tablename=self._arcfile)
        #self._arcfile = joblib.load(os.path.join(self._wd, "lamp-blue0062.fits_for_blue0060.fits.dump"))

    def initUi(self):
        self.toolButton_load_arc.clicked.connect(self._select_arc)
        self.toolButton_load_linelist.clicked.connect(self._select_linelist)
        self.lineEdit_arc.textChanged.connect(self._get_tab_lin_order)
        self.tableWidget_files.itemSelectionChanged.connect(self._draw_scatter_pos)
        self.pushButton_upper_order.clicked.connect(self._upper_order)
        self.pushButton_next_order.clicked.connect(self._next_order)
        self.pushButton_update_table.clicked.connect(self._update_datatable_button)
        self.pushButton_clear_line.clicked.connect(self._clear_line)
        self.pushButton_add_line.clicked.connect(self._add_line)
        self.pushButton_del_line.clicked.connect(self._del_line)
        self.pushButton_save_line.clicked.connect(self._save_lines)
        self.pushButton_fit.clicked.connect(self._fit)
        self.lineEdit_xdeg.setText(f"deg(x) = ")
        self.lineEdit_ydeg.setText(f"deg(y) = ")
        # self.listWidget_files.currentItemChanged.connect(self._scatter_point)

    def _select_linelist(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open arc", "dump files (*.dump)")
        self.lineEdit_linelist.setText(fileName)
        self._read_linelist(fileName)
        print("Linelist set to ", os.path.basename(fileName))

    def _read_linelist(self, fileName):
        tab = Table.read(fileName, format='ascii')
        self.tab_linelist = tab
        self.linelist = tab['wavelength'] 

    def _select_arc(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Open arc", "dump files (*.dump)")
        self.lineEdit_arc.setText(fileName)
        self._arcfile = fileName
        self._read_arc(fileName)
        print(f"arc loaded: {os.path.basename(fileName)}")

    def _read_arc(self, fileName, tablename=None):
        '''
        parameters:
        --------------------
        fileName [str]: the filename of arc dump file
        table [str]: file name of the found emission lines table
                  that must include 'order', 'xcoord', 'line', 'Fpeak','good',  'wv_fit'
        '''
        self.arcdic = joblib.load(fileName)
        flux = self.arcdic['flux_arc']
        shape  = flux.shape
        if len(shape) == 1:
            Norder, Npixel = 1,shape[0]
            flux = np.array([flux]).T
        else:
            Norder, Npixel = shape
        if tablename is None: tablename = self.tablename_lines
        self.fluxes = flux
        self.Norder = Norder
        self.tables = self._read_linetable(Norder=Norder, tablename=tablename)
        self.Npixel = Npixel
        self.Xgrid, self.Ygrid = np.meshgrid(np.arange(Norder), np.arange(Npixel))
        self.xindex = np.arange(Npixel)
        self.order = 1
        self._update_order()
        self.tab_lin_order = self.tables[0]
        self._draw_spec(tab=None)
        self._draw_residual()
        self._draw_wvfit()

    def _add_tab_oneline(self, colnames=None):
        if colnames is None:
           colnames = ['order', 'xcoord', 'line', 'maskgood', 'wv_fit', 'Fpeak']
        tab = Table(data=np.zeros(len(colnames)).T, names=colnames)
        tab['order'] = self.order
        tab['maskgood'] = 1
        return tab

    def _read_linetable(self, Norder=None, tablename=None, table=None):
        '''
        parameters:
        --------------------
        Norder [int]: the number of spectra order
        tablename [str] file name of the found emission lines table
                  that must include 'order', 'xcoord', 'line', 'maskgood', 'wv_fit', 'Fpeak'
        returns:
        tables [list]: a list of astropy.Table of emission lines of all order spectra
               len(tables) = Norder
        '''
        if Norder is None: Norder = self.Norder
        orders = np.arange(Norder)+1
        if (tablename is None) and (table is None):
            names = ['order', 'xcoord', 'line', 'maskgood', 'wv_fit', 'Fpeak']
            tables = [Table(data= [[i],[0.], [0.], [0], [0.], [0.]], names=names) for i in orders]
        else:
            if table is None:
                filetype = tablename.split('.')[-1]
                if (filetype == 'dump') or (filetype =='z'):
                    try:
                        tab = joblib.load(tablename)['tlines']
                        if isinstance(tab, list): tab = vstack(tab)
                    except:
                        tab = joblib.load(tablename)['tab_lines']
                        if isinstance(tab, list): tab = vstack(tab)
                    tab['order'] = tab['order'] + 1
                    tab.rename_columns(['line_x_ccf', 'line_wave_init_ccf', 'line_peakflux', 'ind_good', ],
                                       ['xcoord', 'wv_fit', 'Fpeak', 'maskgood'])
                else:
                    tab = Table.read(tabname)
            else:
                tab = tabel
            tab = tab[np.array(tab['maskgood'], dtype=bool)]
            colnames = tab.colnames
            def change_columorder(colnames,_i, name):
                colnames1 = colnames.copy()
                ind = np.where(np.array(colnames)==name)[0][0]
                name_orig = colnames1[_i]
                colnames[_i] = name; colnames[ind] = name_orig
                return colnames
            for _i, _name in enumerate(['order', 'xcoord', 'line', 'maskgood', 'wv_fit', 'Fpeak' ]):
                colnames = change_columorder(colnames, _i, _name)
            tab = tab[colnames]
            tables = self._split_tab_lin_all(tab_lin_all=tab) 
        self.tab_lin_all = vstack(tables)
        self.colnames = tables[0].colnames
        return tables

    def _split_tab_lin_all(self, tab_lin_all=None):
        tab = self.tab_lin_all if tab_lin_all is None else tab_lin_all
        orders = np.arange(self.Norder)+1
        tables = [tab[tab['order'] == i] for i in orders]
        self.tables = tables
        return tables

    def _set_wd(self):
        self._arcfile = self.lineEdit_arc.text()

    def _make_datatable(self):
        iorder = self.order -1
        tab_lin_order =  self.tables[iorder]
        #datatable.sort('xcoord')
        self.tab_lin_order = tab_lin_order
        self.color_list = [[255, 255, 255],
                           [211, 211, 211]]
        #                   [255, 182, 193],
        #                   [255, 228, 181],
        #                   [173, 216, 230], ]


    def _find_line(self, line_tmp, linelist=None):
        if linelist is None:
           linelist = self.linelist
        if (line_tmp >= np.min(linelist)) & (line_tmp <= np.max(linelist)):
            dline = np.abs(linelist-line_tmp)
            line = linelist[np.argmin(dline)]
        else:
            line = line_tmp
        return line

    def _update_datatable_button(self):
        self._update_datatable(addlinebool=True)

    def _update_datatable(self, addlinebool=True):
        Nrow = np.arange(len(self.tab_lin_order))
        self.tab_lin_order["maskgood"] = [self.tableWidget_files.cellWidget(irow, 3).currentIndex() for irow in Nrow]
        if addlinebool:
            try:
                currentitem = mainprocess.tableWidget_files.currentItem()
                colname = self.tab_lin_order.colnames[currentitem.column()]
                _tab = self.tab_lin_order
                line = np.float(currentitem.text())
                if self.linelist is not None:
                    _dline = np.abs(self.linelist - line)
                    line = self.linelist[np.argmin(_dline)]
                    print(line)
                _tab[currentitem.row()][colname] = line
                self.tab_lin_order =_tab
            except:
                pass
                #print("'NoneType' object has no attribute 'column'")
        self._refresh_datatable()
        self.tables[self.order-1]= self.tab_lin_order
        tab = vstack(self.tables)
        self.tab_line_all = tab
        self._draw_wvfit()
        self._draw_residual()

    def _get_tab_lin_order(self):
        self._make_datatable()
        self._refresh_datatable()

    def _refresh_datatable(self):
        if self.tab_lin_order is None:
            return
        # change to Table Widget
        self.tableWidget_files.clear()
        self.tableWidget_files.verticalHeader().setVisible(True)
        Nrow = len(self.tab_lin_order)
        self.tableWidget_files.setRowCount(Nrow)
        n_col = 6
        self.tableWidget_files.setColumnCount(n_col)
        self.tableWidget_files.setHorizontalHeaderLabels(self.tab_lin_order.colnames)
        for irow in range(Nrow):
            self.tableWidget_files.setItem(irow, 0, QtWidgets.QTableWidgetItem(str(self.tab_lin_order["order"][irow])))
            self.tableWidget_files.setItem(irow, 1, QtWidgets.QTableWidgetItem(str(self.tab_lin_order["xcoord"][irow])))
            self.tableWidget_files.setItem(irow, 2, QtWidgets.QTableWidgetItem(str(self.tab_lin_order["line"][irow])))
            comboBoxItem = QtWidgets.QComboBox()
            comboBoxItem.addItems(['drop', 'good'])
            # print(self.type_dict[self.tab_lin_order["type"][irow]])
            good_bool = int(self.tab_lin_order["maskgood"][irow])
            comboBoxItem.setCurrentIndex(good_bool)
            self.tableWidget_files.setCellWidget(irow, 3, comboBoxItem)
            self.tableWidget_files.setItem(irow, 4, QtWidgets.QTableWidgetItem(str(self.tab_lin_order["wv_fit"][irow])))
            self.tableWidget_files.setItem(irow, 5, QtWidgets.QTableWidgetItem(f'{self.tab_lin_order["Fpeak"][irow]:.2f}'))

            for icol in range(3):
                self.tableWidget_files.item(irow, icol).setBackground(
                    QtGui.QBrush(QtGui.QColor(*self.color_list[good_bool])))

        self.tableWidget_files.resizeColumnsToContents()
        self.tableWidget_files.resizeRowsToContents()

    def _draw_scatter_pos(self, **keywds):
        ind_currentRow = self.tableWidget_files.currentRow()
        _tab = self.tab_lin_order[ind_currentRow]
        print("Plot order {}: xcoord = {}".format(*_tab['order', 'xcoord']))
        # try to draw it
        self.pos_line = [_tab['xcoord'], _tab['Fpeak']]
        #self._draw_spec(tab=_tab)
        try:
            self.scatter_pos_flux.remove()
            self.scatter_pos_wvfit.remove()
            self.scatter_pos_resid.remove()
        except:
            pass
        self.scatter_pos_flux = self.axs[0].scatter(_tab['xcoord'], _tab['Fpeak'], marker='o', lw=1, color='b',facecolors='None', **keywds)
        self.scatter_pos_wvfit = self.axs[1].scatter(_tab['xcoord'], _tab['line'], marker='o', lw=1, color='b',facecolors='None', **keywds)
        self.scatter_pos_resid = self.axs[2].scatter(_tab['xcoord'], _tab['line'] - _tab['wv_fit'], marker='o', lw=1, color='b',facecolors='None', **keywds)
        self.canvas.draw()

    def _draw_spec(self, tab=None):
        # draw
        iorder = self.order -1 if tab is None else int(tab['order'] - 1)
        flux = self.fluxes[iorder]
        #self.figure.clear()
        #self.ax = self.figure.add_axes([0.08, 0.08, 0.99, 0.92])
        #[ax.clear() for ax in self.axs]
        self.axs[0].clear()
        self.axs[0].plot(self.xindex, flux, color='k', lw=1)
        self.axs[0].scatter(self.xindex, flux, s=2, color='k')
        if tab is not None:
            self.axs[0].scatter(tab['xcoord'], tab['Fpeak'], marker='o', lw=1, color='b',facecolors='None')
        if self.tab_lin_order is not None:
            _ind = self.tab_lin_order['maskgood'] == 1
            _tab = self.tab_lin_order[_ind]
            self.axs[0].scatter(_tab['xcoord'], _tab['Fpeak'], marker='+', lw=1, color='b', label='Good')
            _tab = self.tab_lin_order[~_ind]
            self.axs[0].scatter(_tab['xcoord'], _tab['Fpeak'], marker='x', lw=1, color='r', label='Bad')
            self.axs[0].legend()
        #self.ax.legend()
        #self.ax.set_ylabel('Counts (ADU)')
        #if self.waves is not None:
        #def wv2index()
        #if self.wave is not None:
        #   ax2 = ax.twiny()
        #   ax2.set_xlim(ax.get_xlim())
        #ax.secondary_xaxis?
        # refresh canvas
        self.axs[0].set_ylabel('Counts')
        #self.axs[1].set_ylabel(r'$\lambda$ (${\rm \AA}$)')
        #self.axs[2].set_ylabel(r'Res. (${\rm \AA}$)')
        #self.axs[2].set_xlabel('Pixel')
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def _draw_wvfit(self, table=None):
        tab = self.tab_lin_all
        self.axs[1].clear()
        x = tab['xcoord']
        y = tab['line']
        wvfit = tab['wv_fit']
        maskgood = np.array(tab['maskgood'], dtype=bool)
        #try:
        #    self.draw_wvfit_good.remove()
        #    self.draw_wvfit_bad.remove()
        #    self.draw_wvfit.remove()
        #except: pass
        #try:
        #   self.draw_wvfit_good = self.axs[1].scatter(x[maskgood], y[maskgood], marker='+', color='b', label='Good')
        #   self.draw_wvfit_bad = self.axs[1].scatter(x[~maskgood], y[~maskgood], marker='x', color='r', label='Bad')
        #   self.draw_wvfit = self.axs[1].plot(x, wvfit)
        #except: pass
        self.axs[1].scatter(x[maskgood], y[maskgood], marker='+', color='b', label='Good', lw=1)
        self.axs[1].scatter(x[~maskgood], y[~maskgood], marker='x', color='r', label='Bad', lw=1)
        self.axs[1].plot(x, wvfit, color='k')
        self.axs[1].set_ylabel(r'$\lambda$ (${\rm \AA}$)')
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def _draw_residual(self, table=None):
        if table is None: table = self.tab_lin_all
        self.axs[2].clear()
        x = table['xcoord']
        y = table['line']- table['wv_fit']
        maskgood = np.array(table['maskgood'], dtype=bool)
        #try:
        #    self.draw_residual_good.remove()
        #    self.draw_residual_bad.remove()
        #except: pass
        #try:
        #   self.draw_residual_good = self.axs[2].scatter(x[maskgood], y[maskgood], marker='+', lw=1, color='b', label='Good')
        #   self.draw_residual_bad =self.axs[2].scatter(x[~maskgood], y[~maskgood], marker='x', lw=1, color='r', label='Bad')
        #except: pass
        self.axs[2].scatter(x[maskgood], y[maskgood], marker='+', lw=1, color='b', label='Good')
        self.axs[2].scatter(x[~maskgood], y[~maskgood], marker='x', lw=1, color='r', label='Bad')
        self.axs[2].axhline(y=0, lw=0.8, ls='--', color='k')
        self.axs[2].set_ylabel(r'Res. (${\rm \AA}$)')
        self.axs[2].set_xlabel('Pixel')
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def onclick(self, event):
        # capture cursor position ===============
        # ref: https://matplotlib.org/stable/users/event_handling.html
        # https://stackoverflow.com/questions/39351388/control-the-mouse-click-event-with-a-subplot-rather-than-a-figure-in-matplotlib
        _tab = self.tab_lin_order
        if event.inaxes == self.axs[0]:
           tab = self.tab_lin_order
           self.pos_line_clickevent = event.xdata, event.ydata
           x_tmp, y_tmp = event.xdata, event.ydata
           dx = np.abs(tab['xcoord'] - x_tmp)
           ind = np.where(dx == np.min(dx))[0][0]
        elif event.inaxes == self.axs[1]:
           tab = self.tab_lin_all
           x_tmp, y_tmp = event.xdata, event.ydata
           y = tab['wv_fit']
           dx = np.sqrt((tab['xcoord']-x_tmp)**2 +(y-y_tmp)**2)
           ind = np.where(dx == np.min(dx))[0][0]
        elif event.inaxes == self.axs[2]:
           tab = self.tab_lin_all
           x_tmp, y_tmp = event.xdata, event.ydata
           res = tab['line'] - tab['wv_fit']
           dx = np.sqrt((tab['xcoord']-x_tmp)**2 +(res-y_tmp)**2)
           ind = np.where(dx == np.min(dx))[0][0]
        _tab = tab[ind]
        iorder = int(_tab['order'] - 1)
        self.tab_lin_order = self.tables[iorder]
        print('iorder=', iorder)
        self.order = iorder +1
        self._refresh_datatable()
        rowID = np.where(self.tab_lin_order['xcoord']==_tab['xcoord'])[0][0]
        self.tableWidget_files.selectRow(rowID)



    def _draw_add_line(self):
        x, y = self.pos_line
        self.axs[0].scatter(x, y, marker='+', lw=1, color='b')
        self.canvas.draw()
        # and trace this aperture
        print(self.pos_line)

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

    def _upper_order(self):
        order = self.order -1 if (self.order > 1) else self.Norder
        self.order = order
        self.tab_lin_order = self.tables[order-1]
        self._refresh_datatable()
        self._draw_spec()
        self._update_order()

    def _next_order(self):
        order = self.order +1 if (self.order < self.Norder) else 1
        self.order = order
        self.tab_lin_order = self.tables[order-1]
        self._refresh_datatable()
        self._draw_spec()
        self._update_order()

    def _clear_line(self):
        # from twodspec.trace import trace_naive_max
        # self.ap_trace = trace_naive_max(self.master_flat.T, sigma=7, maxdev=10, irow_start=1300) 
        self.ap_trace = np.zeros((0, self.nwv), dtype=int)
        self._update_nap()
        # print(self.ap_trace.shape)
        self._draw_aperture()


    def find_local_max(self, x, y, x_tmp, dx=7):
        ind_l = (x >= x_tmp-dx)
        ind_r = (x <= x_tmp+dx)
        ind = ind_l & ind_r
        y_select = y[ind]
        ind = np.sum(~ind_l) + np.argmax(y_select)
        x_max, y_max = x[ind], y[ind]
        return x_max, y_max


    def _add_line(self):
        #try:
        #if self.aperture_image is None:
        x = self.xindex
        y = self.fluxes[self.order-1]
        x_tmp, y_tmp = self.pos_line_clickevent
        x_max, y_max = self.find_local_max(x, y, x_tmp)
        _tab = self.tab_lin_order
        ind = (_tab['xcoord'] == x_max)  & (_tab['Fpeak'] == y_max)
        if any(ind):
            print('This line is existence!')
        else:
            _tab = self._add_tab_oneline(colnames=self.tab_lin_order.colnames)
            _tab['xcoord'] = x_max
            _tab['Fpeak'] = y_max
            _tab = vstack([_tab, self.tab_lin_order])
            _tab.sort('xcoord')
            self.tab_lin_order = _tab
            self.tables[self.order-1]= self.tab_lin_order
            self.tab_lin_all = vstack(self.tables)
            print("An line is added")
        self._refresh_datatable()
        self.pos_line = [x_max, y_max]
        #self.table[self.order-1] = self.tab_lin_order
        #self.tab_line
        self._update_datatable(addlinebool=True)
        self._draw_add_line()
        self._draw_wvfit()
        self._draw_residual()
        #except Exception as _e:
        #    print("Error occurred, aperture not added!")

    def _del_line(self):
        if len(self.tab_lin_order) == 0:
            print(f'The Number of lines = {len(self.tab_lin_order)}')
            return
        x_tmp, y_tmp = self.pos_line
        _tab = self.tab_lin_order
        dx = np.abs(_tab['xcoord'] - x_tmp)
        ind = dx == np.min(dx)
        _tab = _tab[~ind]
        self.tab_lin_order = _tab
        self.tables[self.order-1]= self.tab_lin_order
        tab = vstack(self.tables)
        self.tab_lin_all = tab
        self._refresh_datatable()
        self._update_datatable(addlinebool=False)
        self._draw_spec()
        self._draw_wvfit()
        self._draw_residual()
        print("An line is deleted")

    def _save_lines(self):
        dire_table = os.path.join(self._wd, 'TABLE')
        tab = self.tab_line_all
        if not os.path.exists(dire_table): os.makedirs(dire_table)
        tab.write(os.path.join(dire_table, 'table_linelist.csv'), overwrite=True)
        self._draw_spec()
        self._draw_wvfit()
        self._draw_residual()


    def _update_order(self):
        self.lineEdit_order.setText(f"order({self.Norder}) = {self.order}")



    def _fit(self, xdeg=None, ydeg=None):
        '''
        Using modified np.polynomial.chebyshev to fit the wavelength, which copy from twospec.polynomial written by Zhang, Bo.
        '''
        #from PyQt6.QtWidgets import QLabel
        #self.label_xdeg = QLabel()
        #self.lineEdit_xdeg.textChanged.connect(self.label_xdeg.setText)
        from pyexspec.fitfunc import Polyfitdic, Poly1DFitter,Poly2DFitter
        print(self.lineEdit_xdeg.text())
        print(self.lineEdit_ydeg.text())
        Fittername = 'Poly1DFitter'
        xdeg = int(self.lineEdit_xdeg.text())
        ydeg = int(self.lineEdit_ydeg.text())
        tab = self.tab_lin_all
        x, y, z, indselect = tab['xcoord'], tab['order']-1, tab['line'], tab['maskgood']
        indselect = np.array(indselect, dtype=bool)
        if Fittername == 'Poly1DFitter':
            deg = xdeg
            Fitter = Poly1DFitter
            Polyfunc = Fitter(x[indselect], z[indselect], deg=xdeg, pw=2, robust=False)
            wv_fit = Polyfunc.predict(x)
            wave_solu = Polyfunc.predict(self.xindex)
        elif Fittername == 'Poly2DFitter':
            Fitter = Poly2DFitter
            deg = (xdeg, ydeg)
            Polyfunc = Fitter(x[indselect], y[indselect], z[indselect], deg=xdeg, pw=1, robust=False)
            wv_fit = Polyfunc.predict(x, y)
            wave_solu = Polyfunc.predict(self.Xgrid, self.Ygrid)
        self.wave_solu = wave_solu
        tab['wv_fit'] = wv_fit
        self.tab_lin_all = tab
        tables = self._split_tab_lin_all()
        self._draw_wvfit()
        self._draw_residual()
        self._update_datatable()
        #Polyfunc = Poly1DFitter(x[indselect], z[indselect], deg=xdeg, pw=2, robust=False)
        # pf2 = Poly2DFitter(x[indselect], y[indselect], z[indselect], deg=deg, pw=2, robust=False)
        pass

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
    mainprocess = UiWvcalib()
    # ui.setupUi(mainWindow)
    # ui.initUi(mainWindow)
    mainprocess.show()
    try:
        sys.exit(app.exec_())# pyqt5
        #sys.exit(app.exec())# pyqt5
    except:
        sys.exit(app.exec())# pyqt6
