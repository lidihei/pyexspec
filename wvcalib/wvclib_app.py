import sys, os, glob
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    pyqt5_bool = True
except:
    from PyQt6 import QtCore, QtGui, QtWidgets
    pyqt5_bool = False
from pyexspec.gui import WvClibWindow
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from astropy.table import Table, vstack, hstack
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
#from pyexspec.findline import find_lines 
import warnings
warnings.filterwarnings('ignore')

#matplotlib.use('Qt5Agg')
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
        self.pos_line = [0, 0]
        # UI
        self.setupUi(self)
        self.add_canvas()
        self.order = 0
        #self.assumption() # debug
        self.initUi()
        # debug
        #self._wd = os.path.dirname(self._arcfile) if wd is None else wd
        self.templatefile = None

    def add_canvas(self):
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(600, 10, 670, 600))
        self.widget2.setObjectName("widget")
        self.verticalLayout2 = QtWidgets.QVBoxLayout(self.widget2)
        #self.verticalLayout2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout2.setObjectName("layout_plot_spec")

        # a figure instance to plot on
        #self.figure, self.axs = plt.subplots(2,1, sharex=True,
        #            gridspec_kw={'height_ratios': [3.5, 1]})
        self.figure, self.ax = plt.subplots(1,1)
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
        #if self.checkBox_invert_xaxis.checkState().value == 0:
        #   self.axs[0].invert_xaxis()
        # layout.addWidget(self.button)
        # self.setLayout(layout)


    def assumption(self):
        test_dir = "test"
        self._wd = test_dir
        self._arcfile = os.path.join(self._wd, "test.dump")
        test_dir = "test"
        self._wd = test_dir
        self._arcfile = os.path.join(self._wd, "lamp-202403140107_SPECSLAMP_FeAr_slit16s_G10_E9.z")
        self._wd = '../bfoscE9G10/template'
        self._arcfile = os.path.join(self._wd, "template_from_202305220046.dump")
        self._linelistfile = '../arc_linelist/FeAr.dat'
        self.lineEdit_arc.setText(self._arcfile)
        self.lineEdit_linelist.setText(self._linelistfile)
        self._read_linelist(self._linelistfile)
        tablename = os.path.join(self._wd, "TABLE", "table_linelist.csv")
        tablename = self._arcfile
        self._read_arc(self._arcfile, tablename=None)
        self._get_tab_line_order()
        #self._read_arc(self._arcfile, tablename=self._arcfile)
        #self._arcfile = joblib.load(os.path.join(self._wd, "lamp-blue0062.fits_for_blue0060.fits.dump"))

    def initUi(self):
        self.toolButton_load_arc.clicked.connect(self._select_arc)
        self.toolButton_load_linelist.clicked.connect(self._select_linelist)
        self.toolButton_load_template.clicked.connect(self._select_template)
        self.lineEdit_arc.textChanged.connect(self._get_tab_line_order)
        self.tableWidget_files.itemSelectionChanged.connect(self._draw_scatter_pos)
        self.pushButton_upper_order.clicked.connect(self._upper_order)
        self.pushButton_next_order.clicked.connect(self._next_order)
        self.pushButton_plot_spec.clicked.connect(self._plot_spec)
        self.pushButton_invert_xaxis.clicked.connect(self._invert_xaxis)
        self.pushButton_update_table.clicked.connect(self._update_datatable_button)
        self.pushButton_add_line.clicked.connect(self._add_line)
        self.pushButton_del_line.clicked.connect(self._del_line)
        self.pushButton_save_line.clicked.connect(self._save_lines)
        self.pushButton_fit.clicked.connect(self._fit)
        self.pushButton_autofind.clicked.connect(self._autofind)
        #self.lineEdit_xdeg.setPlaceholderText('X deg (int)')
        #self.lineEdit_xdeg.setText(f"deg(x) = ")
        #self.lineEdit_ydeg.setText(f"deg(y) = ")
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
        self._wd = os.path.dirname(fileName)
        self.lineEdit_arc.setText(fileName)
        self._arcfile = fileName
        self._read_arc(fileName)
        self._get_tab_line_order()
        print(f"arc loaded: {os.path.basename(fileName)}")


    def _select_template(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open arc", "dump files (*.dump)")
        self.lineEdit_template.setText(filename)
        self.templatefile = filename
        basename = os.path.basename(filename)
        if ('.dump' in basename) or ('.z' in basename):
            data = joblib.load(filename)
            try:
                waves_template, fluxes_template = data['wave_solu'], data['flux_arc']
            except:
                waves_template, fluxes_template = data['wave'], data['flux']
        if ('.fits' in basename) or ('csv' in basename):
            data = Table.read(filename)
            waves_template, fluxes_template = data['wave'], data['flux']
        self.waves_template = waves_template
        self.fluxes_template = fluxes_template

    def _read_arc(self, fileName, tablename=None, wave_init=None, wave_solu =None):
        '''
        parameters:
        --------------------
        fileName [str]: the filename of arc dump file
        table [str]: file name of the found emission lines table
                  that must include 'order', 'xcoord', 'line', 'Fpeak','good',  'wv_fit'
        '''
        ext = os.path.splitext(fileName)[1]
        print(f'arcname = {fileName}')
        if (ext == '.dump') or (ext == '.z'):
            self.arcdic = joblib.load(fileName)
        elif (ext == '.fits') or (ext=='.csv'):
            data = Table.read(fileNmae)
            arcdic = {'flux' : np.array(data['flux'])}
            self.arcdic = arcdic
        try:
            flux = self.arcdic['flux_arc']
        except:
            try:
                flux = self.arcdic['spec_extr']
            except:
                flux = self.arcdic['flux']
        shape  = flux.shape
        try:
            wave_init = np.array(self.arcdic['wave_init']) if wave_init is None else wave_init
            wave_solu = np.array(self.arcdic['wave_solu']) if wave_solu is None else wave_solu
        except:
            print('the arc file does not cantain calibratied wave')
        if len(shape) == 1:
            Norder, Npixel = 1,shape[0]
            flux = np.array([flux]).T
        else:
            Norder, Npixel = shape
        if wave_init is not None:
            shape_wave = wave_init.shape
            if len(shape_wave) == 1: wave_init = np.array([wave_init]).T
        if wave_solu is not None:
            shape_wave = wave_solu.shape
            if len(shape_wave) == 1: wave_solu = np.array([wave_solu]).T
        if tablename is None: tablename = self.tablename_lines
        print(f'the table name of arc lamp = {tablename}')
        self.wave_inits = wave_init if wave_init is not None else np.ones_like(flux)
        self.wave_solus = wave_solu if wave_solu is not None else self.wave_inits.copy()
        self.fluxes = flux
        self.Norder = Norder
        self.tables = self._read_linetable(Norder=Norder, tablename=tablename)
        self.Npixel = Npixel
        #self.Xgrid, self.Ygrid = np.meshgrid(np.arange(Norder), np.arange(Npixel))
        self.Xgrid, self.Ygrid = np.meshgrid(np.arange(Npixel), np.arange(Norder))
        self.xindex = np.arange(Npixel)
        self._update_order()
        self.tab_line_order = self.tables[0]
        #self._draw_spec()
        #self._draw_residual()
        #self._draw_wvfit()

    def _add_tab_oneline(self, colnames=None, dtype=None):
        if dtype is None:
            if colnames is None:
               colnames = ['order', 'xcoord', 'line', 'maskgood', 'wv_fit', 'Fpeak']
            tab = Table(data=np.zeros(len(colnames)).T, names=colnames)
        else:
            tab = Table(data=np.zeros(len(dtype)).T, dtype=dtype)
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
        orders = np.arange(Norder)
        if (tablename is None) and (table is None):
            names = ['order', 'xcoord', 'line', 'maskgood', 'wv_fit', 'Fpeak']
            tables = [Table(data= [[i],[-999.], [-999.], [0], [-999.], [-999.]], names=names) for i in orders]
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
                    colnames_old = ['line_x_ccf', 'line_wave_init_ccf', 'line_peakflux', 'ind_good']
                    colnames_new = ['xcoord', 'wv_fit', 'Fpeak', 'maskgood']
                    for _i, colname in enumerate(colnames_old):
                        if colname in tab.colnames:
                            tab.rename_column(colname, colnames_new[_i])
                else:
                    tab = Table.read(tablename)
            else:
                tab = table
            #tab = tab[np.array(tab['maskgood'], dtype=bool)]
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
            tables = self._split_tab_line_all(tab_line_all=tab)
        self.tab_line_all = vstack(tables)
        self.colnames = tables[0].colnames
        return tables

    def _split_tab_line_all(self, tab_line_all=None):
        tab = self.tab_line_all if tab_line_all is None else tab_line_all
        orders = np.arange(self.Norder)
        tables = [tab[tab['order'] == i] for i in orders]
        self.tables = tables
        return tables

    def _set_wd(self):
        self._arcfile = self.lineEdit_arc.text()

    def _make_datatable(self):
        iorder = self.order
        tab_line_order =  self.tables[iorder]
        #datatable.sort('xcoord')
        self.tab_line_order = tab_line_order
        self.color_list = [[255, 255, 255],
                           [211, 211, 211]]
        #                   [255, 182, 193],
        #                   [255, 228, 181],
        #                   [173, 216, 230], ]


    def _find_line(self, line_tmp, linelist=None):
        try:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState()
        except:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState().value
        if (line_tmp ==0) and checkBox_xaxis_wave_state != 0:
           try: line_tmp = self.wv_max
           except: line_tmp = 0
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
        Nrow = np.arange(len(self.tab_line_order))
        self.tab_line_order["maskgood"] = [self.tableWidget_files.cellWidget(irow, 3).currentIndex() for irow in Nrow]
        if addlinebool:
            try:
                currentitem = self.tableWidget_files.currentItem()
                colname = self.tab_line_order.colnames[currentitem.column()]
                print(f'colname = {colname}')
                line_tmp = np.float64(currentitem.text())
                print(f'currentitem.text() = {currentitem.text()}')
                line = self._find_line(line_tmp, linelist=None)
                _tab = self.tab_line_order
                _tab[currentitem.row()][colname] = line
                self.tab_line_order =_tab
            except Exception as e:
                print(e)
                print('set value for the line in Table')
            #    print("'NoneType' object has no attribute 'column'")
        self._refresh_datatable()
        self.tables[self.order]= self.tab_line_order
        tab = vstack(self.tables)
        self.tab_line_all = tab
        #self._draw_wvfit()
        #self._draw_residual()

    def _get_tab_line_order(self):
        self._make_datatable()
        self._refresh_datatable()

    def _refresh_datatable(self):
        if self.tab_line_order is None:
            return
        # change to Table Widget
        self.tableWidget_files.clear()
        self.tableWidget_files.verticalHeader().setVisible(True)
        Nrow = len(self.tab_line_order)
        self.tableWidget_files.setRowCount(Nrow)
        n_col = 6
        self.tableWidget_files.setColumnCount(n_col)
        self.tableWidget_files.setHorizontalHeaderLabels(self.tab_line_order.colnames)
        for irow in range(Nrow):
            self.tableWidget_files.setItem(irow, 0, QtWidgets.QTableWidgetItem(str(self.tab_line_order["order"][irow])))
            self.tableWidget_files.setItem(irow, 1, QtWidgets.QTableWidgetItem(str(self.tab_line_order["xcoord"][irow])))
            self.tableWidget_files.setItem(irow, 2, QtWidgets.QTableWidgetItem(str(self.tab_line_order["line"][irow])))
            comboBoxItem = QtWidgets.QComboBox()
            comboBoxItem.addItems(['drop', 'good'])
            # print(self.type_dict[self.tab_line_order["type"][irow]])
            good_bool = int(self.tab_line_order["maskgood"][irow])
            comboBoxItem.setCurrentIndex(good_bool)
            self.tableWidget_files.setCellWidget(irow, 3, comboBoxItem)
            self.tableWidget_files.setItem(irow, 4, QtWidgets.QTableWidgetItem(str(self.tab_line_order["wv_fit"][irow])))
            self.tableWidget_files.setItem(irow, 5, QtWidgets.QTableWidgetItem(f'{self.tab_line_order["Fpeak"][irow]:.2f}'))

            for icol in range(3):
                self.tableWidget_files.item(irow, icol).setBackground(
                    QtGui.QBrush(QtGui.QColor(*self.color_list[good_bool])))

        self.tableWidget_files.resizeColumnsToContents()
        self.tableWidget_files.resizeRowsToContents()
        self._get_selfxcoord(xname='xcoord', yname='Fpeak', wvname='line')

    def _get_selfxcoord(self, xname='xcoord', yname='Fpeak', wvname='line'):
        try:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState()
        except:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState().value
        tab = self.tab_line_order.copy()
        if checkBox_xaxis_wave_state == 0:
            self.xcoord = self.xindex
            self.xscatter = tab[xname]
        else:
            if wvname == 'line':
                tab = tab[tab['line']>0]
            self.xcoord = self.wave_solus[self.order]
            self.xscatter = tab[wvname]
        self.yscatter = tab[yname]

    def _draw_scatter_pos(self, ind_currentRow=None, **keywds):
        if ind_currentRow is None: ind_currentRow = self.tableWidget_files.currentRow()
        _tab = self.tab_line_order[ind_currentRow]
        print("Plot order {}: xcoord = {}".format(*_tab['order', 'xcoord']))
        # try to draw it
        self.pos_line = [_tab['xcoord'], _tab['Fpeak']]
        xscatter = self.xscatter[ind_currentRow]
        yscatter = self.yscatter[ind_currentRow]
        #self._draw_spec(tab=_tab)
        try:
            self.scatter_pos_flux.remove()
        except:
            pass
        if yscatter != -999:
             scatter_pos_flux = self.ax.scatter(xscatter, yscatter, marker='o', lw=1, color='g',facecolors='None', **keywds)
             self.scatter_pos_flux = scatter_pos_flux
             self.canvas.draw()
        try:self.fitWindow._draw_scatter_pos()
        except: pass

    def _plot_spec(self):
        self._draw_spec()

    def _draw_spec(self, xname='xcoord', yname='Fpeak', wvname='line'):
        # draw
        self._get_selfxcoord(xname=xname, yname=yname, wvname=wvname)
        try:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState()
        except:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState().value
        if checkBox_xaxis_wave_state == 0:
            self.ax.set_xlabel('Pixel')
        else:
            self.ax.set_xlabel('Wavelength')
        xscatter = self.xscatter
        yscatter = self.yscatter
        x = self.xcoord
        flux = self.fluxes[self.order]
        #self.figure.clear()
        self.ax.clear()
        self.ax.plot(x, flux, color='k', lw=1, alpha=0.5)
        self.ax.scatter(x, flux, s=2, color='k', facecolor='none')
        #if tab is not None:
        #    self.ax.scatter(xscatter, yscatter, marker='o', lw=1, color='b',facecolors='None')
        if self.tab_line_order is not None:
            _ind = (self.tab_line_order['maskgood'] == 1)
            _ind1 = (self.tab_line_order['xcoord'] >= 0)
            self.ax.scatter(xscatter[_ind], yscatter[_ind], marker='+', lw=1, color='b', label='Good')
            ind_ = (~_ind) & _ind1
            self.ax.scatter(xscatter[ind_], yscatter[ind_], marker='x', lw=1, color='r', label='Bad')
            self.ax.legend()
        #ax.secondary_xaxis?
        # refresh canvas
        self.ax.set_ylabel('Counts')
        if checkBox_xaxis_wave_state == 0:
            self.ax.set_xlabel('Pixel')
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()



    def onclick(self, event):
        # capture cursor position ===============
        # ref: https://matplotlib.org/stable/users/event_handling.html
        # https://stackoverflow.com/questions/39351388/control-the-mouse-click-event-with-a-subplot-rather-than-a-figure-in-matplotlib
        if event.inaxes == self.ax:
           tab = self.tab_line_order
           self.pos_line_clickevent = event.xdata, event.ydata
           x_tmp, y_tmp = event.xdata, event.ydata
           #dx = np.abs(tab['xcoord'] - x_tmp)
           dx = np.abs(self.xscatter - x_tmp)
           self.dx_test = dx
           ind = np.nanargmin(dx)
           print(f'rowID = {ind}')
           print(f'(x_tmp, y_tmp) = [{x_tmp}, {y_tmp}]')
        _tab = tab[ind]
        xclick = self.xscatter[ind].copy()
        order = int(_tab['order'])
        self.tab_line_order = self.tables[order]
        print('order=', order)
        self.order = order
        self._refresh_datatable()
        rowID = np.where(self.xscatter==xclick)[0][0]
        self.tableWidget_files.selectRow(rowID)

    def _draw_add_line(self, x, y):
        self.ax.scatter(x, y, marker='+', lw=1, color='b')
        self.canvas.draw()
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
        order = self.order -1 if (self.order > 0) else self.Norder-1
        self.order = order
        self.tab_line_order = self.tables[order]
        self._refresh_datatable()
        self._draw_spec()
        self._update_order()

    def _next_order(self):
        order = self.order +1 if (self.order < self.Norder-1) else 0
        self.order = order
        self.tab_line_order = self.tables[order]
        self._refresh_datatable()
        self._draw_spec()
        self._update_order()

    def _invert_xaxis(self):
        self.ax.invert_xaxis()
        self.canvas.draw()


    def find_local_max(self, x, y, x_tmp, y_tmp, index_window=7, x_window =7,  Gaussianfit=False):
        '''
        If using Gassian function to find peak, index_window (Ind Windows) is the window width to find max;
        x_window (X Window) is the window width to fit Gaussian function.
        '''
        ds_squared = (x-x_tmp)**2 + (y-y_tmp)**2
        ind_nearest = np.nanargmin(ds_squared)
        ind_l = ind_nearest-index_window
        y_select = y[ind_l: ind_nearest+index_window]
        indmax = ind_l + np.argmax(y_select)
        x_max, y_max = x[indmax], y[indmax]
        ind_max = indmax
        try:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState()
        except:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState().value
        xarange = x if checkBox_xaxis_wave_state == 0 else np.arange(len(x))
        ind = (xarange >= indmax -x_window) & (xarange < indmax+x_window)
        if Gaussianfit:
            from pyexspec.fitfunc.function import gaussian_linear_func
            from scipy.optimize import curve_fit
            p0 = [y_max/2, indmax, index_window/2, 0, 0]  # A mu sigma, K c
            xdata = xarange[ind]
            ydata = y[ind]
            print(xdata)
            popt, pcov = curve_fit(gaussian_linear_func, xdata, ydata, p0=p0)
            print('Gaussian peak A={}, mu={}, sigma={}, k={}, c={}'.format(*popt))
            ind_max = popt[1]
            y_max = gaussian_linear_func(popt[1], *popt)
        if checkBox_xaxis_wave_state == 0:
            self.wv_max = np.nan
        else:
            self.wv_max = np.interp(ind_max, xarange[ind], x[ind])
        print(f'[x_max, y_max] = [{x_max}, {y_max}]; ind_max = {ind_max}')
        return ind_max, y_max


    def _add_line(self):
        x = self.xcoord
        y = self.fluxes[self.order]
        x_tmp, y_tmp = self.pos_line_clickevent
        try:
            checkBox_gaussianfit_state = self.checkBox_gaussianfit.checkState()
        except:
            checkBox_gaussianfit_state = self.checkBox_gaussianfit.checkState().value

        if checkBox_gaussianfit_state != 0:
           Gaussianfit = True
        else: Gaussianfit = False; print('Finding peak by maxium value')
        try: index_window = int(self.lineEdit_index_window.text())
        except: index_window = 7
        try: x_window = float(self.lineEdit_x_window.text())
        except: x_window = 7
        x_max, y_max = self.find_local_max(x, y, x_tmp, y_tmp, index_window=index_window, x_window=x_window, Gaussianfit=Gaussianfit)
        _tab = self.tab_line_order
        ind = (_tab['xcoord'] == x_max)  & (_tab['Fpeak'] == y_max)
        if any(ind):
            print('This line is existence!')
            self._update_datatable(addlinebool=True)
        else:
            try:
                _tab = self._add_tab_oneline(dtype=self.tab_line_order.dtype)
            except:
                _tab = self._add_tab_oneline(colnames=self.tab_line_order.colnames)
            _tab['xcoord'] = x_max
            _tab['Fpeak'] = y_max
            _tab['wv_fit'] = self.wv_max
            _tab = vstack([_tab, self.tab_line_order])
            _tab.sort('xcoord')
            self.tab_line_order = _tab
            self.tables[self.order]= self.tab_line_order
            self.tab_line_all = vstack(self.tables)
            print("An line is added")
        self._refresh_datatable()
        self._update_datatable(addlinebool=True)
        self.pos_line = [x_max, y_max]
        try:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState()
        except:
            checkBox_xaxis_wave_state = self.checkBox_xaxis_wave.checkState().value
        if checkBox_xaxis_wave_state != 0: x_max = self.wv_max
        self._draw_add_line(x_max, y_max)
        try: self.fitWindow._update_fitWindow(UiWvcalib=self)
        except:pass

    def _del_line(self):
        if len(self.tab_line_order) == 0:
            print(f'The Number of lines = {len(self.tab_line_order)}')
            return
        x_tmp, y_tmp = self.pos_line
        _tab = self.tab_line_order
        dx = np.abs(_tab['xcoord'] - x_tmp)
        ind = dx == np.min(dx)
        _tab = _tab[~ind]
        self.tab_line_order = _tab
        self.tables[self.order]= self.tab_line_order
        tab = vstack(self.tables)
        self.tab_line_all = tab
        self._refresh_datatable()
        self._update_datatable(addlinebool=False)
        self._draw_spec()
        try:self.fitWindow._update_fitWindow(UiWvcalib=self)
        except: pass
        print("Delete a line")

    def _save_lines(self):
        dire_table = os.path.join(self._wd, 'TABLE')
        tab = self.tab_line_all
        print('##---save_lines---------------')
        if not os.path.exists(dire_table): os.makedirs(dire_table)
        tab.write(os.path.join(dire_table, 'table_linelist.csv'), overwrite=True)
        self._draw_spec()


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
        print(f'xdeg = {self.lineEdit_xdeg.text()}')
        print(f'ydeg = {self.lineEdit_ydeg.text()}')
        Fittername = self.comboBox_fitfunc.currentText()
        try:
            xdeg = int(self.lineEdit_xdeg.text())
        except:
            xdeg = 1
        try:
            ydeg = int(self.lineEdit_ydeg.text())
        except:
            ydeg = 1
        tab = self.tab_line_all.copy()
        x, y, z, indselect = tab['xcoord'], tab['order'], tab['line'], tab['maskgood']
        indselect = np.array(indselect, dtype=bool)
        print(f'np.sum(indselect) = {np.sum(indselect)}')
        if Fittername == 'Poly1DFitter':
            deg = xdeg
            Fitter = Polyfitdic[Fittername]
            Polyfunc = Fitter(x[indselect], z[indselect], deg=xdeg, pw=2, robust=False)
            wv_fit = Polyfunc.predict(x)
            wave_solu = np.array([Polyfunc.predict(self.xindex)])
        elif Fittername == 'Poly2DFitter':
            Fitter = Polyfitdic[Fittername]
            deg = (xdeg, ydeg)
            Polyfunc = Fitter(x[indselect], y[indselect], z[indselect], deg=deg, pw=1, robust=False)
            wv_fit = Polyfunc.predict(x, y)
            wave_solu = Polyfunc.predict(self.Xgrid, self.Ygrid)
        self.Polyfunc = Polyfunc
        self.wave_solus = wave_solu
        tab['wv_fit'] = wv_fit
        self.tab_line_all = tab
        tables = self._split_tab_line_all()
        #self._draw_wvfit()
        #self._draw_residual()
        #self._update_datatable(addlinebool=False) ## lijiao
        #Polyfunc = Poly1DFitter(x[indselect], z[indselect], deg=xdeg, pw=2, robust=False)
        # pf2 = Poly2DFitter(x[indselect], y[indselect], z[indselect], deg=deg, pw=2, robust=False)
        #try: plt.close(self.fitWindow.fig)
        #except: pass
        self.fitWindow = fitWindow(self)
        self.fitWindow.show()
        self.fit_deg = deg
        #try: plt.close(self.fitWindow.figure)
        #except: pass
        #self.fitWindow.add_canvas()
        #self.fitWindow.setLayout(self.fitWindow.Layout_fit)
        #self.fitWindow._draw_wvfit()
        #self.fitWindow._draw_residual()
        self._save_fit_resault()
        #self._wave_template()
        pass

    def _save_fit_resault(self):
        self.savedata = self.arcdic.copy()
        self.savedata['wave_solu'] = self.wave_solus
        self.savedata['tab_lines'] = self.tab_line_all
        self.savedata['nlines'] = np.sum(self.tab_line_all['maskgood'])
        self.savedata['deg'] = (self.fit_deg, 'The degree of the 1or2D polynomial')
        if slef.autofind_bool:
            self.savedata['wave_int'] = self.waves_init
        tab = self.tab_line_all
        ind = tab['maskgood'] == 1
        rms = np.std(tab[ind]['wv_fit'] - tab[ind]['line'])
        self.savedata['rms'] = rms
        basename = os.path.basename(self._arcfile)
        name = os.path.splitext(basename)[0]
        #name = basename.replace('.dump', '') if '.dump' in basename else basename
        fname = os.path.join(self._wd, f'{name}.z')
        print(f'rms = {rms}')
        joblib.dump(self.savedata, fname)

    def _save_template(self):
        templatedic = {"wave": self.wavesolus,
                       "flux": self.fluxes
                      }
        fname = os.path.join(self._wd, f'arc_template_spec.z')
        joblib.dump(templatedic, fname)

    def _autofind(self, waves_init: np.array = None, fluxes: np.array = None, ccf_kernel_width=1.5):
        '''
        automaticlly find lines by using CCF
        npix_chunk : int, optional
        the chunk length (half). The default is 20.
        ccf_kernel_width: float, the default is 1.5
        '''
        from pyexspec.wavecalibrate.findlines import findline, find_lines
        if fluxes is None: fluxes = self.fluxes
        find_line = findline()
        if self.templatefile is not None:
            waves_template, fluxes_template = self.waves_template, self.fluxes_template
            waves_init = np.zeros_like(fluxes)
            xcoord = np.arange(fluxes.shape[1])
            x_template = np.arange(fluxes_template.shape[1])
            for _order, flux in enumerate(fluxes):
                flux_template = fluxes_template[_order]
                wave_template = waves_template[_order]
                shifts, ccf = find_line.calCCF(xcoord, flux, x_template, flux_template, show=False)
                wave_init = find_line.estimate_wave_init(xcoord, find_line.xshift, x_template, wave_template)
                waves_init[_order] = wave_init
        if waves_init is None: waves_init = self.wave_solus
        try: npix_chunk = int(self.lineEdit_npix_chunk.text())
        except:
            npix_chunk = 20
            print(f'  |-Using the default chunk pixels, npix_chunk = {npix_chunk}')
        try: ccf_kernel_width = np.float64(self.lineEdit_ccf_kernel_width.text())
        except:
            ccf_kernel_width = 1.5
            print(f'  |-Using the default width of ccf kernel, ccf_kernel_width = {ccf_kernel_width}')
        try: num_sigma_clip = np.float64(self.lineEdit_num_sigma_clip.text())
        except:
            num_sigma_clip = 0
            print(f'  |-Using the default number of simga clipping (Num*std), num_sigma_clip = {num_sigma_clip}')
        print(f'  |-npix_chunk = {npix_chunk}; ccf_kernel_width = {ccf_kernel_width}; num_sigma_clip = {num_sigma_clip}')
        self.waves_init = waves_init
        tab_auto_lines = find_lines(waves_init, fluxes, self.linelist, npix_chunk=npix_chunk,
                        ccf_kernel_width=ccf_kernel_width, num_sigma_clip=num_sigma_clip)
        tab_maskgood = Table(data=[np.ones(len(tab_auto_lines))], names=['maskgood'])
        tab = hstack([tab_auto_lines, tab_maskgood])
        colnames_old = ['line_x_ccf', 'line_wave_init_ccf', 'line_peakflux']
        colnames_new = ['xcoord', 'wv_fit', 'Fpeak']
        for _i, colname in enumerate(colnames_old):
            if colname in tab.colnames:
                tab.rename_column(colname, colnames_new[_i])
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
        tables = self._split_tab_line_all(tab_line_all=tab)
        self.tab_line_all = vstack(tables)
        self.colnames = tables[0].colnames
        self.tables = tables
        self.tab_auto_lines = tab_auto_lines
        self.autofind_bool = True


class fitWindow(QtWidgets.QWidget):

    def __init__(self, UiWvcalib):
        super().__init__()
        self.UiWvcalib = UiWvcalib
        self.add_canvas()
        self.setLayout(self.Layout_fit)
        self._draw_wvfit()
        self._draw_residual()
        #self.show()

    def add_canvas(self):
        self.setWindowTitle("Fitting Window")
        self.widget_fit = QtWidgets.QWidget()
        self.widget_fit.setObjectName("widget_fit")
        self.Layout_fit = QtWidgets.QVBoxLayout(self.widget_fit)
        self.Layout_fit.setObjectName("Layout_fit")

        # a figure instance to plot on
        self.figure, self.axs = plt.subplots(2,1, sharex=True,
                    gridspec_kw={'height_ratios': [2, 1]})
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
        self.Layout_fit.addWidget(self.toolbar)
        self.Layout_fit.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()
        #if self.UiWvcalib.checkBox_invert_xaxis.checkState().value == 0:
        #   self.axs[0].invert_xaxis()

    def _update_fitWindow(self, UiWvcalib=None):
        self.UiWvcalib = self.UiWvcalib if UiWvcalib is None else UiWvcalib
        self._draw_wvfit()
        self._draw_residual()

    def _draw_wvfit(self, tab=None, ax=None):
        tab = self.UiWvcalib.tab_line_all if tab is None else tab
        ax = self.axs[0] if ax is None else ax
        ax.clear()
        orders = np.unique(tab['order'])
        ax.clear()
        for _order in orders:
            _ind = tab['order'] == _order
            x = tab[_ind]['xcoord']
            y = tab[_ind]['line']
            wvfit = tab[_ind]['wv_fit']
            maskgood = np.array(tab[_ind]['maskgood'], dtype=bool)
            ax.scatter(x[maskgood], y[maskgood], marker='+', color='b', label='Good', lw=1)
            ax.scatter(x[~maskgood], y[~maskgood], marker='x', color='r', label='Bad', lw=1)
            ax.plot(x, wvfit, color='k')
        ax.set_ylabel(r'$\lambda$ (${\rm \AA}$)')
        ax.set_ylabel(r'$\lambda$ (${\rm \AA}$)')
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def _draw_residual(self, table=None, ax=None):
        if table is None: table = self.UiWvcalib.tab_line_all
        ax = self.axs[1] if ax is None else ax
        ax.clear()
        x = table['xcoord']
        y = table['line']- table['wv_fit']
        maskgood = np.array(table['maskgood'], dtype=bool)
        ax.scatter(x[maskgood], y[maskgood], marker='+', lw=1, color='b', label='Good')
        #ax.scatter(x[~maskgood], y[~maskgood], marker='x', lw=1, color='r', label='Bad')
        rms = np.std(y[maskgood])
        ax.text(0.75, 0.9, f'rms={rms:.5f}', transform=ax.transAxes, fontsize=13)
        ax.axhline(y=0, lw=0.8, ls='--', color='k')
        ax.set_ylabel(r'Res. (${\rm \AA}$)')
        ax.set_xlabel('Pixel')
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.draw()

    def onclick(self, event):
        # capture cursor position ===============
        # ref: https://matplotlib.org/stable/users/event_handling.html
        # https://stackoverflow.com/questions/39351388/control-the-mouse-click-event-with-a-subplot-rather-than-a-figure-in-matplotlib
        _tab = self.UiWvcalib.tab_line_order
        if event.inaxes == self.axs[0]:
           tab = self.UiWvcalib.tab_line_all
           x_tmp, y_tmp = event.xdata, event.ydata
           y = tab['wv_fit']
           dx = np.sqrt((tab['xcoord']-x_tmp)**2 +(y-y_tmp)**2)
           ind = np.where(dx == np.min(dx))[0][0]
        elif event.inaxes == self.axs[1]:
           tab = self.UiWvcalib.tab_line_all
           x_tmp, y_tmp = event.xdata, event.ydata
           res = tab['line'] - tab['wv_fit']
           dx = np.sqrt((tab['xcoord']-x_tmp)**2 +(res-y_tmp)**2)
           ind = np.where(dx == np.min(dx))[0][0]
        _tab = tab[ind]
        iorder = int(_tab['order'])
        self.UiWvcalib.tab_line_order = self.UiWvcalib.tables[iorder]
        self.order = iorder
        #self.UiWvcalib._refresh_datatable()
        rowID = np.where(self.UiWvcalib.tab_line_order['xcoord']==_tab['xcoord'])[0][0]
        print(f'iorder={iorder}; rowID = {rowID}')
        self.UiWvcalib.tableWidget_files.selectRow(rowID)
        self.UiWvcalib.order =  iorder
        self.UiWvcalib._refresh_datatable()
        self.UiWvcalib._draw_spec()
        self.UiWvcalib._update_order()
        self.UiWvcalib.tableWidget_files.selectRow(rowID)
        #self.UiWvcalib._draw_scatter_pos(ind_currentRow=rowID)
        #self._draw_scatter_pos()
        #self.UiWvcalib.rowID = rowID

    def _draw_scatter_pos(self, tab=None, ind_currentRow=None, **keywds):
        ind_currentRow = self.UiWvcalib.tableWidget_files.currentRow() if ind_currentRow is None else ind_currentRow
        _tab = self.UiWvcalib.tab_line_order[ind_currentRow] if tab is None else tab
        print("fitWindow Plot order {}: xcoord = {}".format(*_tab['order', 'xcoord']))
        # try to draw it
        self.UiWvcalib.pos_line = [_tab['xcoord'], _tab['Fpeak']]
        print('fitWindow draw_scatter_pos')
        #self._draw_spec(tab=_tab)
        try:
            self.scatter_pos_wvfit.remove()
            self.scatter_pos_resid.remove()
        except: pass
        try:
            self.scatter_pos_wvfit = self.axs[0].scatter(_tab['xcoord'], _tab['line'], marker='o', lw=1, color='g',facecolors='None', **keywds)
            self.scatter_pos_resid = self.axs[1].scatter(_tab['xcoord'], _tab['line'] - _tab['wv_fit'], marker='o', lw=1, color='g',facecolors='None', **keywds)
            self.canvas.draw()
        except:
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # mainWindow = QtWidgets.QMainWindow()
    mainprocess = UiWvcalib()
    # ui.setupUi(mainWindow)
    # ui.initUi(mainWindow)
    mainprocess.show()
    if pyqt5_bool:
        sys.exit(app.exec_())# pyqt5
    else:
        sys.exit(app.exec())# pyqt6
