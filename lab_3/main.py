__author__ = 'boiko'
# coding: utf8

import sys
from PyQt5 import QtCore, QtWidgets, uic

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QTextDocument, QFont, QCursor, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QSplashScreen, QVBoxLayout, QSpacerItem, QSizePolicy, QProgressBar, QLabel
from PyQt5.uic import loadUiType
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from lab_3.presentation import PolynomialBuilder, PolynomialBuilderExpTh
from lab_3.solve import Solve
from lab_3.solve_custom import SolveExpTh, SolveExpTh1
from lab_3.bruteforce import BruteForceWindow

app = QApplication(sys.argv)
app.setApplicationName('SA-3')
form_class, base_class = loadUiType('lab_3/main_window.ui')


class MainWindow(QDialog, form_class):
    # signals:
    input_changed = pyqtSignal('QString')
    output_changed = pyqtSignal('QString')


    def __init__(self, parent =None):
        super(MainWindow, self).__init__(parent)

        # setting up ui
        self.setupUi(self)

        # other initializations
        self.dimensions = [self.x1_dim.value(), self.x2_dim.value(),
                                    self.x3_dim.value(), self.y_dim.value()]
        self.degrees = [self.x1_deg.value(), self.x2_deg.value(), self.x3_deg.value()]
        self.type = 'null'
        if self.radio_sh_cheb.isChecked():
            #self.type = 'sh_cheb_doubled'
            self.type = 'legendre'
        elif self.radio_cheb.isChecked():
            self.type = 'sh_legendre'
            #self.type = 'cheb'
        elif self.radio_sh_cheb_2.isChecked():
            #self.type = 'sh_cheb_2'
            self.type = 'sh_cheb_2'
        self.custom_func_struct = self.custom_check.isChecked()
        self.input_path = self.line_input.text()
        self.output_path = self.line_output.text()
        self.samples_num = self.sample_spin.value()
        self.lambda_multiblock = self.lambda_check.isChecked()
        self.weight_method = self.weights_box.currentText().lower()
        self.solution = None
        doc = self.results_field.document()
        assert isinstance(doc, QTextDocument)
        font = doc.defaultFont()
        assert isinstance(font, QFont)
        font.setFamily('Courier New')
        font.setPixelSize(12)
        doc.setDefaultFont(font)
        self.plotWidget = None
        self.lay = QtWidgets.QVBoxLayout(self.content_plot)  
        self.addToolBar = None
        return

    @staticmethod
    def launch():
        dialog = MainWindow()
        dialog.setWindowTitle("Lab_3")
        dialog.show()

    @pyqtSlot()
    def input_clicked(self):
        filename = QFileDialog.getOpenFileName(self, 'Open data file', '.', 'Data file (*.txt *.dat)')[0]
        if filename == '':
            return
        if filename != self.input_path:
            self.input_path = filename
            self.input_changed.emit(filename)
        return

    @pyqtSlot('QString')
    def input_modified(self, value):
        if value != self.input_path:
            self.input_path = value
        return

    @pyqtSlot()
    def output_clicked(self):
        filename = QFileDialog.getSaveFileName(self, 'Save data file', '.', 'Spreadsheet (*.xlsx)')[0]
        if filename == '':
            return
        if filename != self.output_path:
            self.output_path = filename
            self.output_changed.emit(filename)
        return

    @pyqtSlot('QString')
    def output_modified(self, value):
        if value != self.output_path:
            self.output_path = value
        return

    @pyqtSlot(int)
    def samples_modified(self, value):
        self.samples_num = value
        return

    @pyqtSlot(int)
    def dimension_modified(self, value):
        sender = self.sender().objectName()
        if sender == 'x1_dim':
            self.dimensions[0] = value
        elif sender == 'x2_dim':
            self.dimensions[1] = value
        elif sender == 'x3_dim':
            self.dimensions[2] = value
        elif sender == 'y_dim':
            self.dimensions[3] = value
        return

    @pyqtSlot(int)
    def degree_modified(self, value):
        sender = self.sender().objectName()
        if sender == 'x1_deg':
            self.degrees[0] = value
        elif sender == 'x2_deg':
            self.degrees[1] = value
        elif sender == 'x3_deg':
            self.degrees[2] = value
        return

    @pyqtSlot(bool)
    def type_modified(self, isdown):
        if (isdown):
            sender = self.sender().objectName()
            if sender == 'radio_sh_cheb':
                #self.type = 'sh_cheb_doubled'
                self.type = 'legendre'
            elif sender == 'radio_cheb':
                self.type = 'sh_legendre'
            elif sender == 'radio_sh_cheb_2':
                self.type = 'sh_cheb_2'
        return


    @pyqtSlot(bool)
    def structure_changed(self, isdown):
        self.custom_func_struct = isdown


    @pyqtSlot()
    def click_next(self):
        if self.solution:
            try:
                fig = self.solution.plot_graphs(poly_type=self.type)
                for i in reversed(range(self.lay.count())): 
                    self.lay.itemAt(i).widget().setParent(None)
                self.plotWidget = FigureCanvas(fig)
                self.lay.setContentsMargins(0, 0, 0, 0)      
                self.lay.addWidget(self.plotWidget)
                self.lay.addWidget(NavigationToolbar(self.plotWidget, self))


            except Exception as e:
                QMessageBox.warning(self,'Error!','Error happened during plotting: ' + str(e))
        return


    @pyqtSlot()
    def plot_clicked(self):
        if self.solution:
            arima_st = self.predictBox.value()
            try:
                if arima_st > 0:
                    self.solution.plot_graphs_with_prediction(arima_st)
                else:
                    self.solution.plot_graphs()
                    fig = self.solution.plot_graphs(0, poly_type=self.type)
                    for i in reversed(range(self.lay.count())): 
                        self.lay.itemAt(i).widget().setParent(None)
                    self.plotWidget = FigureCanvas(fig)
                    self.lay.setContentsMargins(0, 0, 0, 0)      
                    self.lay.addWidget(self.plotWidget)
                    self.lay.addWidget(NavigationToolbar(self.plotWidget, self))
            except Exception as e:
                QMessageBox.warning(self,'Error!','Error happened during plotting: ' + str(e))
        return

    @pyqtSlot()
    def exec_clicked(self):
        self.exec_button.setEnabled(False)
        try:
            if self.custom_func_struct:
                solver = SolveExpTh(self._get_params())
                solver.prepare()
                self.solution = PolynomialBuilderExpTh(solver)
                self.results_field.setText(solver.show()+'\n\n'+self.solution.get_results())
            else:
                solver = Solve(self._get_params())
                solver.prepare()
                self.solution = PolynomialBuilder(solver)
                self.results_field.setText(solver.show()+'\n\n'+self.solution.get_results())
        except Exception as e:
            QMessageBox.warning(self,'Error!','Error happened during execution: ' + str(e))
        self.exec_button.setEnabled(True)
        return

    @pyqtSlot()
    def bruteforce_called(self):
        BruteForceWindow.launch(self)
        return

    @pyqtSlot(int, int, int)
    def update_degrees(self, x1_deg, x2_deg, x3_deg):
        self.x1_deg.setValue(x1_deg)
        self.x2_deg.setValue(x2_deg)
        self.x3_deg.setValue(x3_deg)
        return

    @pyqtSlot(bool)
    def lambda_calc_method_changed(self, isdown):
        self.lambda_multiblock = isdown
        return

    @pyqtSlot('QString')
    def weights_modified(self, value):
        self.weight_method = value.lower()
        return

    def _get_params(self):
        return dict(poly_type=self.type, degrees=self.degrees, dimensions=self.dimensions,
                    samples=self.samples_num, input_file=self.input_path, output_file=self.output_path,
                    weights=self.weight_method, lambda_multiblock=self.lambda_multiblock)


# -----------------------------------------------------#
'''
form = MainWindow()
form.setWindowTitle('System Analysis - Lab 3')
form.show()
sys.exit(app.exec_())
'''
