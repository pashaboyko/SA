# -*- coding: utf-8 -*-

__author__ = 'lex'

import sys
import numpy as np


from PyQt5 import QtCore, QtWidgets, uic

import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtGui import QTextDocument, QFont
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.uic import loadUiType
from PyQt5 import QtGui
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from output import PolynomialBuilder
from task_solution import Solve

app = QApplication(sys.argv)
app.setApplicationName('SA-2')
form_class, base_class = loadUiType('main_window.ui')


class MainWindow(QDialog, form_class):
    # signals:
    input_changed = pyqtSignal('QString')
    output_changed = pyqtSignal('QString')
    # x1_dim_changed = pyqtSignal(int)
    # x2_dim_changed = pyqtSignal(int)
    # x3_dim_changed = pyqtSignal(int)
    # x1_deg_changed = pyqtSignal(int)
    # x2_deg_changed = pyqtSignal(int)
    # x3_deg_changed = pyqtSignal(int)
    # type_cheb = pyqtSignal()
    # type_lege = pyqtSignal()
    # type_lagg = pyqtSignal()
    # type_herm = pyqtSignal()


    def __init__(self, *args):
        super(MainWindow, self).__init__(*args)

        # setting up ui
        self.setupUi(self)

        self.progress.setValue(0)

        # other initializations
        self.dimensions = [self.x1_dim.value(), self.x2_dim.value(),
                                    self.x3_dim.value(), self.y_dim.value()]
        self.degrees = [self.x1_deg.value(), self.x2_deg.value(), self.x3_deg.value()]
        
        self.method = 'null'
        if self.radioConjucate.isChecked():
            self.method = 'conjucate'
        elif self.radioCoordDesc.isChecked():
            self.method = 'coordDesc'

        self.type = 'null'  
        if self.radio_cheb.isChecked():
            self.type = 'chebyshev'
        elif self.radio_legend.isChecked():
            self.type = 'legendre'
        elif self.radio_lagg.isChecked():
            self.type = 'laguerre'
        elif self.radio_herm.isChecked():
            self.type = 'hermit'

        self.input_path = 'Data/input.txt'
        self.output_path = 'output.xlsx'
        self.samples_num = self.sample_spin.value()
        self.lambda_multiblock = self.lambda_check.isChecked()
        self.weight_method = self.weights_box.currentText().lower()
        self.solution = None
        doc = self.results_field.document()
        assert isinstance(doc, QTextDocument)
        font = doc.defaultFont()
        assert isinstance(font, QFont)
        font.setFamily('Times New Roman')
        font.setPixelSize(14)
        doc.setDefaultFont(font)
        self.plotWidget = None
        self.lay = QtWidgets.QVBoxLayout(self.content_plot)  
        self.addToolBar = None
        return

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
            if sender == 'radio_cheb':
                self.type = 'chebyshev'
            elif sender == 'radio_legend':
                self.type = 'legendre'
            elif sender == 'radio_lagg':
                self.type = 'laguerre'
            elif sender == 'radio_herm':
                self.type = 'hermit'
        return

    @pyqtSlot(bool)
    def method_modified(self, isdown):
        if (isdown):
            sender = self.sender().objectName()
            if sender == 'radioCoordDesc':
                self.method = 'coordDesc'
                print(1123)
            elif sender == 'radioConjucate':
                self.method = 'conjucate'
        return

    @pyqtSlot()
    def click_next(self):
        if self.solution:
            try:
                fig = self.solution.plot_graphs(poly_type=self.type, method=self.method)
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
            try:
                fig = self.solution.plot_graphs(0, poly_type=self.type, method=self.method)
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
        self.progress.setValue(0)
        try:
            solver = Solve(self.__get_params())
            self.progress.setValue(25)
            solver.prepare()
            self.progress.setValue(50)
            self.solution = PolynomialBuilder(solver)
            self.progress.setValue(75)
            self.results_field.setText(solver.show()+'\n\n'+self.solution.get_results())
            self.progress.setValue(100)
        except Exception as e:
            QMessageBox.warning(self,'Error!','Error happened during execution: ' + str(e))
        self.exec_button.setEnabled(True)
        return

    @pyqtSlot(bool)
    def lambda_calc_method_changed(self, isdown):
        self.lambda_multiblock = isdown
        return

    @pyqtSlot('QString')
    def weights_modified(self, value):
        self.weight_method = value.lower()
        return

    def __get_params(self):
        return dict(poly_type=self.type, method=self.method, degrees=self.degrees, dimensions=self.dimensions,
                    samples=self.samples_num, input_file=self.input_path, output_file=self.output_path,
                    weights=self.weight_method, lambda_multiblock=self.lambda_multiblock)


# -----------------------------------------------------#
form = MainWindow()
form.setWindowTitle('System Analysis - Lab2')
form.show()
sys.exit(app.exec_())
