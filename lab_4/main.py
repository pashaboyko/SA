__author__ = 'boiko'

import matplotlib
from PyQt5.uic import loadUiType

matplotlib.use("Qt5Agg", force=True)

import sys
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox


from lab_4.solver_manager import *
from lab_4.begin import BruteForceWindow




app = QApplication(sys.argv)
app.setApplicationName('Реанимобиль 2')
form_class, base_class = loadUiType('lab_4/main_window_2.ui')


class MainWindow(QDialog, form_class):
    # signals:
    input_changed = pyqtSignal('QString')
    output_changed = pyqtSignal('QString')


    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)



        # setting up ui
        self.setupUi(self)

        # other initializations
        self.dimensions = [self.x1_dim.value(), self.x2_dim.value(),
                                    self.x3_dim.value(), self.y_dim.value()]
        self.degrees = [self.x1_deg.value(), self.x2_deg.value(), self.x3_deg.value()]
        self.remove = self.remove_old.isChecked()
        self.status_bar = None
        self.type = 'null'
        self.timer = None
        self.setWindowTitle('Діагностування складних технічних систем')
        if self.radio_sh_cheb.isChecked():
            self.type = 'sh_cheb_doubled'
        elif self.radio_cheb.isChecked():
            self.type = 'cheb'
        elif self.radio_sh_cheb_2.isChecked():
            self.type = 'sh_cheb_2'
        self.custom_func_struct = self.custom_check.isChecked()
        self.input_path = self.line_input.text()
        if len(self.input_path) == 0:
            self.input_path = os_path
        self.output_path = './output.xlsx'
        self.samples_num = self.sample_spin.value()
        self.lambda_multiblock = self.lambda_check.isChecked()
        self.weight_method = self.weights_box.currentText().lower()
        self.lambda_multiblock = False
        self.weight_method = "average"
        self.manager = None


        #set tablewidget
        self.tablewidget.verticalHeader().hide()
        self.tablewidget.setRowCount(0)


        column_size = [60, 70, 100, 100,200, 60, 200, 80]
        for index, size in enumerate(column_size):
             self.tablewidget.setColumnWidth(index,size)
        return

    @pyqtSlot()
    def input_clicked(self):
        filename = QFileDialog.getOpenFileName(self, 'Відкрити данні', '.', 'Data file (*.xlsx)')[0]
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
                self.type = 'sh_cheb_doubled'
            elif sender == 'radio_cheb':
                self.type = 'cheb'
            elif sender == 'radio_sh_cheb_2':
                self.type = 'sh_cheb_2'
        return

    @pyqtSlot(bool)
    def structure_changed(self, isdown):
        self.custom_func_struct = isdown

    @pyqtSlot()
    def plot_clicked(self):
        if self.manager:
            try:
                self.manager.plot(self.predictBox.value())
            except Exception as e:
                QMessageBox.warning(self,'Error!','Error happened during plotting: ' + str(e))
        return

    @pyqtSlot()
    def exec_clicked(self):
        self.exec_button.setEnabled(False)
        try:
            self.tablewidget.setRowCount(self.predictBox.value())
            self.manager = SolverManager(self._get_params(), self)
            self.manager.prepare(self.input_path)
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

    def initial_graphics_fill(self, real_values, predicted_values, risk_values, time_ticks):
        for i, graph in enumerate(self.graphs):
            graph.compute_initial_figure(real_values.T[i], predicted_values[i], risk_values[i], time_ticks)

    def update_graphics(self, real_value, predicted_values, risk_values, forecast_ticks):
        for i, graph in enumerate(self.graphs):
            # print(real_value[i], risk_values[i])
            graph.update_figure(real_value[i], predicted_values[i], risk_values[i], forecast_ticks)

    def closeEvent(self, event):
        if self.timer and self.timer.isActive():
            self.timer.stop()
            self.timer.disconnect()
            self.timer.deleteLater()
        super(QDialog, self).closeEvent(event)

    @pyqtSlot()
    def manipulate_timer(self):
        if not self.timer:
            print(1)
            self.start_button.setText('ПАУЗА')
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.execute_iteration)
            self.timer.start(50)
        elif self.timer.isActive():
            self.start_button.setText('ПРОДОВЖИТИ')
            self.timer.stop()
        else:
            self.start_button.setText('ПАУЗА')
            self.timer.start()

    @pyqtSlot()
    def execute_iteration(self):
        self.engine.launch()

    def _get_params(self):
        return dict(custom_struct=self.custom_func_struct,poly_type=self.type, degrees=self.degrees,
                    dimensions=self.dimensions,
                    samples=self.samples_num, output_file=self.output_path,
                    weights=self.weight_method, lambda_multiblock=self.lambda_multiblock,
                    pred_steps = self.predictBox.value(), tablewidget = self.tablewidget, \
                    lbl = {'rmr':self.lbl_rmr, 'time': self.lbl_time, 'y1': self.lbl_y1,\
                           'y2':self.lbl_y2, 'y3':self.lbl_y3})



