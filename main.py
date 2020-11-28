import matplotlib
import sys
from PyQt5 import QtCore, QtWidgets, uic

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QTextDocument, QFont, QCursor, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QSplashScreen, QVBoxLayout, QSpacerItem, QSizePolicy, QProgressBar, QLabel
from PyQt5.uic import loadUiType

from lab_3.main import MainWindow
from lab_2.main import MainWindow_2


app = QApplication(sys.argv)
app.setApplicationName('SA-3')
form_class, base_class = loadUiType('main_window.ui')

matplotlib.use("Qt5Agg", force=True)
__author__ = 'boiko'

class StartWindow(QDialog, form_class):
	def __init__(self, *args):
		super(StartWindow, self).__init__(*args)
		self.setupUi(self)
		return


	@pyqtSlot()
	def show3lab(self):
		self.hide()
		dialog = MainWindow(parent=self)
		if dialog.exec():
		    pass # do stuff on success
		self.show()
		#MainWindow.launch(self)
		return

	@pyqtSlot()
	def show2lab(self):
		self.hide()
		dialog = MainWindow_2(parent=self)
		if dialog.exec():
		    pass # do stuff on success
		self.show()
		return

	@pyqtSlot()
	def show4lab(self):
		'''
		self.close()
		self.twoWindow = MainWindow()
		self.twoWindow.setWindowTitle("Лабороторная 3 Бойко Зинченко Худиков")
		self.twoWindow.show()
		'''
		return

	@staticmethod
	def launch():
		self.dialog = StartWindow()
		self.dialog.setWindowTitle("Вибери програму")
		return self.dialog




if __name__ == '__main__':
	form = StartWindow()
	form.show()
	sys.exit(app.exec_())
