import matplotlib
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



matplotlib.use("Qt5Agg", force=True)
__author__ = 'vlad'

if __name__ == '__main__':
    number = None
    if not number:
        number = input('Which lab to launch => ')
    filename = 'lab_{0}/main.py'.format(number)
    # Python 3 version
    with open(filename,'r') as f:
       code = compile(f.read(), filename, 'exec')
       exec(code)
    #execfile(filename)
