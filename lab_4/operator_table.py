from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure, Axes
import numpy as np


class DynamicRiskCanvas(FigureCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, parent=None, coordinate=1, description=None, dpi=100, tail=10, warning=0, failure=0,
                 remove_old=True):
        self.coordinate = coordinate
        self.warning_threshold = warning
        self.failure_threshold = failure
        self.tail = tail
        self.remove_old = remove_old
        self.time_separator = None
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.title = description
        self.axes.set_title(self.title, fontsize=10)
        self.real_line, self.predicted_line, self.risk_line = self.axes.plot([], [], 'black', [], [], 'g', [], [], 'g--')
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self, real_values, predicted_values, risk_values, time_ticks):
        self.real_line.set_data(time_ticks[:-self.tail], real_values)
        self.predicted_line.set_data(time_ticks[-self.tail - 1:], np.append(real_values[-1], predicted_values))
        self.risk_line.set_data(time_ticks[-self.tail - 1:], np.append(real_values[-1], risk_values))
        self.axes.axhline(y=self.warning_threshold, color='r', linestyle='dotted')
        self.axes.axhline(y=self.failure_threshold, color='r', linewidth=3)
        self.axes.axhspan(ymin=self.failure_threshold, ymax=self.warning_threshold, alpha=0.2, color='r')
        self.time_separator = self.axes.axvline(x=time_ticks[-self.tail - 1], color='black')
        bot = min(map(min,[real_values, predicted_values, risk_values]))
        top = max(map(max,[real_values, predicted_values, risk_values]))
        bot -= (top - bot)*0.1
        top += (top - bot)*0.1
        self.axes.set_ylim(bot, top)
        self.axes.set_xlim(time_ticks[0], time_ticks[-1])
        self.draw()

    def update_figure(self, real_value, predicted_values, risk_values, time_ticks):
        # retrieve old plot data
        time_head, values_head = self.real_line.get_data()
        # remove very old time point if needed
        if self.remove_old:
            time_head = time_head[1:]
            values_head = values_head[1:]
        # update registered data for current time point
        time_head = np.append(time_head, time_ticks[0])
        values_head = np.append(values_head, real_value)
        # change graphics depending on new data sets
        self.real_line.set_data(time_head, values_head)
        self.predicted_line.set_data(time_ticks, np.append(real_value, predicted_values))
        self.risk_line.set_data(time_ticks, np.append(real_value, risk_values))
        # move current time marker
        self.time_separator.set_xdata([time_head[-1]] * 2)
        # redraw with new limits
        bot = min(map(min,[values_head, predicted_values, risk_values]))
        top = max(map(max,[values_head, predicted_values, risk_values]))
        bot -= (top - bot)*0.1
        top += (top - bot)*0.1
        self.axes.set_ylim(bot, top)
        self.axes.set_xlim(time_head[0], time_ticks[-1])
        self.draw()




        return self
