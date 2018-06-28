from threading import Thread
import time
import logging
import functools
import IPython
import itertools
import collections

import pyqtgraph as pg
import pyqtgraph.parametertree
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import pandas as pd

from qtune.history import History, plot_parameters, plot_gradients

IPython.get_ipython().magic('gui qt')
IPython.get_ipython().magic('matplotlib qt')


def log_exceptions(channel='qtune', catch_exceptions=True):
    def logging_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logging.getLogger(channel).exception("Error in %s" % func.__name__)
                if not catch_exceptions:
                    raise
        return wrapper
    return logging_decorator


class FunctionHandler(logging.Handler):
    def __init__(self, func, level=logging.NOTSET):
        super().__init__(level=level)
        self._function = func

    def emit(self, record):
        self._function(self.format(record))


class LogLevelSelecter(QtCore.QObject):
    @QtCore.pyqtSlot(int)
    def select(self, level_index):
        log_level = self.gui.log_level.itemData(level_index)
        self.logger.setLevel(log_level)

    def __init__(self, gui, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gui = gui
        self.logger = logger


class PlotOrganizer(pg.LayoutWidget):
    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        (_, gates), (_, params), (_, grads) = self.parameter.getValues().values()
        pens = (pg.intColor(idx, len(gates) + len(params)) for idx in itertools.count(0))
        plot_item = self.plot.getPlotItem()

        for pen, (gate, (plot_gate, _)) in zip(pens, gates.items()):
            if plot_gate:
                data = self.history.get_gate_values(gate)

                if gate in self._plots:
                    self._plots[gate].setData(data)
                else:
                    self._plots[gate] = plot_item.plot(data, name=gate, pen=pen)

            else:
                self._remove_plot(gate)

        for pen, (param, (plot_param, _)) in zip(pens, params.items()):
            if plot_param.startswith('plot'):
                data = self.history.get_parameter_values(param)
                if param in self._plots:
                    self._plots[param].setData(data)
                else:
                    self._plots[param] = plot_item.plot(data, name=param, pen=pen)

            else:
                self._remove_plot(param)

            error_plot_name = param + "#err_bar"
            if plot_param.endswith('error'):
                data_std = self.history.get_parameter_std(param)
                error_plot_args = dict(x=data.index.values, y=data,
                                       height=data_std)

                if error_plot_name in self._plots:
                    self._plots[error_plot_name].setData(**error_plot_args)
                else:
                    self._plots[error_plot_name] = pg.ErrorBarItem(**error_plot_args)
                    plot_item.addItem(self._plots[error_plot_name])

            else:
                self._remove_plot(error_plot_name)

        for param, (_, gates) in grads.items():
            data = self.history.get_gradients(param)
            data_var = self.history.get_gradient_variances(param)

            for pen, (gate, (plot_grad, _)) in zip(pens, gates.items()):
                grad_name = param + '#' + gate
                grad_err_name = grad_name + '#err_bar'

                if plot_grad.startswith('plot'):
                    gate_data = data[gate]

                    if grad_name in self._plots:
                        self._plots[grad_name].setData(gate_data)

                    else:
                        self._plots[grad_name] = plot_item.plot(gate_data, name=grad_name, pen=pen)

                else:
                    self._remove_plot(grad_name)

                if plot_grad.endswith('error') and gate in data_var:
                    gate_data_var = data_var[gate]
                    error_plot_args = dict(x=gate_data.index.values, y=gate_data,
                                           height=gate_data_var.apply(np.sqrt))

                    if grad_err_name in self._plots:
                        self._plots[grad_err_name].setData(**error_plot_args)
                    else:
                        self._plots[grad_err_name] = pg.ErrorBarItem(**error_plot_args)
                        plot_item.addItem(self._plots[grad_err_name])

                else:
                    self._remove_plot(grad_err_name)

    def _remove_plot(self, name):
        if name in self._plots:
            plot_item = self.plot.getPlotItem()
            plot_item.legend.removeItem(name)
            plot_item.removeItem(self._plots.pop(name))

    @log_exceptions('plotting')
    def plot_selection_change(self, _, changes):
        for param, change, plot_activated in changes:
            for child in param.children():
                child.setValue(plot_activated)
        self.refresh()

    def __init__(self, history: History, **kwargs):
        super().__init__(**kwargs)

        gates = {'name': 'Gate Voltages', 'type': 'group', 'children': [
            {'name': name, 'type': 'bool', 'value': False} for name in history.gate_names
        ]}

        params = {'name': 'Parameters', 'type': 'group', 'children': [
            {'name': name, 'type': 'list', 'values': ['', 'plot', 'plot + error'], 'value': ''}
            for name in history.parameter_names
        ]}

        grads = {'name': 'Gradients', 'type': 'group', 'children': [
            {'name': parameter_name, 'type': 'list', 'values': ['', 'plot', 'plot + error'], 'value': '',
             'expanded': False, 'children': [
                {'name': gate_name, 'type': 'list', 'values': ['', 'plot', 'plot + error'], 'value': ''}
                for gate_name in gates
            ]} for parameter_name, gates in history.gradient_controlled_parameter_names.items()
        ]}

        p = pg.parametertree.Parameter(name='entries', type='group', children=[gates, params, grads])

        p.sigTreeStateChanged.connect(self.plot_selection_change)

        refresh_btn = QtWidgets.QPushButton('Refresh')
        refresh_btn.setFixedWidth(400)
        refresh_btn.clicked.connect(self.refresh)

        self._color_counter = iter(itertools.cycle(range(100)))

        self.addWidget(refresh_btn, 0, 0)

        self.tree = pg.parametertree.ParameterTree()
        self.tree.setParameters(p, showTop=False)
        self.addWidget(self.tree, 1, 0)

        self.plot = pg.PlotWidget()
        self._plots = dict()

        self.addWidget(self.plot, 1, 1)

        self.parameter = p

        self.history = history

        self.plot.getPlotItem().addLegend()


class ParameterWidget(pg.LayoutWidget):
    def __init__(self,
                 history: History, parameter_names=None,
                 parent=None):
        super().__init__(parent=parent)

        self.matplot = MatplotlibWidget()
        self.history = history
        self.parameter_names = parameter_names

        self.addWidget(self.matplot, 0, 0)
        self.refresh()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        if self.parameter_names is None:
            parameter_names = self.history.parameter_names
        else:
            parameter_names = self.parameter_names

        parameter_values = pd.DataFrame(collections.OrderedDict(
            (parameter_name, self.history.get_parameter_values(parameter_name))
            for parameter_name in parameter_names
        ))

        parameter_std = pd.DataFrame({par_name: self.history.get_parameter_std(parameter_name=par_name)
                                      for par_name in parameter_names})

        if len(self.matplot.getFigure().axes) == len(parameter_names):
            for ax in self.matplot.getFigure().axes:
                ax.clear()
            axes = self.matplot.getFigure().axes
        else:
            self.matplot.getFigure().clear()
            axes = self.matplot.getFigure().subplots(nrows=len(parameter_names))

        plot_parameters(parameter_values, parameter_std, axes=axes)
        self.matplot.draw()


class GradientWidget(pg.LayoutWidget):
    def __init__(self,
                 history: History, parameter_names=None,
                 parent=None):
        super().__init__(parent=parent)

        self.matplot = MatplotlibWidget()
        self.history = history
        self.parameter_names = parameter_names

        self.addWidget(self.matplot, 0, 0)
        self.refresh()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        if self.parameter_names is None:
            parameter_names = self.history.parameter_names
        else:
            parameter_names = self.parameter_names

        gradients = {par_name: self.history.get_gradients(parameter_name=par_name)
                     for par_name in parameter_names}

        gradient_variances = {par_name: self.history.get_gradient_variances(parameter_name=par_name)
                              for par_name in parameter_names}

        if len(self.matplot.getFigure().axes) == len(parameter_names):
            for ax in self.matplot.getFigure().axes:
                ax.clear()
            axes = self.matplot.getFigure().axes
        else:
            self.matplot.getFigure().clear()
            axes = self.matplot.getFigure().subplots(nrows=len(parameter_names))

        plot_gradients(gradients, gradient_variances, axes=axes)
        self.matplot.draw()


class GUI(QtWidgets.QMainWindow):
    _log_signal = QtCore.pyqtSignal(str)
    _update_plots = QtCore.pyqtSignal()

    def __init__(self, auto_tuner, history, logger='qtune'):
        super().__init__()
        self._auto_tuner = auto_tuner
        self._history = history

        self.setWindowTitle('QTune: Main Window')
        self.resize(1000, 500)

        start_btn = QtWidgets.QPushButton('Start')
        start_btn.setFixedWidth(100)
        start_btn.clicked.connect(self.start)

        stop_btn = QtWidgets.QPushButton('Pause')
        stop_btn.setFixedWidth(100)
        stop_btn.clicked.connect(self.pause)
        stop_btn.setEnabled(False)

        step_btn = QtWidgets.QPushButton('Step')
        step_btn.setFixedWidth(100)
        step_btn.clicked.connect(self.step)

        log_level = QtWidgets.QComboBox()
        for level in (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG, logging.NOTSET):
            log_level.addItem(logging._levelToName[level], level)
        log_level.setCurrentIndex(log_level.count() - 1)
        log_level.setFixedWidth(100)

        log_label = QtWidgets.QLabel('Log level')
        log_label.setFixedWidth(100)

        log = QtWidgets.QTextEdit()
        log.setReadOnly(True)
        log.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse)

        auto_set_btn = QtWidgets.QPushButton("Autoset")
        auto_set_btn.setFixedWidth(100)
        auto_set_btn.setToolTip("You just need to press Autoset")

        plot_organizer_btn = QtWidgets.QPushButton("Real Time Plot")
        plot_organizer_btn.setFixedWidth(100)
        plot_organizer_btn.clicked.connect(self.spawn_plot_window)

        parameter_window_btn = QtWidgets.QPushButton("Parameter Plot")
        parameter_window_btn.setFixedWidth(100)
        parameter_window_btn.clicked.connect(self.spawn_parameter_window)

        gradient_window_btn = QtWidgets.QPushButton("Gradient Plot")
        gradient_window_btn.setFixedWidth(100)
        gradient_window_btn.clicked.connect(self.spawn_gradient_window)

        top = pg.LayoutWidget()
        top.addWidget(start_btn, 0, 0)
        top.addWidget(step_btn, 1, 0)
        top.addWidget(stop_btn, 2, 0)

        top.addWidget(auto_set_btn, 0, 1)
        top.addWidget(log_label, 1, 1)
        top.addWidget(log_level, 2, 1)

        top.addWidget(plot_organizer_btn, 0, 2)
        top.addWidget(parameter_window_btn, 1, 2)
        top.addWidget(gradient_window_btn, 2, 2)

        left = pg.LayoutWidget()
        left.addWidget(top, 0, 0)
        left.addWidget(log, 1, 0)

        self.setCentralWidget(left)

        self._log = log
        self._log_signal.connect(self._log.append)
        self.log_level = log_level

        self._start_btn = start_btn
        self._stop_btn = stop_btn
        self._step_btn = step_btn

        self._thread = Thread(target=self._work)
        self._continuous = False
        self._stepped = False
        self._stop = False
        self._thread.start()

        self._logger = logging.getLogger(logger)

        self._plot_organizer = []
        self._parameter_windows = []
        self._gradient_windows = []

    def spawn_plot_window(self):
        new_window = QtWidgets.QMainWindow(parent=self)
        plot_organizer = PlotOrganizer(history=self._history, parent=new_window)
        new_window.setCentralWidget(plot_organizer)
        self._plot_organizer.append(plot_organizer)

        new_window.setWindowTitle("QTune: Real Time Plot")

        new_window.show()
        new_window.raise_()
        self._update_plots.connect(plot_organizer.refresh)

    def spawn_parameter_window(self):
        new_window = QtWidgets.QMainWindow()
        parameter_widget = ParameterWidget(self.history, parent=new_window)

        new_window.setCentralWidget(parameter_widget)

        self._update_plots.connect(parameter_widget.refresh)
        self._parameter_windows.append(new_window)

        new_window.setWindowTitle("QTune: Parameter Plot")

        new_window.show()
        new_window.raise_()

    def spawn_gradient_window(self):
        new_window = QtWidgets.QMainWindow()
        parameter_widget = GradientWidget(self.history, parent=new_window)

        new_window.setCentralWidget(parameter_widget)

        self._update_plots.connect(parameter_widget.refresh)
        self._gradient_windows.append(new_window)

        new_window.setWindowTitle("QTune: Gradient Plot")

        new_window.show()
        new_window.raise_()

    @property
    def auto_tuner(self):
        return self._auto_tuner

    @property
    def history(self):
        return self._history

    def _join_worker(self):
        if self._thread.is_alive():
            self._logger.info('Joining thread')
        self._stop = True
        self._thread.join()

    def restart_thread(self):
        self._join_worker()
        self._logger.info('Restarting thread')
        self._thread = Thread(target=self._work)
        self._continuous = False
        self._stepped = False
        self._stop = False
        self._thread.start()

    def _work(self):
        while not self._stop:
            while not (self._stop or self._continuous or self._stepped):
                time.sleep(0.05)

            while self._continuous or self._stepped and not self._stop:
                if self.auto_tuner:
                    try:

                        if self.auto_tuner.is_tuning_complete():
                            self._logger.info('Tuning completed')
                            self.pause()
                        else:
                            self.auto_tuner.iterate()
                            self._history.append_autotuner(self._auto_tuner)
                            self._update_plots.emit()

                    except Exception:
                        self._logger.exception('Error during auto tuner iteration: Pausing...')
                        self.pause()
                else:
                    self._logger.error('No Autotuner: Pausing...')
                    self.pause()

                self._stepped = False

    def log(self, msg: str):
        self._log_signal.emit(msg)

    def start(self):
        if not self._thread.is_alive():
            self._logger.warning('Worker thread is dead. Restarting...')
            self.restart_thread()

        elif self._continuous:
            self._logger.info('Already started')

        else:
            self._logger.info('Starting worker thread')

        self._continuous = True
        self._start_btn.setEnabled(False)
        self._step_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def step(self):
        if not self._thread.is_alive():
            self._logger.warning('Worker thread is dead. Restarting...')
            self.restart_thread()

        self._logger.info('Order worker to do one step')
        self._stepped = True

    def pause(self):
        if not self._thread.is_alive():
            self._logger.warning('Worker thread already dead')
        elif self._continuous:
            self._logger.info('Pausing worker thread')
        elif not self._stepped:
            self._logger.info('Already paused')

        self._continuous = False
        self._stepped = False
        self._start_btn.setEnabled(True)
        self._step_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def close(self):
        self._stop = True
        self._thread.join()
        super().close()

    def __del__(self):
        self.close()

    def configure_logging(self, logger='qtune'):
        if isinstance(logger, str):
            logger = logging.getLogger(logger)

        self.log_level.currentIndexChanged.connect(LogLevelSelecter(self, logger, parent=self).select)

        while logger.hasHandlers():
            to_rm = next(iter(logger.handlers))
            logger.removeHandler(to_rm)

        handler = FunctionHandler(self.log)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)


def setup_default_gui(auto_tuner, history=None):
    if history is None:
        history = History(None)
        history.append_autotuner(auto_tuner)

    gui = GUI(auto_tuner, history)
    gui.configure_logging('plotting')
    gui.configure_logging('qtune')
    gui.log_level.setCurrentIndex(3)
    gui.show()

    gui.raise_()

    return gui
