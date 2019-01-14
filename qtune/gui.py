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
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import numpy as np
import pandas as pd

from qtune.history import History, plot_parameters, plot_gradients
import qtune.autotuner

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


class EvaluatorWidget(pg.LayoutWidget):
    def __init__(self, history: History, logger, **kwargs):
        super().__init__(**kwargs)
        self._tune_run_number = 0
        self._logger = logger
        self.history = history
        self.bottom = None
        self.refresh()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        top = pg.LayoutWidget()

        for i, evaluator_name in enumerate(self.history.evaluator_names):
            btn = QtWidgets.QPushButton(evaluator_name)
            btn.setFixedWidth(250)
            top.addWidget(btn, i // 2, i % 2)
            btn.clicked.connect(self.plot_raw_data)

        tune_run_label = QtWidgets.QLabel('Tune run Number')
        tune_run_label.setFixedWidth(150)
        top.addWidget(tune_run_label, 0, 2)
        self.tune_run_line = QtWidgets.QLineEdit()
        self.tune_run_line.setText('0')
        self.tune_run_line.setValidator(QtGui.QIntValidator())
        self.tune_run_line.editingFinished.connect(self.refreh_tune_run_number)
        top.addWidget(self.tune_run_line, 1, 2)
        self.addWidget(top, 0, 0)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def plot_raw_data(self):
        self.bottom = MeasurementDataWidget(evaluator_name=self.sender().text(),
                                            tune_run_number=self.tune_run_number,
                                            history=self.history)
        self.addWidget(self.bottom, 1, 0)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refreh_tune_run_number(self):
        i = self.tune_run_line.text()
        self.tune_run_number = i
        self.tune_run_line.setText(str(self.tune_run_number))

    @property
    def logger(self):
        return logging.getLogger(self._logger)

    @property
    def tune_run_number(self):
        return self._tune_run_number

    @tune_run_number.setter
    def tune_run_number(self, i: int):
        try:
            i = int(i)
        except ValueError:
            i = 0
            self.logger.warning("The tune run number must be integer!")
        if i < 0:
            i = 0
        elif i >= self.history.number_of_stored_iterations:
            i = self.history.number_of_stored_iterations - 1
        self._tune_run_number = i


class PlotOrganizer(pg.LayoutWidget):
    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        (_, gates), (_, params), (_, grads) = self.parameter.getValues().values()
        pens = (pg.intColor(idx, max(1, len(gates) + len(params))) for idx in itertools.count(0))
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
                                       height=data_std * 2)
                # QT graph is plotting only half the height to each side. Our errors are 2 sided

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
                                           height=gate_data_var.apply(np.sqrt) * 2)
                    # QT graph is plotting only half the height to each side. Our errors are 2 sided

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


class QTuneMatplotWidget(MatplotlibWidget):
    def __init__(self,
                 history: History, parameter_names=None,
                 parent=None, **kwargs):
        super().__init__(**kwargs)

        if parent:
            self.setParent(parent)

        self.history = history
        self._parameter_names = parameter_names

        self.vbox.setContentsMargins(0, 0, 0, 0)

        self.refresh()

    def refresh(self):
        raise NotImplementedError()

    def get_clear_axes(self, number_rows=None):
        if number_rows is None:
            number_rows = len(self.parameter_names)
        if len(self.getFigure().axes) == number_rows:
            for ax in self.getFigure().axes:
                ax.clear()
            return self.getFigure().axes
        else:
            self.getFigure().clear()
            return self.getFigure().subplots(nrows=number_rows)

    @property
    def parameter_names(self):
        return self.history.parameter_names if self._parameter_names is None else self._parameter_names


class ParameterWidget(QTuneMatplotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        parameter_values = pd.DataFrame(collections.OrderedDict(
            (parameter_name, self.history.get_parameter_values(parameter_name))
            for parameter_name in self.parameter_names
        ))

        parameter_std = pd.DataFrame({par_name: self.history.get_parameter_std(parameter_name=par_name)
                                      for par_name in self.parameter_names})

        axes = self.get_clear_axes()

        plot_parameters(parameter_values, parameter_std, axes=axes)
        self.draw()


class GradientWidget(QTuneMatplotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        gradients = {par_name: self.history.get_gradients(parameter_name=par_name)
                     for par_name in self.parameter_names}

        gradient_variances = {par_name: self.history.get_gradient_variances(parameter_name=par_name)
                              for par_name in self.parameter_names}

        axes = self.get_clear_axes()

        plot_gradients(gradients, gradient_variances, axes=axes)
        self.draw()


class MeasurementDataWidget(QTuneMatplotWidget):
    def __init__(self, evaluator_name, tune_run_number, history, *args, **kwargs):
        self.evaluator_name = evaluator_name
        self.tune_run_number = tune_run_number
        self.history = history
        super().__init__(history=history, *args, **kwargs)
        self.resize(600, 400)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        if self.history.evaluator_data.iloc[self.tune_run_number - 1][self.evaluator_name] != \
                self.history.evaluator_data.iloc[self.tune_run_number - 1][self.evaluator_name]:
            return
        if self.evaluator_name.startswith('Averaging'):
            num_measurements = len(
                self.history.evaluator_data.iloc[self.tune_run_number - 1][self.evaluator_name][
                    'evaluator'].measurements)
        else:
            num_measurements = len(
                self.history.evaluator_data.iloc[self.tune_run_number - 1][self.evaluator_name]['measurements'])

        axes = self.get_clear_axes(number_rows=num_measurements)
        self.history.plot_single_evaluator_data(ax=axes, evaluator_name=self.evaluator_name,
                                                tune_run_number=self.tune_run_number - 1)
        self.getFigure().tight_layout()
        self.draw()


class ReloadWidget(pg.LayoutWidget):
    gui_get_auto_tuner = QtCore.pyqtSignal()

    def __init__(self, history, autotuner, logger='qtune', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = history
        self.auto_tuner = autotuner
        self._logger = logger
        self._tune_run_number = 0
        warning_label = QtWidgets.QLabel('This Window allows you to reload a previous state of the Autotuner!')
        self.addWidget(warning_label, 0, 0)
        load_voltages_button = QtWidgets.QPushButton('Reload Voltages')
        load_voltages_button.setFixedWidth(150)
        load_voltages_button.clicked.connect(self.reload_voltages)
        self.addWidget(load_voltages_button, 1, 0)
        load_state_button = QtWidgets.QPushButton('Reload Entire State!')
        load_state_button.setFixedWidth(150)
        load_state_button.clicked.connect(self.reload_autotuner)
        self.addWidget(load_state_button, 2, 0)
        tune_run_label = QtWidgets.QLabel('Reload tune run number')
        self.addWidget(tune_run_label, 1, 1)
        self.tune_run_line = QtWidgets.QLineEdit()
        self.tune_run_line.setText('1')
        self.tune_run_line.setValidator(QtGui.QIntValidator())
        self.tune_run_line.editingFinished.connect(self.refreh_tune_run_number)
        self.addWidget(self.tune_run_line, 2, 1)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refresh(self):
        pass

    @property
    def tune_run_number(self):
        return self._tune_run_number

    @tune_run_number.setter
    def tune_run_number(self, i: int):
        try:
            i = int(i)
        except ValueError:
            i = 0
            self.logger.warning("The tune run number must be integer!")
        if i < 0:
            i = 0
        elif i >= self.history.number_of_stored_iterations:
            i = self.history.number_of_stored_iterations - 1
        self._tune_run_number = i

    @property
    def logger(self):
        return logging.getLogger(self._logger)

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def refreh_tune_run_number(self):
        i = self.tune_run_line.text()
        self.tune_run_number = i
        self.tune_run_line.setText(str(self.tune_run_number))

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def reload_voltages(self):
        voltages = self.history.get_gate_values(pd.Index(self.history.gate_names)).iloc[self.tune_run_number]
        if self.auto_tuner:
            self.auto_tuner._experiment.set_gate_voltages(new_gate_voltages=voltages)
            self.auto_tuner.restart()
        else:
            self.logger.warning("No Autotuner connected to the History. Voltages could not be reloaded!")

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def reload_autotuner(self):
        if self.auto_tuner:
            reload_file = self.history.get_reload_path(self.tune_run_number)
            new_autotuner = qtune.autotuner.load_auto_tuner(file=reload_file, reserved=self.auto_tuner.reserved)
            self.auto_tuner = new_autotuner
            new_autotuner.restart()
            self.gui_get_auto_tuner.emit()
        else:
            self.logger.warning("No Autotuner connected to the History. Previous state could not be reloaded!")


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

        evaluator_btn = QtWidgets.QPushButton("Raw Data Plot")
        evaluator_btn.setFixedWidth(100)
        evaluator_btn.clicked.connect(self.spawn_evaluator_window)

        reload_btn = QtWidgets.QPushButton("Load")
        reload_btn.setFixedWidth(100)
        reload_btn.clicked.connect(self.spawn_reload_window)

        top = pg.LayoutWidget()
        top.addWidget(start_btn, 0, 0)
        top.addWidget(step_btn, 1, 0)
        top.addWidget(stop_btn, 2, 0)
        top.addWidget(reload_btn, 3, 0)

        top.addWidget(auto_set_btn, 0, 1)
        top.addWidget(log_label, 1, 1)
        top.addWidget(log_level, 2, 1)

        top.addWidget(plot_organizer_btn, 0, 2)
        top.addWidget(parameter_window_btn, 1, 2)
        top.addWidget(gradient_window_btn, 2, 2)
        top.addWidget(evaluator_btn, 3, 2)

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

        self._logger = logger

        self._plot_organizer = []
        self._parameter_windows = []
        self._gradient_windows = []
        self._raw_data_windows = []
        self._reload_windows = []

    @log_exceptions('plotting')
    def _close_child(self, window: QtWidgets.QWidget, event: QtGui.QCloseEvent):
        for child_list in (self._plot_organizer, self._parameter_windows, self._gradient_windows, self._raw_data_windows):
            try:
                child_list.remove(window)
                break
            except ValueError:
                pass
        try:
            self._update_plots.disconnect(lambda x: None)
        except TypeError:
            pass
        window.deleteLater()
        event.accept()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def spawn_plot_window(self):
        plot_organizer = PlotOrganizer(history=self._history, parent=None)
        plot_organizer.setWindowFlags(QtCore.Qt.Window)
        plot_organizer.closeEvent = functools.partial(self._close_child, plot_organizer)

        self._plot_organizer.append(plot_organizer)
        self._update_plots.connect(plot_organizer.refresh)

        plot_organizer.setWindowTitle("QTune: Real Time Plot")

        plot_organizer.show()
        plot_organizer.raise_()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def load_auto_tuner_from_reload_widget(self):
        self._auto_tuner = self.sender().auto_tuner

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def spawn_reload_window(self):
        reload_widget = ReloadWidget(history=self.history, autotuner=self.auto_tuner, logger=self._logger)
        reload_widget.setWindowFlags(QtCore.Qt.Window)
        reload_widget.closeEvent = functools.partial(self._close_child, reload_widget)

        reload_widget.gui_get_auto_tuner.connect(self.load_auto_tuner_from_reload_widget)

        self._update_plots.connect(reload_widget.refresh)
        self._reload_windows.append(reload_widget)

        reload_widget.setWindowTitle("QTune: Reload")

        reload_widget.show()
        reload_widget.raise_()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def spawn_parameter_window(self):
        parameter_widget = ParameterWidget(self.history, parent=None)
        parameter_widget.setWindowFlags(QtCore.Qt.Window)
        parameter_widget.closeEvent = functools.partial(self._close_child, parameter_widget)

        self._update_plots.connect(parameter_widget.refresh)
        self._parameter_windows.append(parameter_widget)

        parameter_widget.setWindowTitle("QTune: Parameter Plot")

        parameter_widget.show()
        parameter_widget.raise_()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def spawn_gradient_window(self):
        parameter_widget = GradientWidget(self.history, parent=None)
        parameter_widget.setWindowFlags(QtCore.Qt.Window)
        parameter_widget.closeEvent = functools.partial(self._close_child, parameter_widget)

        self._update_plots.connect(parameter_widget.refresh)
        self._parameter_windows.append(parameter_widget)

        parameter_widget.setWindowTitle("QTune: Gradient Plot")

        parameter_widget.show()
        parameter_widget.raise_()

    @QtCore.pyqtSlot()
    @log_exceptions('plotting')
    def spawn_evaluator_window(self):
        evaluator_widget = EvaluatorWidget(self.history, logger=self._logger)
        evaluator_widget.setWindowFlags(QtCore.Qt.Window)
        evaluator_widget.closeEvent = functools.partial(self._close_child, evaluator_widget)

        # self._update_plots.connect(evaluator_widget.refresh)
        self._raw_data_windows.append(evaluator_widget)

        evaluator_widget.setWindowTitle("Qtune: Evaluator Plot")

        evaluator_widget.show()
        evaluator_widget.raise_()

    @property
    def auto_tuner(self):
        return self._auto_tuner

    @property
    def history(self):
        return self._history

    @property
    def logger(self):
        return logging.getLogger(self._logger)

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

                            if self.history:
                                self.history.append_autotuner(self._auto_tuner)

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
