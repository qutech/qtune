from threading import Thread
import time
import logging
import IPython

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


IPython.get_ipython().magic('gui qt')
IPython.get_ipython().magic('matplotlib qt')


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


class GUI(QtWidgets.QMainWindow):
    _log_signal = QtCore.pyqtSignal(str)

    def __init__(self, autotuner, app=None, logger='qtune'):
        super().__init__()
        self.autotuner = autotuner

        self.setWindowTitle('qtune GUI')
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

        log = QtWidgets.QTextEdit()
        log.setReadOnly(True)
        log.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse)

        # this is the top row
        top = pg.LayoutWidget()
        top.addWidget(start_btn, 0, 0)
        top.addWidget(step_btn, 1, 0)
        top.addWidget(stop_btn, 2, 0)
        top.addWidget(QtWidgets.QLabel('Log level'), 0, 1)
        top.addWidget(log_level, 1, 1)

        main = pg.LayoutWidget()
        main.addWidget(top, 0, 0)
        main.addWidget(log, 1, 0)

        self.setCentralWidget(main)

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
                if self.autotuner:
                    try:
                        self.autotuner.iterate()
                    except Exception:
                        self._logger.exception('Error during autotuner iteration: Pausing...')
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
