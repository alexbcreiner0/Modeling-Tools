from PyQt6 import (
    QtCore as qc,
    QtWidgets as qw,
    QtGui as qg
)

class BridgeWorker(qc.QObject):
    """
    Worker that rapidly drains the data queue. Later will be rejobbed into the
    guy which 'assembles' the data from smaller serialized data points.
    """

    progress = qc.pyqtSignal(object, object)
    done = qc.pyqtSignal()
    error = qc.pyqtSignal(object)

    def __init__(self, mp_queue, run_id, parent= None):
        super().__init__(parent)
        self.mp_queue = mp_queue
        self._running = True
        self._drain_timer = None
        self.run_id = run_id

    @qc.pyqtSlot()
    def start(self):
        self._drain_timer = qc.QTimer(self)
        self._drain_timer.setInterval(10)
        self._drain_timer.timeout.connect(self._drain_once)
        self._drain_timer.start()
    
    @qc.pyqtSlot()
    def stop(self):
        if self._drain_timer is not None:
            self._drain_timer.stop()
            self._drain_timer.deleteLater()
            self._drain_timer = None

    @qc.pyqtSlot()
    def _drain_once(self):
        import queue as py_queue
        latest = None
        saw_done = False

        while True:
            try:
                msg = self.mp_queue.get_nowait()
            except py_queue.Empty:
                break

            if not (isinstance(msg, tuple) and msg):
                continue

            if msg[0] != self.run_id:
                continue

            if len(msg) >= 2 and msg[1] == "DONE":
                saw_done = True
                continue

            if len(msg) >= 2 and msg[1] == "ERROR":
                self.stop()
                self.error.emit(msg)
                return

            latest = msg

        if latest is not None:
            _, traj, t = latest
            if traj is not None or t is not None:
                self.progress.emit(traj, t)

        if saw_done:
            self.stop()
            self.done.emit()

