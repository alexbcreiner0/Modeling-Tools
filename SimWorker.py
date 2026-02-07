import logging
import traceback
logger = logging.getLogger(__name__)

from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
)
import time
from multiprocessing import Process, Pool
from paths import rpath
import importlib
from tools.loader import to_plain

def child_run(queue, run_id, module_path, func_name, params_path, params, stop_event, pause_event, sleep_value):
    try:
        from tools.loader import params_from_mapping
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        params_dataclass = params_from_mapping(params, params_path)

        for traj, t in func(params_dataclass):
            if stop_event.is_set():
                break

            while not pause_event.is_set():
                if stop_event.is_set():
                    break
                time.sleep(0.02)

            dt = sleep_value.value
            if dt > 0:
                time.sleep(dt)

            queue.put((run_id, traj, t))

        queue.put((run_id, "DONE",))

    except Exception as ex:
        tb = traceback.format_exc()
        queue.put((run_id, "ERROR", repr(ex), tb, module_path, func_name))

class SimWorker(qc.QObject):
    progress = qc.pyqtSignal(object, object)          # traj, t
    finished = qc.pyqtSignal(object, object, object)  # traj, t, e

    def __init__(self, params, stream_func, *, yield_every=1, sleep_time= 0.01,
                 ctx= None, mp_queue= None, model_info= None, run_id= 1):
        super().__init__()
        self.params = params
        self.stream_func = stream_func
        self.yield_every = yield_every
        self._stop = False
        self._pause = False
        self.sleep_time = sleep_time
        self.multiprocessing = False
        self.run_id = run_id
        self.ctx = ctx

        if mp_queue is not None and ctx is not None and model_info is not None:
            self._stop_event = ctx.Event()
            self._pause_event = ctx.Event()
            self._pause_event.set()
            self._sleep_value = ctx.Value("d", float(self.sleep_time))
            self.mp_queue = mp_queue
            self.multiprocessing = True
            sim_model = model_info["details"]["simulation_model"]
            self.sim_function_name = model_info["details"]["simulation_function"]
            self.module_path = f"models.{sim_model}.simulation.simulation" # multiprocessing expects the string
            self.params_path = rpath("models", sim_model, "simulation", "parameters.py") # but my own function needs a path
            self.params = to_plain(params)
            self._proc = None

    @qc.pyqtSlot()
    def request_stop(self, force: bool = False):
        self._stop = True

        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        if force:
            p = getattr(self, "_proc", None)
            if p is not None and p.is_alive():
                p.terminate()
                p.join(timeout= 0.5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout= 0.5)

    @qc.pyqtSlot()
    def toggle_pause(self):
        self._pause = not self._pause

        pause_event = getattr(self, "_pause_event", None)
        if pause_event:
            if self._pause:
                pause_event.clear()
            else:
                pause_event.set()

    def _should_stop(self) -> bool:
        return self._stop or qc.QThread.currentThread().isInterruptionRequested()

    def join(self, timeout= 1.0) -> bool:
        """ During a demo switch, while a sim is running, this is used to force the app to wait until the old demo is down before the new one gets made """
        p = getattr(self, "_proc", None)
        if p is not None:
            return True
        p.join(timeout= timeout)
        return not p.is_alive()

    @qc.pyqtSlot()
    def run(self):
        e = None
        latest_traj, latest_t = None, None
        animating = True

        if self.multiprocessing:
            self._proc = self.ctx.Process(
                target= child_run, 
                args=(
                    self.mp_queue, self.run_id, self.module_path, 
                    self.sim_function_name, self.params_path, 
                    self.params, self._stop_event, self._pause_event, self._sleep_value
                )
            )
            self._proc.start()
            return

        try:
            result = self.stream_func(self.params)

            # if it's a normal function output (i.e. the user is not animating)
            if isinstance(result, tuple) and len(result) == 2:
                animating = False
                traj, t = result
                latest_traj, latest_t = traj, t
                self.progress.emit(traj, t)

            else:
                for i, frame in enumerate(result):
                    if self._should_stop():
                        break

                    time.sleep(self.sleep_time)
                    # stop receiving new outputs if sim is paused
                    while self._pause and not self._should_stop():
                        qc.QThread.msleep(25) # recheck every 25 ms

                    if not (isinstance(frame, tuple) and len(frame) == 2):
                        raise TypeError(f"Streaming sim must yield (traj, t) tuples. Got {type(frame)} {frame!r}")

                    latest_traj, latest_t = frame
                    if latest_traj is None or latest_t is None:
                        continue
                    if (i % self.yield_every) == 0:
                        self.progress.emit(latest_traj, latest_t)

        except Exception as ex:
            latest_t_val = latest_t[-1] if latest_t is not None else None
            extra = {
                "Sim function": self.stream_func.__name__,
                "Animating from generator": animating,
                "latest t value": latest_t_val
            }
            info = (extra, ex)
            self.finished.emit(latest_traj, latest_t, info)
            return

        self.finished.emit(latest_traj, latest_t, e)

