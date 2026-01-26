from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
) 
import yaml
import sys
from pathlib import Path
from pathlib import Path
from paths import rpath
import logging, atexit
import logging.config
from logging.handlers import RotatingFileHandler
import threading

LOG_DIR = rpath("logs")
LOG_DIR.mkdir(exist_ok= True)

LOG_FILE = LOG_DIR / "log.jsonl"

with open(rpath("logging_config.yml"), "r") as f:
    logging_config = yaml.safe_load(f)

logging_config["handlers"]["app_file"]["filename"] = str(LOG_FILE)
logging.config.dictConfig(config= logging_config)
logging.captureWarnings(True)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger(__name__)
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def handle_thread_exception(args: threading.ExceptHookArgs):
    logging.getLogger(__name__).error(
        "Uncaught thread exception",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )

threading.excepthook = handle_thread_exception
sys.excepthook = handle_exception

from MainWindow import MainWindow

def apply_dpi_scaled_font(app: qw.QApplication, base_pt: float = 10.0) -> None:
    screen = app.primaryScreen()
    if not screen:
        return
    dpi = screen.logicalDotsPerInch()  # ~96 at 100% scaling
    scale = dpi / 96.0

    f = app.font()
    f.setPointSizeF(base_pt * scale)
    app.setFont(f)

if __name__ == "__main__":

    app = qw.QApplication(sys.argv)
    apply_dpi_scaled_font(app)
    app.setStyle("Fusion")

    light_palette = qg.QPalette()

    light_palette.setColor(qg.QPalette.ColorRole.Window, qg.QColor(245, 245, 245))
    light_palette.setColor(qg.QPalette.ColorRole.WindowText, qc.Qt.GlobalColor.black)
    light_palette.setColor(qg.QPalette.ColorRole.Base, qg.QColor(255, 255, 255))
    light_palette.setColor(qg.QPalette.ColorRole.AlternateBase, qg.QColor(240, 240, 240))
    light_palette.setColor(qg.QPalette.ColorRole.Text, qc.Qt.GlobalColor.black)
    light_palette.setColor(qg.QPalette.ColorRole.Button, qg.QColor(240, 240, 240))
    light_palette.setColor(qg.QPalette.ColorRole.ButtonText, qc.Qt.GlobalColor.black)
    light_palette.setColor(qg.QPalette.ColorRole.Highlight, qg.QColor(76, 163, 224))
    light_palette.setColor(qg.QPalette.ColorRole.HighlightedText, qc.Qt.GlobalColor.white)

    app.setPalette(light_palette)

    # doesn't seem to work
    app.setStyleSheet("""
        QToolTip {
            max-width: 300px;
            white-space: normal;
        }
    """)

    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))

    with open(rpath(f"config.yml"), "r") as f:
        config = yaml.safe_load(f)

    window = MainWindow(config)

    window.show()
    app.exec()
