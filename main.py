from PyQt6 import (
    QtWidgets as qw,
    QtGui as qg,
    QtCore as qc
) 
import yaml
import sys, os
from pathlib import Path
from MainWindow import MainWindow
from pathlib import Path
from paths import rpath

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
