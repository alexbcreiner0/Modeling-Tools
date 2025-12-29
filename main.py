from PyQt6 import (
    QtWidgets as qw,
) 
import yaml
import sys, os
from pathlib import Path
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

    # doesn't seem to work
    app.setStyleSheet("""
        QToolTip {
            max-width: 300px;
            white-space: normal;
        }
    """)

    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))

    with open(f"config.yml", "r") as f:
        config = yaml.safe_load(f)

    window = MainWindow(config)

    window.show()
    app.exec()

