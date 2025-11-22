from PyQt6 import (
    QtWidgets as qw,
) 
import yaml
import sys
from pathlib import Path
from MainWindow import MainWindow
import yaml

if __name__ == "__main__":
    app = qw.QApplication([])

    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))

    with open(f"demos.yml", "r") as f:
        demos = yaml.safe_load(f)

    window = MainWindow(demos)

    window.show()
    app.exec()

    with open("dimensions.txt", "w") as f:
        xlim, ylim = window.graph_panel.xlim, window.graph_panel.ylim
        xlim_float = (float(xlim[0]), float(xlim[1]))
        ylim_float = (float(ylim[0]), float(ylim[1]))
        print(xlim_float, file= f)
        print(ylim_float, file= f)
