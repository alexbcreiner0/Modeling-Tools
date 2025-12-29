from matplotlib.backends.backend_qt import NavigationToolbar2QT
from PyQt6 import QtWidgets as qw
from pathlib import Path

class CustomNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent=None, default_dir=None):
        super().__init__(canvas, parent)
        self.default_dir = Path(default_dir).expanduser() if default_dir else None
        if self.default_dir:
            self.default_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(self, *args):
        """
        Same as stock toolbar, but dialog starts in self.default_dir.
        """
        if self.default_dir:
            # Use Qt dialog directly so we can force starting directory
            fname, _ = qw.QFileDialog.getSaveFileName(
                self.parent(),                       # parent widget
                "Save the figure",
                str(self.default_dir / "figure.png"),# suggested path/name
                "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
            )
            if not fname:
                return

            fig = self.canvas.figure

            # 2. Match figure size to the on-screen canvas (fix cramped / huge legend issue)
            w_px = self.canvas.width()
            h_px = self.canvas.height()
            dpi = fig.dpi
            fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)

            # Optional: tighten layout if you want
            fig.tight_layout()

            # 3. Actually save
            fig.savefig(fname, dpi=dpi)

            # self.canvas.figure.savefig(fname)
        else:
            super().save_figure(*args)


    # def save_figure(self, *args):
    #     # 1. Ask where to save
    #     path, _ = qw.QFileDialog.getSaveFileName(
    #         self,
    #         "Save figure",
    #         str(self.default_dir) if self.default_dir else "",
    #         "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
    #     )
    #     # path, _ = qw.QFileDialog.getSaveFileName(
    #     #     self,
    #     #     "Save figure",
    #     #     self.default_dir,
    #     #     "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
    #     # )
    #     if not path:
    #         return

    #     fig = self.canvas.figure

    #     # 2. Match figure size to the on-screen canvas (fix cramped / huge legend issue)
    #     w_px = self.canvas.width()
    #     h_px = self.canvas.height()
    #     dpi = fig.dpi
    #     fig.set_size_inches(w_px / dpi, h_px / dpi, forward=True)

    #     # Optional: tighten layout if you want
    #     fig.tight_layout()

    #     # 3. Actually save
    #     fig.savefig(path, dpi=dpi)
