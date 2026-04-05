import sys
import traceback

from PyQt5.QtWidgets import QApplication

from app.main_window import MainWindow


sys.excepthook = lambda exctype, value, tb: traceback.print_exception(exctype, value, tb)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
