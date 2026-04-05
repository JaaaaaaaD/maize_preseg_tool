from .main_window_annotation import MainWindowAnnotationMixin
from .main_window_base import MainWindowBase
from .main_window_io import MainWindowIOMixin
from .main_window_project import MainWindowProjectMixin
from .main_window_sam import MainWindowSamMixin


class MainWindow(
    MainWindowSamMixin,
    MainWindowIOMixin,
    MainWindowAnnotationMixin,
    MainWindowProjectMixin,
    MainWindowBase,
):
    pass
