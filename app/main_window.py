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
    def __init__(self):
        super().__init__()
        if hasattr(self, "try_restore_last_session_on_startup"):
            self.try_restore_last_session_on_startup()
