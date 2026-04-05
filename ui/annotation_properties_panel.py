# ???????????
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QProgressBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AnnotationPropertiesPanel(QWidget):
    """??????????????"""

    ACTION_BUTTON_LABELS = {
        "mark_completed": "??????????",
        "mark_incomplete": "?????????",
        "run_inference": "??????AI???",
        "accept_candidate": "??????",
        "accept_all_candidates": "??????",
        "clear_candidates": "????",
        "delete_selected": "????????",
        "manual_train": "??????",
        "rollback_model": "???????",
        "rebuild_split": "?????",
    }

    entity_selected = pyqtSignal(str, object)
    class_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.btn_mark_completed = QPushButton("??????????")
        self.btn_mark_incomplete = QPushButton("?????????")
        self.btn_run_inference = QPushButton("??????AI???")
        self.btn_accept_candidate = QPushButton("??????")
        self.btn_accept_all_candidates = QPushButton("??????")
        self.btn_clear_candidates = QPushButton("????")
        self.btn_delete_selected = QPushButton("????????")
        self.btn_manual_train = QPushButton("??????")
        self.btn_rollback_model = QPushButton("???????")
        self.btn_rebuild_split = QPushButton("?????")
        self.label_project_name = QLabel("???")
        self.label_active_model = QLabel("????")
        self.label_training_status = QLabel("??")
        self.label_completed_count = QLabel("0")
        self.label_dirty_count = QLabel("0")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        self.restore_button_texts()

    def restore_button_texts(self):
        self.btn_mark_completed.setText(self.ACTION_BUTTON_LABELS["mark_completed"])
        self.btn_mark_incomplete.setText(self.ACTION_BUTTON_LABELS["mark_incomplete"])
        self.btn_run_inference.setText(self.ACTION_BUTTON_LABELS["run_inference"])
        self.btn_accept_candidate.setText(self.ACTION_BUTTON_LABELS["accept_candidate"])
        self.btn_accept_all_candidates.setText(self.ACTION_BUTTON_LABELS["accept_all_candidates"])
        self.btn_clear_candidates.setText(self.ACTION_BUTTON_LABELS["clear_candidates"])
        self.btn_delete_selected.setText(self.ACTION_BUTTON_LABELS["delete_selected"])
        self.btn_manual_train.setText(self.ACTION_BUTTON_LABELS["manual_train"])
        self.btn_rollback_model.setText(self.ACTION_BUTTON_LABELS["rollback_model"])
        self.btn_rebuild_split.setText(self.ACTION_BUTTON_LABELS["rebuild_split"])

    def update_project_info(self, project_name, active_model_version, training_status, completed_count, dirty_count):
        self.restore_button_texts()
        self.label_project_name.setText(project_name or "???")
        self.label_active_model.setText(active_model_version or "????")
        self.label_training_status.setText(training_status or "??")
        self.label_training_status.setToolTip(training_status or "??")
        self.label_completed_count.setText(str(completed_count or 0))
        self.label_dirty_count.setText(str(dirty_count or 0))

    def update_training_progress(self, value, text=None):
        self.training_progress.setValue(max(0, min(100, int(value))))
        if text:
            self.label_training_status.setText(text)
            self.label_training_status.setToolTip(text)

    def update_preannotation_records(self, current_image_path, records_by_image):
        pass

    def populate_instance_tree(self, formal_instances, candidates):
        pass

    def update_selected_entity(self, entity_kind, entity, class_names):
        pass

    def select_tree_entity(self, entity_kind, entity_id):
        pass

    def _on_tree_selection_changed(self):
        pass

    def _emit_class_change(self, index):
        pass
