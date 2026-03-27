# 标注属性与项目状态面板
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
    """统一展示项目状态、训练状态。"""

    ACTION_BUTTON_LABELS = {
        "mark_completed": "标记当前图片为已完成",
        "mark_incomplete": "取消当前图片已完成",
        "run_inference": "对当前图执行AI预标注",
        "accept_candidate": "接受当前候选",
        "accept_all_candidates": "接受全部候选",
        "clear_candidates": "清空候选",
        "delete_selected": "删除当前选中实例",
        "manual_train": "手动启动训练",
        "rollback_model": "回退到上一模型",
        "rebuild_split": "重建验证集",
    }

    entity_selected = pyqtSignal(str, object)
    class_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 项目状态
        project_group = QGroupBox("项目状态")
        project_form = QFormLayout()
        project_group.setLayout(project_form)
        self.label_project_name = QLabel("未加载")
        self.label_active_model = QLabel("暂无模型")
        self.label_training_status = QLabel("空闲")
        self.label_completed_count = QLabel("0")
        self.label_dirty_count = QLabel("0")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        for label in (
            self.label_project_name,
            self.label_active_model,
            self.label_training_status,
        ):
            label.setWordWrap(True)
        project_form.addRow("项目", self.label_project_name)
        project_form.addRow("Active 模型", self.label_active_model)
        project_form.addRow("训练状态", self.label_training_status)
        project_form.addRow("已完成图片", self.label_completed_count)
        project_form.addRow("Dirty Completed", self.label_dirty_count)
        project_form.addRow("训练进度", self.training_progress)
        layout.addWidget(project_group)

        # 快捷操作
        action_group = QGroupBox("项目操作")
        action_layout = QVBoxLayout()
        action_group.setLayout(action_layout)
        self.btn_mark_completed = QPushButton("标记当前图片为已完成")
        self.btn_mark_incomplete = QPushButton("取消当前图片已完成")
        self.btn_run_inference = QPushButton("对当前图执行AI预标注")
        self.btn_accept_candidate = QPushButton("接受当前候选")
        self.btn_accept_all_candidates = QPushButton("接受全部候选")
        self.btn_clear_candidates = QPushButton("清空候选")
        self.btn_delete_selected = QPushButton("删除当前选中实例")
        self.btn_manual_train = QPushButton("手动启动训练")
        self.btn_rollback_model = QPushButton("回退到上一模型")
        self.btn_rebuild_split = QPushButton("重建验证集")
        for button in (
            self.btn_mark_completed,
            self.btn_mark_incomplete,
            self.btn_run_inference,
            self.btn_accept_candidate,
            self.btn_accept_all_candidates,
            self.btn_clear_candidates,
            self.btn_delete_selected,
            self.btn_manual_train,
            self.btn_rollback_model,
            self.btn_rebuild_split,
        ):
            action_layout.addWidget(button)
        layout.addWidget(action_group)

        self.restore_button_texts()

    def restore_button_texts(self):
        """按钮文本集中恢复，避免异常回调后出现空白按钮。"""
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
        self.label_project_name.setText(project_name or "未加载")
        self.label_active_model.setText(active_model_version or "暂无模型")
        self.label_training_status.setText(training_status or "空闲")
        self.label_training_status.setToolTip(training_status or "空闲")
        self.label_completed_count.setText(str(completed_count or 0))
        self.label_dirty_count.setText(str(dirty_count or 0))

    def update_training_progress(self, value, text=None):
        self.training_progress.setValue(max(0, min(100, int(value))))
        if text:
            self.label_training_status.setText(text)
            self.label_training_status.setToolTip(text)

    def populate_instance_tree(self, formal_instances, candidates):
        # 移除实例树功能
        pass

    def update_selected_entity(self, entity_kind, entity, class_names):
        # 移除实例属性功能
        pass

    def select_tree_entity(self, entity_kind, entity_id):
        # 移除实例树功能
        pass

    def _on_tree_selection_changed(self):
        # 移除实例树功能
        pass

    def _emit_class_change(self, index):
        # 移除实例属性功能
        pass