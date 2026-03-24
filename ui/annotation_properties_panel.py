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
    """统一展示项目状态、训练状态、实例树和当前选中实例属性。"""

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
        "create_group": "新建植株组并绑定",
    }

    entity_selected = pyqtSignal(str, object)
    class_changed = pyqtSignal(int)
    owner_changed = pyqtSignal(object)
    create_group_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._owner_options = []
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

        # 分组树
        tree_group = QGroupBox("实例分组")
        tree_layout = QVBoxLayout()
        tree_group.setLayout(tree_layout)
        self.instance_tree = QTreeWidget()
        self.instance_tree.setHeaderLabels(["实例 / 植株组", "信息"])
        self.instance_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.instance_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.instance_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        tree_layout.addWidget(self.instance_tree)
        layout.addWidget(tree_group, 1)

        # 当前选中实例属性
        property_group = QGroupBox("实例属性")
        property_form = QFormLayout()
        property_group.setLayout(property_form)
        self.label_entity_kind = QLabel("未选中")
        self.combo_class = QComboBox()
        self.combo_owner = QComboBox()
        self.btn_create_group = QPushButton("新建植株组并绑定")
        self.label_source = QLabel("-")
        self.label_origin_model = QLabel("-")
        self.label_origin_conf = QLabel("-")
        for label in (
            self.label_entity_kind,
            self.label_source,
            self.label_origin_model,
            self.label_origin_conf,
        ):
            label.setWordWrap(True)
        property_form.addRow("选中对象", self.label_entity_kind)
        property_form.addRow("类别", self.combo_class)
        property_form.addRow("所属植株组", self.combo_owner)
        property_form.addRow("", self.btn_create_group)
        property_form.addRow("source", self.label_source)
        property_form.addRow("origin_model_version", self.label_origin_model)
        property_form.addRow("origin_confidence", self.label_origin_conf)
        layout.addWidget(property_group)

        self.combo_class.currentIndexChanged.connect(self._emit_class_change)
        self.combo_owner.currentIndexChanged.connect(self._emit_owner_change)
        self.btn_create_group.clicked.connect(self.create_group_requested.emit)
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
        self.btn_create_group.setText(self.ACTION_BUTTON_LABELS["create_group"])

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

    def populate_instance_tree(self, plant_groups, formal_instances, candidates):
        self._updating = True
        self.instance_tree.clear()

        group_map = {}
        unassigned_item = QTreeWidgetItem(["Unassigned", "未绑定植株组"])
        self.instance_tree.addTopLevelItem(unassigned_item)
        group_map[None] = unassigned_item

        for group in plant_groups or []:
            top_item = QTreeWidgetItem([group.get("plant_name", "Plant"), f"ID={group.get('plant_id')}"])
            top_item.setData(0, Qt.UserRole, {"kind": "plant_group", "id": group.get("plant_id")})
            self.instance_tree.addTopLevelItem(top_item)
            group_map[group.get("plant_id")] = top_item

        for instance in formal_instances or []:
            owner_id = instance.get("owner_plant_id")
            top_item = group_map.get(owner_id, unassigned_item)
            child_item = QTreeWidgetItem(
                [
                    f"{instance.get('class_name', 'instance')} #{instance.get('id')}",
                    f"{instance.get('source', 'manual')} | area={int(instance.get('total_area', 0))}",
                ]
            )
            child_item.setData(0, Qt.UserRole, {"kind": "formal", "id": instance.get("id")})
            top_item.addChild(child_item)

        if candidates:
            candidate_root = QTreeWidgetItem(["Candidates", f"{len(candidates)}"])
            candidate_root.setExpanded(True)
            self.instance_tree.addTopLevelItem(candidate_root)
            for candidate in candidates:
                confidence = candidate.get("confidence")
                confidence_text = "-" if confidence is None else f"{confidence:.2f}"
                child_item = QTreeWidgetItem(
                    [
                        f"{candidate.get('class_name', 'candidate')} {candidate.get('candidate_id')}",
                        f"conf={confidence_text}",
                    ]
                )
                child_item.setData(0, Qt.UserRole, {"kind": "candidate", "id": candidate.get("candidate_id")})
                candidate_root.addChild(child_item)

        self.instance_tree.expandAll()
        self._updating = False

    def update_selected_entity(self, entity_kind, entity, class_names, plant_groups):
        self._updating = True
        self.combo_class.clear()
        for class_id, class_name in enumerate(class_names or []):
            self.combo_class.addItem(class_name, class_id)

        self.combo_owner.clear()
        self._owner_options = [None]
        self.combo_owner.addItem("Unassigned", None)
        for group in plant_groups or []:
            self.combo_owner.addItem(group.get("plant_name", "Plant"), group.get("plant_id"))
            self._owner_options.append(group.get("plant_id"))

        if not entity or not entity_kind:
            self.label_entity_kind.setText("未选中")
            self.label_source.setText("-")
            self.label_origin_model.setText("-")
            self.label_origin_conf.setText("-")
            self.combo_class.setEnabled(False)
            self.combo_owner.setEnabled(False)
            self.btn_create_group.setEnabled(False)
            self._updating = False
            return

        self.combo_class.setEnabled(True)
        self.combo_owner.setEnabled(True)
        self.btn_create_group.setEnabled(True)

        if entity_kind == "formal":
            self.label_entity_kind.setText(f"正式实例 #{entity.get('id')}")
            self.label_source.setText(str(entity.get("source", "-")))
            self.label_origin_model.setText(str(entity.get("origin_model_version") or "-"))
            origin_conf = entity.get("origin_confidence")
            self.label_origin_conf.setText("-" if origin_conf is None else f"{float(origin_conf):.3f}")
        else:
            self.label_entity_kind.setText(f"候选实例 {entity.get('candidate_id')}")
            self.label_source.setText("candidate")
            self.label_origin_model.setText(str(entity.get("model_version") or "-"))
            confidence = entity.get("confidence")
            self.label_origin_conf.setText("-" if confidence is None else f"{float(confidence):.3f}")

        class_index = max(0, self.combo_class.findData(entity.get("class_id", 0)))
        owner_index = self.combo_owner.findData(entity.get("owner_plant_id"))
        self.combo_class.setCurrentIndex(class_index)
        self.combo_owner.setCurrentIndex(max(0, owner_index))
        self._updating = False

    def select_tree_entity(self, entity_kind, entity_id):
        self._updating = True
        self.instance_tree.clearSelection()

        def visit(item):
            data = item.data(0, Qt.UserRole)
            if isinstance(data, dict) and data.get("kind") == entity_kind and data.get("id") == entity_id:
                self.instance_tree.setCurrentItem(item)
                return True
            for index in range(item.childCount()):
                if visit(item.child(index)):
                    return True
            return False

        for index in range(self.instance_tree.topLevelItemCount()):
            if visit(self.instance_tree.topLevelItem(index)):
                break
        self._updating = False

    def _on_tree_selection_changed(self):
        if self._updating:
            return
        current_item = self.instance_tree.currentItem()
        if not current_item:
            return
        data = current_item.data(0, Qt.UserRole)
        if isinstance(data, dict) and data.get("kind") in ("formal", "candidate"):
            self.entity_selected.emit(data.get("kind"), data.get("id"))

    def _emit_class_change(self, index):
        if self._updating or index < 0:
            return
        class_id = self.combo_class.itemData(index)
        self.class_changed.emit(int(class_id))

    def _emit_owner_change(self, index):
        if self._updating or index < 0:
            return
        owner_id = self.combo_owner.itemData(index)
        self.owner_changed.emit(owner_id)
