import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import SHORTCUTS
from components.image_label import ImageLabel
from components.toolbars import Toolbars
from models.sam_manager import SamManager
from services.sam_training_manager import SamTrainingManagerV2
from ui.annotation_properties_panel import AnnotationPropertiesPanel
from utils.annotation_schema import make_image_state
from utils.interaction_state import InteractionStateMachine


class MainWindowBase(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玉米标注与项目级预标注闭环工具")
        self.setMinimumSize(1024, 680)

        self.image_paths = []
        self.image_sequence_map = {}
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.current_image_state = make_image_state("")
        self.current_annotation_hash = None

        self.annotation_dir = os.getcwd()
        self.preprocess_cache = {}
        self.annotation_changed = False
        self.undo_stack = []
        self.redo_stack = []
        self.is_undo_redo = False

        self.project_id = None
        self.project_metadata = None
        self.project_paths = None

        self.coco_container = {}
        self.save_path = None
        self.import_path = None
        self.export_path = None

        self.ignoring_region = False

        self.sam_manager = SamManager()
        self.sam_training_manager = SamTrainingManagerV2(self.sam_manager)
        self.sam_training_worker = None
        self.current_preannotation_candidate = None
        self.preannotation_adjustment_records = []
        self.preannotation_record_counter = 1
        self.preannotation_fine_tune_sessions = {}
        self.interaction_state_machine = InteractionStateMachine()

        self.init_ui()
        self.resize_to_available_screen()
        self.init_shortcuts()
        self.restore_button_texts()
        self.restore_button_visuals()
        self._update_preannotation_controls()

        self.update_status_bar()

    def _resolve_image_sequence(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return None
        sequence = self.image_sequence_map.get(target_image)
        if sequence is not None:
            return sequence
        if target_image in self.image_paths:
            sequence = self.image_paths.index(target_image) + 1
            self.image_sequence_map[target_image] = sequence
            return sequence
        if self.current_image_index >= 0:
            return self.current_image_index + 1
        return None

    def sync_interaction_state(self):
        if self.left_label.preannotation_box_mode:
            return self.interaction_state_machine.force(InteractionStateMachine.PREANNOTATION_BOX)
        if self.left_label.candidate_instances:
            return self.interaction_state_machine.force(InteractionStateMachine.PREANNOTATION_CANDIDATE)
        if self.ignoring_region:
            return self.interaction_state_machine.force(InteractionStateMachine.IGNORE_REGION)
        if self.left_label.removing_region:
            return self.interaction_state_machine.force(InteractionStateMachine.REMOVAL_REGION)
        if self.left_label.mode == "fine_tune":
            if self.left_label.split_staging_mode:
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_SPLIT_STAGING)
            if getattr(self.left_label, "delete_vertex_mode", False):
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_DELETE_VERTEX)
            if self.left_label.add_vertex_mode:
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_ADD_VERTEX)
            return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE)
        return self.interaction_state_machine.force(InteractionStateMachine.IDLE)

    def init_ui(self):
        """初始化 UI。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(main_splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.addWidget(Toolbars.create_auxiliary_toolbar(self))
        left_layout.addWidget(Toolbars.create_annotation_toolbar(self))
        left_layout.addWidget(Toolbars.create_plant_management_toolbar(self))
        left_layout.addWidget(Toolbars.create_navigation_toolbar(self))
        left_layout.addWidget(Toolbars.create_progress_label(self))
        left_layout.addStretch()
        left_scroll = self.create_scroll_panel(left_panel, min_width=220, preferred_width=260)
        main_splitter.addWidget(left_scroll)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        images_layout = QHBoxLayout()
        images_layout.setSpacing(10)
        center_layout.addLayout(images_layout)

        self.left_label = ImageLabel(is_summary=False, parent=self)
        self.left_label.setToolTip(
            "左键添加顶点/选择实例 | Enter 暂存区域 | Shift+Enter 保存实例 | 右键拖动 | 滚轮缩放"
        )
        images_layout.addWidget(self.left_label, 1)

        self.right_label = ImageLabel(is_summary=True, parent=self)
        self.right_label.setToolTip("正式实例总览，不显示候选层")
        images_layout.addWidget(self.right_label, 1)
        main_splitter.addWidget(center_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        self.properties_panel = AnnotationPropertiesPanel(self)
        right_layout.addWidget(self.properties_panel, 3)
        right_layout.addWidget(Toolbars.create_file_toolbar(self))
        right_layout.addWidget(Toolbars.create_sam_toolbar(self))
        right_layout.addWidget(Toolbars.create_export_toolbar(self))
        right_layout.addWidget(Toolbars.create_aux_toolbar(self))
        right_layout.addStretch()
        right_scroll = self.create_scroll_panel(right_panel, min_width=300, preferred_width=380)
        main_splitter.addWidget(right_scroll)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([260, 1200, 380])

    def create_scroll_panel(self, widget, min_width, preferred_width):
        """为侧栏创建滚动容器，避免小分辨率下内容被截断。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(min_width)
        scroll.setMaximumWidth(max(preferred_width + 80, min_width))
        return scroll

    def resize_to_available_screen(self):
        """根据当前屏幕可用区域设置初始窗口大小并居中。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 900)
            return

        geometry = screen.availableGeometry()
        width = max(self.minimumWidth(), min(int(geometry.width() * 0.94), 1920))
        height = max(self.minimumHeight(), min(int(geometry.height() * 0.94), 1080))
        width = min(width, geometry.width())
        height = min(height, geometry.height())
        self.resize(width, height)

        x = geometry.x() + max(0, (geometry.width() - width) // 2)
        y = geometry.y() + max(0, (geometry.height() - height) // 2)
        self.move(x, y)

    def init_shortcuts(self):
        """初始化快捷键。"""
        QShortcut(QKeySequence(SHORTCUTS["SAVE_POLYGON"]), self, self.save_current_polygon)
        QShortcut(QKeySequence(SHORTCUTS["SAVE_PLANT"]), self, self.save_plant)
        QShortcut(QKeySequence(SHORTCUTS["UNDO"]), self, self.undo)
        QShortcut(QKeySequence(SHORTCUTS["REDO"]), self, self.redo)
        QShortcut(QKeySequence(SHORTCUTS["DELETE_PLANT"]), self, self.delete_plant)
        QShortcut(QKeySequence(SHORTCUTS["DELETE_STAGING_POLYGON"]), self, self.delete_selected_staging_polygon_shortcut)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_EDGE_SNAP"]), self, self.toggle_edge_snap)
        QShortcut(QKeySequence(SHORTCUTS["LOAD_BATCH"]), self, self.load_batch_images)
        QShortcut(QKeySequence(SHORTCUTS["PREV_IMAGE"]), self, self.prev_image)
        QShortcut(QKeySequence(SHORTCUTS["NEXT_IMAGE"]), self, self.next_image)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_IGNORE_REGION"]), self, self.toggle_ignore_region)

    def restore_button_texts(self):
        """集中恢复动态和静态按钮文本，避免异常回调后出现空白按钮。"""
        if hasattr(self, "btn_save_polygon"):
            self.btn_save_polygon.setText(f"暂存当前区域 ({SHORTCUTS['SAVE_POLYGON']})")
        if hasattr(self, "btn_save_plant"):
            self.btn_save_plant.setText(f"保存整株 ({SHORTCUTS['SAVE_PLANT']})")
        if hasattr(self, "btn_undo"):
            self.btn_undo.setText(f"撤销 ({SHORTCUTS['UNDO']})")
        if hasattr(self, "btn_prev"):
            self.btn_prev.setText(f"上一张 ({SHORTCUTS['PREV_IMAGE']})")
        if hasattr(self, "btn_next"):
            self.btn_next.setText(f"下一张 ({SHORTCUTS['NEXT_IMAGE']})")
        if hasattr(self, "btn_delete"):
            self.btn_delete.setText(f"删除选中植株 ({SHORTCUTS['DELETE_PLANT']})")
        if hasattr(self, "btn_delete_staging_polygon"):
            self.btn_delete_staging_polygon.setText(f"删除选中的暂存区域 ({SHORTCUTS['DELETE_STAGING_POLYGON']})")
        if hasattr(self, "btn_delete_vertex"):
            self.btn_delete_vertex.setText("删除顶点")
        if hasattr(self, "btn_load_batch"):
            self.btn_load_batch.setText(f"批量加载图片 ({SHORTCUTS['LOAD_BATCH']})")
        if hasattr(self, "btn_load_folder"):
            self.btn_load_folder.setText("从文件夹加载图片与COCO")
        if hasattr(self, "btn_export_annotated"):
            self.btn_export_annotated.setText("批量导出已完成")
        if hasattr(self, "btn_help"):
            self.btn_help.setText("使用说明")
        if hasattr(self, "btn_debug_coco"):
            self.btn_debug_coco.setText("调试COCO容器")
        if hasattr(self, "btn_toggle_annotation"):
            if self.current_image_state.get("annotation_completed"):
                self.btn_toggle_annotation.setText("取消当前图片已完成")
            else:
                self.btn_toggle_annotation.setText("标记当前图片为已完成")
        if hasattr(self, "btn_ignore_region"):
            if self.ignoring_region:
                self.btn_ignore_region.setText(f"退出忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            else:
                self.btn_ignore_region.setText(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
        if hasattr(self, "btn_toggle_snap"):
            self.update_snap_button_state()
        if hasattr(self, "btn_start_training"):
            self.btn_start_training.setText("开始训练")
        if hasattr(self, "btn_sam_preannotate"):
            self.btn_sam_preannotate.setText("取消框选预标注" if self.left_label.preannotation_box_mode else "框选预标注")
        if hasattr(self, "btn_sam_select_mode"):
            self.btn_sam_select_mode.setText("接受候选并微调")
        if hasattr(self, "btn_save_staging_areas"):
            self.btn_save_staging_areas.setText("拒绝当前候选")
        if hasattr(self, "btn_export_preannotation_records"):
            self.btn_export_preannotation_records.setText("导出预标注调整记录")
        if hasattr(self, "btn_export_weights"):
            self.btn_export_weights.setText("导出当前权重")
        if hasattr(self, "btn_select_weights"):
            self.btn_select_weights.setText("选择权重")
        if hasattr(self, "properties_panel"):
            self.properties_panel.restore_button_texts()

    def restore_button_visuals(self):
        """训练异常后强制刷新按钮调色板和重绘，避免按钮文字可见性丢失。"""
        for button in self.findChildren(QPushButton):
            palette = button.palette()
            background = palette.color(QPalette.Button)
            luminance = (
                background.red() * 299 + background.green() * 587 + background.blue() * 114
            ) / 1000
            text_color = QColor("#111111") if luminance >= 160 else QColor("#f5f5f5")
            palette.setColor(QPalette.ButtonText, text_color)
            palette.setColor(QPalette.WindowText, text_color)
            button.setPalette(palette)
            button.ensurePolished()
            if button.style():
                button.style().unpolish(button)
                button.style().polish(button)
            button.update()
