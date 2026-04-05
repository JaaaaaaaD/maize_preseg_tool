# 工具栏组件
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
)

from config import SHORTCUTS


class Toolbars:
    """工具栏组件管理。"""

    @staticmethod
    def create_file_toolbar(parent):
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        parent.btn_load_batch = QPushButton(f"批量加载图片 ({SHORTCUTS['LOAD_BATCH']})")
        parent.btn_load_batch.clicked.connect(parent.load_batch_images)
        file_layout.addWidget(parent.btn_load_batch)

        return file_group

    @staticmethod
    def create_navigation_toolbar(parent):
        nav_group = QGroupBox("导航")
        nav_layout = QVBoxLayout()
        nav_group.setLayout(nav_layout)

        parent.btn_refresh = QPushButton("刷新项目状态")
        parent.btn_refresh.clicked.connect(parent.refresh_project_status)
        nav_layout.addWidget(parent.btn_refresh)

        parent.btn_toggle_annotation = QPushButton("标记当前图片为已完成")
        parent.btn_toggle_annotation.clicked.connect(parent.toggle_annotation_status)
        nav_layout.addWidget(parent.btn_toggle_annotation)

        parent.btn_prev = QPushButton(f"上一张 ({SHORTCUTS['PREV_IMAGE']})")
        parent.btn_prev.clicked.connect(parent.prev_image)
        parent.btn_prev.setEnabled(False)
        nav_layout.addWidget(parent.btn_prev)

        parent.btn_next = QPushButton(f"下一张 ({SHORTCUTS['NEXT_IMAGE']})")
        parent.btn_next.clicked.connect(parent.next_image)
        parent.btn_next.setEnabled(False)
        nav_layout.addWidget(parent.btn_next)

        return nav_group

    @staticmethod
    def create_annotation_toolbar(parent):
        annotate_group = QGroupBox("标注操作")
        annotate_layout = QVBoxLayout()
        annotate_group.setLayout(annotate_layout)

        label_group = QGroupBox("区域标签")
        label_layout = QVBoxLayout()
        label_group.setLayout(label_layout)

        parent.combo_label = QComboBox()
        parent.combo_label.addItems(["stem", "leaf", "ear"])
        parent.combo_label.setMinimumWidth(150)
        label_layout.addWidget(parent.combo_label)
        annotate_layout.addWidget(label_group)

        parent.btn_save_polygon = QPushButton(f"暂存当前区域 ({SHORTCUTS['SAVE_POLYGON']})")
        parent.btn_save_polygon.clicked.connect(parent.save_current_polygon)
        annotate_layout.addWidget(parent.btn_save_polygon)

        parent.btn_save_plant = QPushButton(f"保存整株 ({SHORTCUTS['SAVE_PLANT']})")
        parent.btn_save_plant.clicked.connect(parent.save_plant)
        annotate_layout.addWidget(parent.btn_save_plant)

        parent.btn_undo = QPushButton(f"撤销 ({SHORTCUTS['UNDO']})")
        parent.btn_undo.clicked.connect(parent.undo)
        annotate_layout.addWidget(parent.btn_undo)

        parent.btn_redo = QPushButton(f"重做 ({SHORTCUTS['REDO']})")
        parent.btn_redo.clicked.connect(parent.redo)
        annotate_layout.addWidget(parent.btn_redo)

        return annotate_group

    @staticmethod
    def create_auxiliary_toolbar(parent):
        aux_func_group = QGroupBox("辅助功能")
        aux_func_layout = QVBoxLayout()
        aux_func_group.setLayout(aux_func_layout)

        parent.btn_toggle_snap = QPushButton(f"边缘吸附: 开启 ({SHORTCUTS['TOGGLE_EDGE_SNAP']})")
        parent.btn_toggle_snap.clicked.connect(parent.toggle_edge_snap)
        aux_func_layout.addWidget(parent.btn_toggle_snap)

        parent.btn_toggle_projection = QPushButton("投影框: 关闭")
        parent.btn_toggle_projection.clicked.connect(parent.toggle_projection)
        aux_func_layout.addWidget(parent.btn_toggle_projection)

        parent.btn_ignore_region = QPushButton(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
        parent.btn_ignore_region.clicked.connect(parent.toggle_ignore_region)
        aux_func_layout.addWidget(parent.btn_ignore_region)

        parent.btn_removal_region = QPushButton("去除区域 (R)")
        parent.btn_removal_region.clicked.connect(parent.toggle_removal_region)
        aux_func_layout.addWidget(parent.btn_removal_region)

        parent.btn_clear_all_ignore = QPushButton("清除所有忽略区域")
        parent.btn_clear_all_ignore.clicked.connect(parent.clear_all_ignore_regions)
        aux_func_layout.addWidget(parent.btn_clear_all_ignore)

        parent.btn_toggle_ai = QPushButton("AI辅助: 开启")
        parent.btn_toggle_ai.clicked.connect(parent.toggle_ai_assist)
        aux_func_layout.addWidget(parent.btn_toggle_ai)

        return aux_func_group

    @staticmethod
    def create_plant_management_toolbar(parent):
        plant_group = QGroupBox("植株管理")
        plant_layout = QVBoxLayout()
        plant_group.setLayout(plant_layout)

        parent.combo_plants = QComboBox()
        parent.combo_plants.setMinimumWidth(150)
        plant_layout.addWidget(parent.combo_plants)

        parent.btn_delete = QPushButton(f"删除选中植株 ({SHORTCUTS['DELETE_PLANT']})")
        parent.btn_delete.clicked.connect(parent.delete_plant)
        plant_layout.addWidget(parent.btn_delete)

        parent.btn_continue_annotation = QPushButton("继续标注选中植株")
        parent.btn_continue_annotation.clicked.connect(parent.continue_annotation)
        plant_layout.addWidget(parent.btn_continue_annotation)

        parent.btn_fine_tune = QPushButton("微调模式")
        parent.btn_fine_tune.clicked.connect(parent.toggle_fine_tune_mode)
        plant_layout.addWidget(parent.btn_fine_tune)

        parent.btn_add_vertex = QPushButton("添加顶点")
        parent.btn_add_vertex.clicked.connect(parent.toggle_add_vertex_mode)
        parent.btn_add_vertex.setEnabled(False)
        plant_layout.addWidget(parent.btn_add_vertex)

        parent.btn_delete_vertex = QPushButton("删除顶点")
        parent.btn_delete_vertex.clicked.connect(parent.toggle_delete_vertex_mode)
        parent.btn_delete_vertex.setEnabled(False)
        plant_layout.addWidget(parent.btn_delete_vertex)

        parent.btn_apply_staging_label = QPushButton("修改选中暂存区域标签")
        parent.btn_apply_staging_label.clicked.connect(parent.apply_selected_staging_label)
        parent.btn_apply_staging_label.setEnabled(False)
        plant_layout.addWidget(parent.btn_apply_staging_label)

        parent.btn_delete_staging_polygon = QPushButton(
            f"删除选中的暂存区域 ({SHORTCUTS['DELETE_STAGING_POLYGON']})"
        )
        parent.btn_delete_staging_polygon.clicked.connect(parent.delete_selected_staging_polygon)
        parent.btn_delete_staging_polygon.setEnabled(False)
        plant_layout.addWidget(parent.btn_delete_staging_polygon)

        parent.btn_split_staging_polygon = QPushButton("切割选中暂存区域")
        parent.btn_split_staging_polygon.clicked.connect(parent.toggle_split_staging_polygon)
        parent.btn_split_staging_polygon.setEnabled(False)
        plant_layout.addWidget(parent.btn_split_staging_polygon)

        parent.btn_undo_delete = QPushButton("撤销删除植株")
        parent.btn_undo_delete.clicked.connect(parent.undo_delete_plant)
        plant_layout.addWidget(parent.btn_undo_delete)

        return plant_group

    @staticmethod
    def create_export_toolbar(parent):
        export_group = QGroupBox("导入/导出")
        export_layout = QVBoxLayout()
        export_group.setLayout(export_layout)

        parent.btn_import_batch = QPushButton("批量导入数据")
        parent.btn_import_batch.clicked.connect(parent.import_batch_data)
        export_layout.addWidget(parent.btn_import_batch)

        parent.btn_export_annotated = QPushButton("批量导出已完成(coco格式)")
        parent.btn_export_annotated.clicked.connect(parent.export_annotated_images)
        export_layout.addWidget(parent.btn_export_annotated)

        return export_group

    @staticmethod
    def create_sam_toolbar(parent):
        sam_group = QGroupBox("SAM辅助")
        sam_layout = QVBoxLayout()
        sam_group.setLayout(sam_layout)

        parent.btn_load_sam = QPushButton("加载SAM模型")
        parent.btn_load_sam.clicked.connect(parent.load_sam_model)
        sam_layout.addWidget(parent.btn_load_sam)

        parent.btn_sam_train = QPushButton("开始训练")
        parent.btn_sam_train.clicked.connect(parent.start_sam_training)
        sam_layout.addWidget(parent.btn_sam_train)

        parent.btn_sam_preannotate = QPushButton("框选预标注")
        parent.btn_sam_preannotate.clicked.connect(parent.run_sam_preannotation)
        sam_layout.addWidget(parent.btn_sam_preannotate)

        parent.btn_sam_select_mode = QPushButton("接受候选并微调")
        parent.btn_sam_select_mode.clicked.connect(parent.enter_sam_select_mode)
        parent.btn_sam_select_mode.setEnabled(False)
        sam_layout.addWidget(parent.btn_sam_select_mode)

        parent.btn_save_staging_areas = QPushButton("拒绝当前候选")
        parent.btn_save_staging_areas.clicked.connect(parent.save_selected_staging_areas)
        parent.btn_save_staging_areas.setEnabled(False)
        sam_layout.addWidget(parent.btn_save_staging_areas)

        parent.btn_export_preannotation_records = QPushButton("导出预标注调整记录")
        parent.btn_export_preannotation_records.clicked.connect(parent.export_preannotation_adjustments)
        sam_layout.addWidget(parent.btn_export_preannotation_records)

        parent.sam_info_text = QTextEdit()
        parent.sam_info_text.setReadOnly(True)
        parent.sam_info_text.setMinimumHeight(60)
        parent.sam_info_text.setPlaceholderText("SAM 信息将显示在这里...")
        sam_layout.addWidget(parent.sam_info_text)

        return sam_group

    @staticmethod
    def create_aux_toolbar(parent):
        aux_group = QGroupBox("辅助")
        aux_layout = QVBoxLayout()
        aux_group.setLayout(aux_layout)

        parent.btn_help = QPushButton("使用说明")
        parent.btn_help.clicked.connect(parent.show_help)
        aux_layout.addWidget(parent.btn_help)

        parent.btn_debug_coco = QPushButton("调试COCO容器")
        parent.btn_debug_coco.clicked.connect(parent.debug_print_coco_container)
        aux_layout.addWidget(parent.btn_debug_coco)

        return aux_group

    @staticmethod
    def create_progress_label(parent):
        parent.image_progress_label = QLabel("0/0")
        parent.image_progress_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        return parent.image_progress_label
