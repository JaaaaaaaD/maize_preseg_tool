from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from config import SHORTCUTS


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class Toolbars:
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
    def create_timing_toolbar(parent):
        timing_group = QGroupBox("标注计时")
        timing_layout = QVBoxLayout()
        timing_group.setLayout(timing_layout)

        parent.label_timing_status = QLabel("状态: 未开始")
        timing_layout.addWidget(parent.label_timing_status)

        parent.label_timing_total = QLabel("总耗时: 00:00:00")
        timing_layout.addWidget(parent.label_timing_total)

        parent.btn_timer_start = QPushButton("开始/继续计时")
        parent.btn_timer_start.clicked.connect(parent.start_annotation_timer)
        timing_layout.addWidget(parent.btn_timer_start)

        parent.btn_timer_pause = QPushButton("暂停计时")
        parent.btn_timer_pause.clicked.connect(parent.pause_annotation_timer)
        timing_layout.addWidget(parent.btn_timer_pause)
        return timing_group

    @staticmethod
    def create_annotation_toolbar(parent):
        annotate_group = QGroupBox("标注操作")
        annotate_layout = QVBoxLayout()
        annotate_group.setLayout(annotate_layout)

        label_group = QGroupBox("区域标签")
        label_layout = QVBoxLayout()
        label_group.setLayout(label_layout)

        parent.combo_label = NoWheelComboBox()
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
        return aux_func_group

    @staticmethod
    def create_plant_management_toolbar(parent):
        plant_group = QGroupBox("植株管理")
        plant_layout = QVBoxLayout()
        plant_group.setLayout(plant_layout)

        parent.combo_plants = NoWheelComboBox()
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

        parent.btn_brush_vertex = QPushButton("画笔建区域")
        parent.btn_brush_vertex.clicked.connect(parent.toggle_brush_vertex_mode)
        parent.btn_brush_vertex.setEnabled(True)
        plant_layout.addWidget(parent.btn_brush_vertex)

        parent.btn_brush_delete = QPushButton("画笔删暂存")
        parent.btn_brush_delete.clicked.connect(parent.toggle_brush_delete_mode)
        parent.btn_brush_delete.setEnabled(True)
        plant_layout.addWidget(parent.btn_brush_delete)

        parent.btn_apply_staging_label = QPushButton("修改选中暂存区域标签")
        parent.btn_apply_staging_label.clicked.connect(parent.apply_selected_staging_label)
        parent.btn_apply_staging_label.setEnabled(False)
        plant_layout.addWidget(parent.btn_apply_staging_label)

        parent.btn_delete_staging_polygon = QPushButton(
            f"删除选中区域/去除区域 ({SHORTCUTS['DELETE_STAGING_POLYGON']})"
        )
        parent.btn_delete_staging_polygon.clicked.connect(parent.delete_selected_staging_polygon)
        parent.btn_delete_staging_polygon.setEnabled(False)
        plant_layout.addWidget(parent.btn_delete_staging_polygon)

        parent.btn_split_staging_polygon = QPushButton("切分选中暂存区域")
        parent.btn_split_staging_polygon.clicked.connect(parent.toggle_split_staging_polygon)
        parent.btn_split_staging_polygon.setEnabled(False)
        plant_layout.addWidget(parent.btn_split_staging_polygon)

        parent.btn_merge_staging_polygon = QPushButton("合并暂存区域")
        parent.btn_merge_staging_polygon.clicked.connect(parent.toggle_merge_staging_polygon)
        parent.btn_merge_staging_polygon.setEnabled(False)
        plant_layout.addWidget(parent.btn_merge_staging_polygon)

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
        sam_group = QGroupBox("SAM模型")
        sam_layout = QVBoxLayout()
        sam_group.setLayout(sam_layout)

        parent.btn_load_sam = QPushButton("加载SAM模型")
        parent.btn_load_sam.clicked.connect(parent.load_sam_model)
        sam_layout.addWidget(parent.btn_load_sam)

        parent.btn_sam_train = QPushButton("开始训练")
        parent.btn_sam_train.clicked.connect(parent.start_sam_training)
        sam_layout.addWidget(parent.btn_sam_train)
        return sam_group

    @staticmethod
    def create_preannotation_toolbar(parent):
        preannotation_group = QGroupBox("预标注")
        preannotation_layout = QVBoxLayout()
        controls_layout = QGridLayout()
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)
        preannotation_group.setLayout(preannotation_layout)

        parent.btn_sam_preannotate = QPushButton("框选预标注")
        parent.btn_sam_preannotate.clicked.connect(parent.run_sam_preannotation)
        controls_layout.addWidget(parent.btn_sam_preannotate, 0, 0)

        parent.btn_sam_select_mode = QPushButton("接受候选并微调")
        parent.btn_sam_select_mode.clicked.connect(parent.enter_sam_select_mode)
        parent.btn_sam_select_mode.setEnabled(False)
        controls_layout.addWidget(parent.btn_sam_select_mode, 0, 1)

        controls_layout.addWidget(QLabel("当前理由"), 1, 0, 1, 2)
        parent.combo_preannotation_reason = NoWheelComboBox()
        parent.combo_preannotation_reason.addItem("未指定理由", "")
        parent.combo_preannotation_reason.addItem("被左侧植株遮挡", "occluded_by_left_plant")
        parent.combo_preannotation_reason.addItem("被右侧植株遮挡", "occluded_by_right_plant")
        parent.combo_preannotation_reason.addItem("被背景遮挡", "occluded_by_background")
        parent.combo_preannotation_reason.addItem("邻居误判", "neighbor_false_positive")
        parent.combo_preannotation_reason.addItem("背景误判", "background_false_positive")
        parent.combo_preannotation_reason.addItem("错误碎片", "wrong_fragment")
        parent.combo_preannotation_reason.addItem("穗茎分割", "ear_stem_segmentation")
        parent.combo_preannotation_reason.addItem("本实例不关注的叶片", "non_target_leaf_in_instance")
        parent.combo_preannotation_reason.addItem("被本实例遮挡", "occluded_by_this_instance")
        parent.combo_preannotation_reason.addItem("精修边界", "boundary_refinement")
        parent.combo_preannotation_reason.currentIndexChanged.connect(parent.on_preannotation_reason_changed)
        controls_layout.addWidget(parent.combo_preannotation_reason, 2, 0, 1, 2)

        parent.btn_save_staging_areas = QPushButton("拒绝当前 proposal")
        parent.btn_save_staging_areas.clicked.connect(parent.save_selected_staging_areas)
        parent.btn_save_staging_areas.setEnabled(False)
        controls_layout.addWidget(parent.btn_save_staging_areas, 3, 0)

        parent.btn_ignore_preannotation = QPushButton("忽略当前 proposal")
        parent.btn_ignore_preannotation.clicked.connect(parent.ignore_selected_preannotation)
        parent.btn_ignore_preannotation.setEnabled(False)
        controls_layout.addWidget(parent.btn_ignore_preannotation, 3, 1)

        parent.btn_export_preannotation_records = QPushButton("导出预标注调整记录")
        parent.btn_export_preannotation_records.clicked.connect(parent.export_preannotation_adjustments)
        controls_layout.addWidget(parent.btn_export_preannotation_records, 4, 0, 1, 2)

        parent.btn_import_preannotation_records = QPushButton("导入预标注记录目录")
        parent.btn_import_preannotation_records.clicked.connect(parent.import_preannotation_adjustments)
        controls_layout.addWidget(parent.btn_import_preannotation_records, 5, 0, 1, 2)

        controls_layout.addWidget(QLabel("已导入记录"), 6, 0, 1, 2)
        parent.combo_imported_preannotation_record = NoWheelComboBox()
        parent.combo_imported_preannotation_record.addItem("当前图片暂无已导入记录", "")
        parent.combo_imported_preannotation_record.setEnabled(False)
        parent.combo_imported_preannotation_record.currentIndexChanged.connect(parent.on_imported_preannotation_record_changed)
        controls_layout.addWidget(parent.combo_imported_preannotation_record, 7, 0, 1, 2)

        parent.btn_apply_imported_preannotation_record = QPushButton("恢复选中记录 final")
        parent.btn_apply_imported_preannotation_record.clicked.connect(parent.apply_selected_imported_preannotation_record)
        parent.btn_apply_imported_preannotation_record.setEnabled(False)
        controls_layout.addWidget(parent.btn_apply_imported_preannotation_record, 8, 0, 1, 2)

        controls_layout.addWidget(QLabel("理由片段"), 9, 0, 1, 2)
        parent.combo_imported_reason_segment = NoWheelComboBox()
        parent.combo_imported_reason_segment.addItem("当前记录暂无理由片段", "")
        parent.combo_imported_reason_segment.setEnabled(False)
        parent.combo_imported_reason_segment.currentIndexChanged.connect(parent.on_imported_reason_segment_changed)
        controls_layout.addWidget(parent.combo_imported_reason_segment, 10, 0, 1, 2)

        parent.label_imported_reason_segment_summary = QLabel("理由改动说明: 暂无")
        parent.label_imported_reason_segment_summary.setWordWrap(True)
        controls_layout.addWidget(parent.label_imported_reason_segment_summary, 11, 0, 1, 2)

        parent.btn_preview_reason_segment_start = QPushButton("预览理由前状态")
        parent.btn_preview_reason_segment_start.clicked.connect(parent.preview_selected_reason_segment_start)
        parent.btn_preview_reason_segment_start.setEnabled(False)
        controls_layout.addWidget(parent.btn_preview_reason_segment_start, 12, 0)

        parent.btn_preview_reason_segment_end = QPushButton("预览理由后状态")
        parent.btn_preview_reason_segment_end.clicked.connect(parent.preview_selected_reason_segment_end)
        parent.btn_preview_reason_segment_end.setEnabled(False)
        controls_layout.addWidget(parent.btn_preview_reason_segment_end, 12, 1)

        parent.btn_clear_reason_segment_preview = QPushButton("清除理由预览")
        parent.btn_clear_reason_segment_preview.clicked.connect(parent.clear_imported_reason_segment_preview)
        parent.btn_clear_reason_segment_preview.setEnabled(False)
        controls_layout.addWidget(parent.btn_clear_reason_segment_preview, 13, 0, 1, 2)

        preannotation_layout.addLayout(controls_layout)

        parent.sam_info_text = QTextEdit()
        parent.sam_info_text.setReadOnly(True)
        parent.sam_info_text.setMinimumHeight(64)
        parent.sam_info_text.setMaximumHeight(96)
        parent.sam_info_text.setPlaceholderText("SAM 信息将显示在这里...")
        preannotation_layout.addWidget(parent.sam_info_text)
        return preannotation_group

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


def _apply_toolbar_button_accents(parent):
    for button_name, accent in (
        ("btn_save_plant", "primary"),
        ("btn_sam_train", "primary"),
        ("btn_sam_preannotate", "primary"),
        ("btn_save_staging_areas", "danger"),
        ("btn_ignore_preannotation", "muted"),
        ("btn_delete", "danger"),
        ("btn_delete_staging_polygon", "danger"),
        ("btn_removal_region", "danger"),
        ("btn_refresh", "muted"),
        ("btn_toggle_annotation", "muted"),
        ("btn_prev", "muted"),
        ("btn_next", "muted"),
        ("btn_toggle_projection", "muted"),
        ("btn_help", "muted"),
        ("btn_debug_coco", "muted"),
        ("btn_timer_pause", "muted"),
    ):
        button = getattr(parent, button_name, None)
        if button is None:
            continue
        button.setProperty("accent", accent)
        if button.style():
            button.style().unpolish(button)
            button.style().polish(button)
