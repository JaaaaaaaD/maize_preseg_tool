# 主入口文件

import sys
import traceback
import copy
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QShortcut, QProgressDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PIL import Image as PILImage

# 导入配置
from config import ANNOTATION_DIR, SAM_MODEL_PATH, SAM_MODEL_TYPE, SHORTCUTS, VERSION

# 导入组件
from components.image_label import ImageLabel
from components.help_dialog import HelpDialog
from components.toolbars import Toolbars

# 导入工具
from utils.image_processor import preprocess_image
from utils.data_manager import (
    save_current_annotation, load_current_annotation,
    export_simple_json, export_coco_format, export_annotated_images
)
from utils.helpers import load_image, format_image_progress

# 尝试导入SAM相关库
try:
    import torch
    from models.sam_model import SamModel
except ImportError:
    print("Warning: SAM library not found. Please install segment-anything package.")
    torch = None
    SamModel = None

# 全局异常捕获
sys.excepthook = lambda exctype, value, tb: traceback.print_exception(exctype, value, tb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玉米植株标注工具（性能优化+快捷键调整版）")
        self.setGeometry(100, 100, 1700, 800)

        self.image_paths = []
        self.current_image_index = -1
        self.annotation_dir = ANNOTATION_DIR
        self.image_annotation_status = {}  # 图片标注状态: {image_path: True/False}

        # 性能优化相关属性
        self.preprocess_cache = {}  # 单图预处理缓存：{image_path: (foreground_mask, edge_map)}
        self.annotation_changed = False  # 标注变化标记：仅True时自动保存

        self.current_image = None
        self.current_image_path = ""
        self.undo_stack = []
        self.redo_stack = []
        self.is_undo_redo = False
        
        # SAM模型相关
        self.sam_model = None
        self.sam_model_loaded = False
        self.sam_model_path = SAM_MODEL_PATH  # 默认模型路径
        self.sam_model_type = SAM_MODEL_TYPE  # 默认模型类型
        self.sam_segmenting = False  # 是否正在进行分割
        
        # 区域生长相关
        self.region_growing_enabled = False  # 是否启用区域生长

        self.init_ui()
        self.init_shortcuts()

        import os
        os.makedirs(self.annotation_dir, exist_ok=True)

        self.update_status_bar()

    def init_ui(self):
        """初始化UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 左侧：图像处理操作（纵向排列）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        main_layout.addWidget(left_panel, 1)

        # 添加工具栏
        left_layout.addWidget(Toolbars.create_auxiliary_toolbar(self))
        left_layout.addWidget(Toolbars.create_annotation_toolbar(self))
        left_layout.addWidget(Toolbars.create_plant_management_toolbar(self))
        left_layout.addWidget(Toolbars.create_navigation_toolbar(self))
        left_layout.addWidget(Toolbars.create_progress_label(self))

        left_layout.addStretch()

        # 中间：图像显示区域
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(center_panel, 5)

        images_layout = QHBoxLayout()
        center_layout.addLayout(images_layout)

        self.left_label = ImageLabel(is_summary=False, parent=self)
        self.left_label.setToolTip("左键添加顶点 | Enter暂存区域 | Shift+Enter保存整株 | 右键拖动 | 滚轮缩放")
        images_layout.addWidget(self.left_label, 1)

        self.right_label = ImageLabel(is_summary=True, parent=self)
        self.right_label.setToolTip("已保存植株总结预览")
        images_layout.addWidget(self.right_label, 1)

        # 右侧：文件操作（纵向排列）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        main_layout.addWidget(right_panel, 1)

        right_layout.addWidget(Toolbars.create_file_toolbar(self))
        right_layout.addWidget(Toolbars.create_export_toolbar(self))
        right_layout.addWidget(Toolbars.create_aux_toolbar(self))

        right_layout.addStretch()

    def init_shortcuts(self):
        """初始化快捷键"""
        QShortcut(QKeySequence(SHORTCUTS["SAVE_POLYGON"]), self, self.save_current_polygon)
        QShortcut(QKeySequence(SHORTCUTS["SAVE_PLANT"]), self, self.save_plant)
        QShortcut(QKeySequence(SHORTCUTS["UNDO"]), self, self.undo)
        QShortcut(QKeySequence(SHORTCUTS["DELETE_PLANT"]), self, self.delete_plant)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_EDGE_SNAP"]), self, self.toggle_edge_snap)
        QShortcut(QKeySequence(SHORTCUTS["LOAD_BATCH"]), self, self.load_batch_images)
        QShortcut(QKeySequence(SHORTCUTS["PREV_IMAGE"]), self, self.prev_image)
        QShortcut(QKeySequence(SHORTCUTS["NEXT_IMAGE"]), self, self.next_image)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_SAM_SEGMENTATION"]), self, self.toggle_sam_segmentation)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_REGION_GROWING"]), self, self.toggle_region_growing)

    def mark_annotation_changed(self):
        """标记当前图片的标注已修改，切换时自动保存"""
        self.annotation_changed = True

    def clear_annotation_changed(self):
        """清除标注变化标记"""
        self.annotation_changed = False

    def load_batch_images(self):
        """批量加载图片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "批量选择图片（可多选）",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not file_paths:
            return

        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.image_paths = [
            path for path in file_paths
            if path.lower().endswith(tuple(valid_extensions))
        ]

        if len(self.image_paths) == 0:
            QMessageBox.warning(self, "警告", "未选择有效的图片文件")
            return

        # 清空预处理缓存（避免旧数据干扰）
        self.preprocess_cache.clear()

        self.current_image_index = -1
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.goto_image(0)

        QMessageBox.information(
            self,
            "加载成功",
            f"成功加载 {len(self.image_paths)} 张图片\n仅当标注修改时自动保存，切换无卡顿"
        )

    def load_single_image(self):
        """加载单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单张图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = -1
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.preprocess_cache.clear()
            self.goto_image(0)

    def goto_image(self, index):
        """跳转到指定图片"""
        if index < 0 or index >= len(self.image_paths):
            return

        # 1. 仅当标注真的修改时才自动保存
        if self.current_image_index >= 0 and self.current_image is not None and self.annotation_changed:
            self.save_current_annotation_auto()
            self.clear_annotation_changed()

        # 2. 加载新图片
        new_image_path = self.image_paths[index]
        try:
            self.current_image = load_image(new_image_path)
            if not self.current_image:
                QMessageBox.warning(self, "警告", f"无法加载图片: {new_image_path}")
                return
            
            self.current_image_path = new_image_path
            self.current_image_index = index

            # 3. 优先从缓存获取预处理数据，否则重新计算并存入缓存
            preprocessed_data = self.preprocess_cache.get(new_image_path, None)
            if not preprocessed_data:
                # 临时用左侧画布计算预处理数据
                foreground_mask, edge_map = preprocess_image(self.current_image)
                preprocessed_data = (foreground_mask, edge_map)
                self.preprocess_cache[new_image_path] = preprocessed_data

            # 4. 重置画布（传入预处理数据）
            self.left_label.set_image(self.current_image, preprocessed_data)
            self.right_label.set_image(self.current_image, preprocessed_data)

            # 5. 尝试加载已有标注
            self.load_current_annotation_auto()

            # 6. 更新UI
            self.update_plant_list()
            self.clear_undo_stack()
            self.clear_annotation_changed()  # 新图片初始状态无变化
            # 更新标注状态按钮文本
            is_annotated = self.image_annotation_status.get(self.current_image_path, False)
            if is_annotated:
                self.btn_toggle_annotation.setText("标记为未标注")
            else:
                self.btn_toggle_annotation.setText("标记为已标注")
            self.update_status_bar()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
            traceback.print_exc()

    def prev_image(self):
        """上一张图片"""
        if self.current_image_index > 0:
            self.goto_image(self.current_image_index - 1)

    def next_image(self):
        """下一张图片"""
        if self.current_image_index < len(self.image_paths) - 1:
            self.goto_image(self.current_image_index + 1)

    def save_current_annotation_auto(self):
        """自动保存当前标注"""
        save_current_annotation(
            self.current_image_path,
            self.left_label.plants,
            self.left_label.current_plant_id
        )

    def load_current_annotation_auto(self):
        """自动加载当前标注"""
        annotation_data = load_current_annotation(self.current_image_path)
        if annotation_data:
            self.left_label.plants = annotation_data["plants"]
            self.left_label.current_plant_id = annotation_data["current_plant_id"]
            # 右侧画布直接同步左侧初始状态
            self.right_label.plants = self.left_label.plants
            self.right_label.current_plant_id = self.left_label.current_plant_id

    def toggle_edge_snap(self):
        """切换边缘吸附"""
        self.left_label.edge_snap_enabled = not self.left_label.edge_snap_enabled
        self.update_snap_button_state()
        if not self.left_label.edge_snap_enabled:
            self.left_label.current_snap_point = None
            self.left_label.update_display()
        self.update_status_bar()

    def update_snap_button_state(self):
        """更新边缘吸附按钮的状态"""
        if self.left_label.edge_snap_enabled:
            self.btn_toggle_snap.setText(f"边缘吸附: 开启 ({SHORTCUTS['TOGGLE_EDGE_SNAP']})")
        else:
            self.btn_toggle_snap.setText(f"边缘吸附: 关闭 ({SHORTCUTS['TOGGLE_EDGE_SNAP']})")

    def load_sam_model(self):
        """加载SAM模型"""
        if SamModel is None:
            QMessageBox.warning(self, "警告", "SAM库未安装，请安装segment-anything包")
            return
        
        # 选择模型文件
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM模型文件", ".", "PTH文件 (*.pth)"
        )
        
        if not model_path:
            return
        
        # 显示加载进度
        progress = QProgressDialog("正在加载SAM模型...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        
        try:
            # 初始化SAM模型
            self.sam_model = SamModel(model_type=self.sam_model_type)
            
            # 加载模型
            progress.setValue(20)
            success = self.sam_model.load_model(model_path)
            
            if success:
                progress.setValue(100)
                QMessageBox.information(self, "成功", f"SAM模型加载成功！使用设备: {self.sam_model.get_device()}")
                self.sam_model_loaded = True
                # 启用分割按钮
                self.btn_sam_segment.setEnabled(True)
            else:
                QMessageBox.critical(self, "错误", "加载SAM模型失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载SAM模型失败: {str(e)}")
        finally:
            progress.close()
    
    def toggle_sam_segmentation(self):
        """切换SAM分割模式"""
        if not self.sam_model_loaded:
            QMessageBox.warning(self, "警告", "请先加载SAM模型")
            return
        
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        
        # 确保退出区域生长模式
        if self.region_growing_enabled:
            self.toggle_region_growing()
        
        # 切换分割模式
        self.sam_segmenting = not self.sam_segmenting
        
        if self.sam_segmenting:
            self.btn_sam_segment.setText(f"退出分割 ({SHORTCUTS['TOGGLE_SAM_SEGMENTATION']})")
            self.left_label.sam_segmenting = True
            self.left_label.sam_predictor = self.sam_model.predictor
            self.left_label.sam_prompt_points = []
            # 设置当前图像到SAM预测器
            import numpy as np
            img_np = np.array(self.current_image.convert("RGB"))
            self.sam_model.set_image(img_np)
        else:
            self.btn_sam_segment.setText(f"SAM分割 ({SHORTCUTS['TOGGLE_SAM_SEGMENTATION']})")
            self.left_label.sam_segmenting = False
            self.left_label.sam_prompt_points = []
        
        self.left_label.update_display()
        self.update_status_bar()
    
    def toggle_region_growing(self):
        """切换膨胀点选模式"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        
        # 确保退出SAM分割模式
        if self.sam_segmenting:
            self.toggle_sam_segmentation()
        
        # 切换膨胀点选模式
        self.region_growing_enabled = not self.region_growing_enabled
        
        if self.region_growing_enabled:
            self.btn_region_growing.setText(f"退出膨胀点选 ({SHORTCUTS['TOGGLE_REGION_GROWING']})")
            self.left_label.region_growing_enabled = True
        else:
            self.btn_region_growing.setText(f"膨胀点选 ({SHORTCUTS['TOGGLE_REGION_GROWING']})")
            self.left_label.region_growing_enabled = False
            self.left_label.region_growing_mask = None
        
        self.left_label.update_display()
        self.update_status_bar()

    def show_help(self):
        """显示使用说明"""
        dialog = HelpDialog(self)
        dialog.exec_()

    def save_current_polygon(self):
        """保存当前多边形"""
        if self.left_label.save_current_polygon():
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()

    def save_plant(self):
        """保存整株"""
        saved_id = self.left_label.confirm_preview_and_save()
        if saved_id:
            saved_plant = next((p for p in self.left_label.plants if p["id"] == saved_id), None)
            if saved_plant:
                self.push_undo_action("add_plant", saved_plant)
            # 右侧画布按需同步
            self.right_label.plants = copy.deepcopy(self.left_label.plants)
            self.right_label.current_plant_id = self.left_label.current_plant_id
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
        else:
            QMessageBox.warning(self, "警告", "请先暂存至少一个区域")

    def undo(self):
        """撤销操作"""
        if self.left_label.undo_last_action():
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
            return
        if not self.undo_stack:
            return
        self.is_undo_redo = True
        try:
            action = self.undo_stack.pop()
            if action["type"] == "add_plant":
                plant = action["data"]
                self.left_label.delete_plant(plant["id"])
                self.right_label.delete_plant(plant["id"])
                self.redo_stack.append(action)
            elif action["type"] == "delete_plant":
                plant = action["data"]
                self.left_label.plants.append(plant)
                self.left_label.plants.sort(key=lambda x: x["id"])
                self.right_label.plants.append(copy.deepcopy(plant))
                self.right_label.plants.sort(key=lambda x: x["id"])
                self.redo_stack.append(action)
            self.update_plant_list()
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
        finally:
            self.is_undo_redo = False

    def delete_plant(self):
        """删除选中植株"""
        selected_text = self.combo_plants.currentText()
        if not selected_text:
            return
        try:
            parts = selected_text.split()
            if len(parts) < 2:
                return
            plant_id = int(parts[1])
            plant = next((p for p in self.left_label.plants if p["id"] == plant_id), None)
            if plant:
                self.push_undo_action("delete_plant", plant)
                self.left_label.delete_plant(plant_id)
                # 同步右侧视图
                self.sync_summary_view()
                self.update_plant_list()
                self.update_undo_redo_state()
                self.mark_annotation_changed()  # 标记变化
                self.update_status_bar()
        except (ValueError, IndexError):
            pass

    def on_plant_selected(self, text):
        """选择植株"""
        try:
            if not text:
                return
            parts = text.split()
            if len(parts) < 2:
                return
            plant_id = int(parts[1])
            self.left_label.select_plant(plant_id)
            # 同步右侧全局概览的选中状态
            self.right_label.select_plant(plant_id)
        except (ValueError, IndexError):
            pass

    def update_plant_list(self):
        """更新植株列表"""
        self.combo_plants.clear()
        for plant in self.left_label.plants:
            self.combo_plants.addItem(f"植株 {plant['id']}")

    def update_undo_redo_state(self):
        """更新撤销重做状态"""
        pass  # 可以根据需要添加实现

    def push_undo_action(self, action_type, data):
        """添加撤销操作"""
        self.undo_stack.append({"type": action_type, "data": data})
        self.redo_stack.clear()

    def clear_undo_stack(self):
        """清空撤销栈"""
        self.undo_stack.clear()
        self.redo_stack.clear()

    def update_status_bar(self):
        """更新状态栏"""
        base_msg = "就绪"

        if len(self.image_paths) > 0 and self.current_image_index >= 0:
            progress_msg = f" | 图片: {self.current_image_index + 1}/{len(self.image_paths)}"
            image_name = os.path.basename(self.current_image_path)
            # 检查当前图片的标注状态
            is_annotated = self.image_annotation_status.get(self.current_image_path, False)
            status_str = "[已标注]" if is_annotated else "[未标注]"
            base_msg += f" | 当前: {image_name} {status_str}"
            base_msg += progress_msg
            # 更新图片进度标签
            self.image_progress_label.setText(f"{self.current_image_index + 1}/{len(self.image_paths)}")
        else:
            # 没有图片时重置标签
            self.image_progress_label.setText("0/0")

        if hasattr(self.left_label, 'edge_snap_enabled'):
            snap_msg = " | 边缘吸附: 开启" if self.left_label.edge_snap_enabled else " | 边缘吸附: 关闭"
            base_msg += snap_msg

        if hasattr(self.left_label, 'current_points') and hasattr(self.left_label, 'current_plant_polygons'):
            if len(self.left_label.current_points) > 0 or len(self.left_label.current_plant_polygons) > 0:
                state_msg = f" | 当前多边形顶点: {len(self.left_label.current_points)} | 已暂存区域: {len(self.left_label.current_plant_polygons)}"
                base_msg += state_msg

        if hasattr(self.left_label, 'plants'):
            base_msg += f" | 已标注植株: {len(self.left_label.plants)}"

        # 显示标注变化状态
        if self.annotation_changed:
            base_msg += " | 【有未保存的修改】"

        # 显示当前模式
        if self.sam_segmenting:
            base_msg += " | SAM分割模式"
        elif self.region_growing_enabled:
            base_msg += " | 膨胀点选模式"

        self.statusBar().showMessage(base_msg)

    def toggle_annotation_status(self):
        """切换标注状态"""
        if not self.current_image_path:
            return
        current_status = self.image_annotation_status.get(self.current_image_path, False)
        new_status = not current_status
        self.image_annotation_status[self.current_image_path] = new_status
        if new_status:
            self.btn_toggle_annotation.setText("标记为未标注")
        else:
            self.btn_toggle_annotation.setText("标记为已标注")
        self.update_status_bar()

    def export_simple_json(self):
        """导出为简单JSON格式"""
        if not self.current_image_path or not self.left_label.plants:
            QMessageBox.warning(self, "警告", "没有可导出的标注数据")
            return
        export_path = export_simple_json(self.current_image_path, self.left_label.plants)
        if export_path:
            QMessageBox.information(self, "成功", f"JSON格式导出成功：{export_path}")

    def export_coco_format(self):
        """导出为COCO格式"""
        if not self.current_image_path or not self.left_label.plants:
            QMessageBox.warning(self, "警告", "没有可导出的标注数据")
            return
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        export_path = export_coco_format(
            self.current_image_path,
            self.left_label.plants,
            self.current_image.width,
            self.current_image.height
        )
        if export_path:
            QMessageBox.information(self, "成功", f"COCO格式导出成功：{export_path}")

    def export_annotated_images(self):
        """批量导出已标注的图片"""
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        export_dir, exported_count = export_annotated_images(self.image_paths, self.image_annotation_status)
        if export_dir:
            QMessageBox.information(self, "成功", f"批量导出完成，共导出 {exported_count} 个已标注图片到：{export_dir}")
        else:
            QMessageBox.warning(self, "警告", "没有已标注的图片可导出")

    def save_undo_state(self):
        """保存撤销状态"""
        pass  # 可以根据需要添加实现

    def sync_summary_view(self):
        """同步总结视图"""
        self.right_label.plants = copy.deepcopy(self.left_label.plants)
        self.right_label.current_plant_id = self.left_label.current_plant_id
        self.right_label.selected_plant_id = self.left_label.selected_plant_id
        self.right_label.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())