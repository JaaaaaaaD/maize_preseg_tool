import traceback

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from utils.data_manager import batch_export_annotations, batch_import_annotations


class MainWindowIOMixin:
    def import_batch_data(self):
        """批量导入COCO数据。"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择包含COCO文件的目录", self.import_path or ""
        )
        if not directory:
            return

        self.import_path = directory

        try:
            imported_count, skipped_count, error_count = batch_import_annotations(
                directory, self.image_paths, self.coco_container
            )

            if self.current_image_path:
                self.load_annotation_from_coco_container()
                self.sync_summary_view()
                self.update_plant_list()

            message = f"导入完成：成功 {imported_count}，跳过 {skipped_count}，错误 {error_count}"
            QMessageBox.information(self, "导入结果", message)
        except Exception as error:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误: {error}")
            traceback.print_exc()

    def export_annotated_images(self):
        """批量导出已完成图片为COCO格式。"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择导出目录", self.export_path or ""
        )
        if not directory:
            return

        self.export_path = directory

        try:
            class_names = self.project_metadata.get("class_names", []) if self.project_metadata else []
            exported_count, skipped_count, error_count = batch_export_annotations(
                directory, self.image_paths, self.coco_container, class_names
            )
            message = f"导出完成：成功 {exported_count}，跳过 {skipped_count}，错误 {error_count}"
            QMessageBox.information(self, "导出结果", message)
        except Exception as error:
            QMessageBox.critical(self, "导出失败", f"导出过程中发生错误: {error}")
            traceback.print_exc()

    def closeEvent(self, event):
        """关闭窗口事件。"""
        if self.sam_training_worker and self.sam_training_worker.isRunning():
            QMessageBox.information(self, "提示", "SAM训练仍在进行中，请等待训练结束后再关闭窗口")
            event.ignore()
            return
        event.accept()
