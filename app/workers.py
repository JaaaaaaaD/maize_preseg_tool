import copy

from PyQt5.QtCore import QThread, pyqtSignal


class SamTrainingWorker(QThread):
    """在后台线程中执行 SAM 训练，避免阻塞 UI。"""

    finished_signal = pyqtSignal(bool, str, str)

    def __init__(self, training_manager, coco_container, image_paths):
        super().__init__()
        self.training_manager = training_manager
        self.coco_container = copy.deepcopy(coco_container)
        self.image_paths = list(image_paths or [])

    def run(self):
        try:
            best_model_path = self.training_manager.train(
                self.coco_container,
                self.image_paths,
            )
            success = True
            message = f"训练完成，最佳模型已保存到: {best_model_path}"
        except Exception as error:
            success = False
            message = str(error)
            best_model_path = ""
        self.finished_signal.emit(success, message, best_model_path)
