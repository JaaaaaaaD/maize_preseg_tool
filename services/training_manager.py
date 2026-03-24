# 项目级训练管理器
import json
import os
import re
import sys

from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal

from config import (
    AUTO_TRAIN_THRESHOLD,
    TRAIN_BATCH,
    TRAIN_DEVICE,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_WORKERS,
    YOLO_DEFAULT_MODEL,
)
from models.project_model_registry import ProjectModelRegistry
from utils.annotation_schema import current_timestamp
from utils.dataset_builder import build_project_dataset
from utils.project_context import (
    mark_training_failed,
    mark_training_started,
    mark_training_success,
    refresh_project_counters,
    update_project_versions,
)


class TrainingManager(QObject):
    """管理自动训练阈值、后台训练进程、版本切换和失败回退。"""

    training_state_changed = pyqtSignal(str)
    training_progress_changed = pyqtSignal(int, str)
    training_finished = pyqtSignal(bool, str)
    active_model_changed = pyqtSignal(str)
    project_counts_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_id = None
        self.registry = None
        self.process = None
        self.pending_retrain = False
        self.current_job = None
        self._stdout_buffer = ""

    def set_project(self, project_id):
        """切换当前训练上下文。"""
        self.project_id = project_id
        self.registry = ProjectModelRegistry(project_id) if project_id else None
        self.pending_retrain = False
        self.current_job = None
        self._emit_project_counts()
        self._emit_current_model_state()

    def _emit_current_model_state(self):
        if not self.registry:
            self.active_model_changed.emit("暂无模型")
            return
        version = self.registry.get_active_version()
        self.active_model_changed.emit(version or "暂无模型")

    def _emit_project_counts(self):
        if not self.project_id:
            self.project_counts_changed.emit(0, 0)
            return
        metadata = refresh_project_counters(self.project_id)
        self.project_counts_changed.emit(
            int(metadata.get("completed_image_count", 0)),
            int(metadata.get("dirty_completed_image_count", 0)),
        )

    def has_active_model(self):
        return bool(self.registry and self.registry.get_active_model_path())

    def get_active_model_path(self):
        if not self.registry:
            return None
        return self.registry.get_active_model_path()

    def get_active_model_version(self):
        if not self.registry:
            return None
        return self.registry.get_active_version()

    def _sanitize_ui_message(self, text, fallback="训练失败", max_length=160):
        """清理训练日志中的控制字符，避免异常文本污染界面布局。"""
        if text is None:
            return fallback
        sanitized = str(text)
        sanitized = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", sanitized)
        sanitized = sanitized.replace("\r", " ").replace("\n", " ")
        sanitized = re.sub(r"[\x00-\x1f\x7f]+", " ", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if not sanitized:
            return fallback
        if len(sanitized) > max_length:
            return sanitized[: max_length - 3].rstrip() + "..."
        return sanitized

    def is_training(self):
        return bool(self.process and self.process.state() != QProcess.NotRunning)

    def rollback_to_previous(self):
        """回退到上一稳定模型。"""
        if not self.registry:
            return False, "当前没有项目"
        success, payload = self.registry.rollback_to_previous()
        payload = self._sanitize_ui_message(payload, fallback="回退完成")
        if success:
            update_project_versions(
                self.project_id,
                self.registry.get_active_version(),
                self.registry.get_previous_version(),
            )
            self._emit_current_model_state()
            self.training_state_changed.emit(f"已回退到 {payload}")
            self.training_finished.emit(True, f"已回退到 {payload}")
            return True, payload
        self.training_finished.emit(False, payload)
        return False, payload

    def check_and_trigger_training(self, force=False, rebuild_split=False):
        """检查阈值并启动训练。"""
        if not self.project_id or not self.registry:
            return False, "当前没有活动项目"

        metadata = refresh_project_counters(self.project_id)
        completed_count = int(metadata.get("completed_image_count", 0))
        dirty_count = int(metadata.get("dirty_completed_image_count", 0))
        threshold = int(metadata.get("auto_train_threshold", AUTO_TRAIN_THRESHOLD))

        if self.is_training():
            self.pending_retrain = self.pending_retrain or dirty_count >= threshold or force
            return False, "训练已在进行中"

        if completed_count <= 0:
            return False, "当前项目还没有已完成图片"

        if not force and dirty_count < threshold:
            return False, f"dirty completed 数量不足 {threshold}"

        return self.start_training(rebuild_split=rebuild_split)

    def start_training(self, rebuild_split=False):
        """启动一次新的训练任务。"""
        if not self.project_id or not self.registry:
            return False, "当前没有活动项目"
        if self.is_training():
            self.pending_retrain = True
            return False, "训练已在进行中"

        try:
            dataset_info = build_project_dataset(self.project_id, rebuild_split=rebuild_split)
        except Exception as error:
            message = self._sanitize_ui_message(f"构建数据集失败: {error}", fallback="构建数据集失败")
            mark_training_failed(self.project_id, message)
            self.training_state_changed.emit(message)
            self.training_finished.emit(False, message)
            return False, message

        version_name = self.registry.allocate_next_version_name()
        init_weights = self.registry.get_active_model_path() or YOLO_DEFAULT_MODEL
        train_args = {
            "project_id": self.project_id,
            "version_name": version_name,
            "created_at": current_timestamp(),
            "data_yaml": dataset_info["data_yaml_path"],
            "dataset_root": dataset_info["dataset_root"],
            "completed_count": dataset_info["completed_count"],
            "epochs": TRAIN_EPOCHS,
            "imgsz": TRAIN_IMGSZ,
            "batch": TRAIN_BATCH,
            "device": TRAIN_DEVICE,
            "workers": TRAIN_WORKERS,
            "init_weights": init_weights,
        }

        version_info = self.registry.create_version_slot(
            version_name,
            dataset_info["completed_count"],
            init_weights,
            train_args,
        )
        self._write_train_args(version_info["train_args_path"], train_args)
        mark_training_started(self.project_id, f"训练中: {version_name}")
        self._emit_project_counts()

        self.current_job = {
            "version_name": version_name,
            "version_dir": version_info["version_dir"],
            "train_log_path": version_info["train_log_path"],
            "metrics_summary_path": version_info["metrics_summary_path"],
            "snapshot_hashes": dataset_info["snapshot_hashes"],
            "script_result": None,
        }
        self._stdout_buffer = ""
        start_message = self._sanitize_ui_message(f"开始训练 {version_name}", fallback="开始训练")
        self.training_state_changed.emit(start_message)
        self.training_progress_changed.emit(0, start_message)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_process_output)
        self.process.finished.connect(self._handle_process_finished)

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools", "train_project_model.py"))
        args = [
            script_path,
            "--project_root",
            os.path.abspath(self.registry.paths["project_root"]),
            "--data_yaml",
            os.path.abspath(dataset_info["data_yaml_path"]),
            "--init_weights",
            str(init_weights),
            "--run_name",
            version_name,
            "--epochs",
            str(TRAIN_EPOCHS),
            "--imgsz",
            str(TRAIN_IMGSZ),
            "--batch",
            str(TRAIN_BATCH),
            "--device",
            str(TRAIN_DEVICE),
            "--workers",
            str(TRAIN_WORKERS),
            "--output_dir",
            os.path.abspath(version_info["version_dir"]),
        ]
        self.process.start(sys.executable, args)
        return True, version_name

    def _write_train_args(self, train_args_path, train_args):
        """写出训练参数，方便追溯。"""
        lines = []
        for key, value in train_args.items():
            lines.append(f"{key}: {value}")
        os.makedirs(os.path.dirname(train_args_path), exist_ok=True)
        with open(train_args_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    def _handle_process_output(self):
        """解析训练脚本 stdout。"""
        if not self.process or not self.current_job:
            return
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not output:
            return

        self._stdout_buffer += output
        lines = self._stdout_buffer.splitlines(keepends=False)
        if self._stdout_buffer and not self._stdout_buffer.endswith(("\n", "\r")):
            self._stdout_buffer = lines.pop() if lines else self._stdout_buffer
        else:
            self._stdout_buffer = ""

        for line in lines:
            self._append_train_log(line)
            self._parse_output_line(line.strip())

    def _append_train_log(self, line):
        """将训练输出附加到 train.log。"""
        if not self.current_job:
            return
        with open(self.current_job["train_log_path"], "a", encoding="utf-8") as file:
            file.write(line + "\n")

    def _parse_output_line(self, line):
        """解析训练脚本约定的状态行。"""
        if not line:
            return
        if line.startswith("STATUS|"):
            _, stage, message = (line.split("|", 2) + [""])[:3]
            text = self._sanitize_ui_message(message or stage, fallback="训练中")
            self.training_state_changed.emit(text)
            return

        if line.startswith("PROGRESS|"):
            parts = line.split("|", 3)
            if len(parts) >= 4:
                try:
                    current_epoch = int(parts[1])
                    total_epoch = max(1, int(parts[2]))
                    text = self._sanitize_ui_message(parts[3], fallback="训练中")
                    progress = int((current_epoch / total_epoch) * 100)
                    self.training_progress_changed.emit(progress, text)
                except ValueError:
                    pass
            return

        if line.startswith("RESULT|"):
            if line.startswith("RESULT|failure|"):
                self.current_job["script_result"] = [
                    "RESULT",
                    "failure",
                    self._sanitize_ui_message(line.split("|", 2)[2], fallback="训练失败"),
                ]
                return
            if line.startswith("RESULT|success|"):
                parts = line.split("|", 4)
                self.current_job["script_result"] = parts
                return
            self.current_job["script_result"] = ["RESULT", "failure", "训练失败"]
            return

    def _handle_process_finished(self, exit_code, exit_status):
        """处理训练进程结束。"""
        if not self.current_job:
            return

        if self._stdout_buffer:
            self._append_train_log(self._stdout_buffer)
            self._parse_output_line(self._stdout_buffer.strip())
            self._stdout_buffer = ""

        result_parts = self.current_job.get("script_result") or []
        version_name = self.current_job["version_name"]
        success = bool(exit_code == 0 and result_parts and len(result_parts) >= 2 and result_parts[1] == "success")

        if success:
            metrics_summary = self._load_metrics_summary(self.current_job["metrics_summary_path"])
            self.registry.promote_version(version_name, metrics_summary=metrics_summary)
            update_project_versions(
                self.project_id,
                self.registry.get_active_version(),
                self.registry.get_previous_version(),
            )
            mark_training_success(self.project_id, version_name, self.current_job["snapshot_hashes"])
            success_message = self._sanitize_ui_message(f"训练完成: {version_name}", fallback="训练完成")
            self.training_state_changed.emit(success_message)
            self.training_progress_changed.emit(100, success_message)
            self.training_finished.emit(True, success_message)
            self._emit_current_model_state()
        else:
            failure_message = "训练失败"
            if result_parts and len(result_parts) >= 3:
                failure_message = self._sanitize_ui_message(result_parts[2], fallback="训练失败")
            self.registry.mark_training_failed(version_name, failure_message)
            mark_training_failed(self.project_id, failure_message)
            self.training_state_changed.emit(failure_message)
            self.training_finished.emit(False, failure_message)

        self.process.deleteLater()
        self.process = None
        self.current_job = None
        self._emit_project_counts()

        if self.pending_retrain:
            self.pending_retrain = False
            QTimer.singleShot(0, lambda: self.check_and_trigger_training(force=False))

    def _load_metrics_summary(self, metrics_path):
        """读取 metrics summary。"""
        if not os.path.exists(metrics_path):
            return None
        try:
            with open(metrics_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return None
