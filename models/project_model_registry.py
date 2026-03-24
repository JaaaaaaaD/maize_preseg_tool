# 项目级模型版本注册表
import copy
import os

from utils.annotation_schema import current_timestamp
from utils.project_context import load_json_file, save_json_file, get_project_paths


class ProjectModelRegistry:
    """管理单个项目的 active / previous stable 模型。"""

    def __init__(self, project_id):
        self.project_id = project_id
        self.paths = get_project_paths(project_id)
        self.registry_path = self.paths["model_registry_path"]
        self.versions_root = self.paths["model_versions_root"]
        os.makedirs(self.versions_root, exist_ok=True)
        self.registry = self._load_registry()

    def _default_registry(self):
        return {
            "project_id": self.project_id,
            "active_model_version": None,
            "previous_model_version": None,
            "last_training_status": "idle",
            "last_training_message": "",
            "last_successful_train_at": None,
            "versions": {},
            "created_at": current_timestamp(),
            "updated_at": current_timestamp(),
        }

    def _load_registry(self):
        registry = load_json_file(self.registry_path, self._default_registry())
        registry.setdefault("versions", {})
        registry.setdefault("active_model_version", None)
        registry.setdefault("previous_model_version", None)
        registry.setdefault("last_training_status", "idle")
        registry.setdefault("last_training_message", "")
        registry.setdefault("last_successful_train_at", None)
        return registry

    def save(self):
        """持久化 registry。"""
        self.registry["updated_at"] = current_timestamp()
        save_json_file(self.registry_path, self.registry)

    def reload(self):
        """重新加载 registry。"""
        self.registry = self._load_registry()
        return self.registry

    def get_active_version(self):
        return self.registry.get("active_model_version")

    def get_previous_version(self):
        return self.registry.get("previous_model_version")

    def get_active_model_path(self):
        version_name = self.get_active_version()
        if not version_name:
            return None
        version_info = self.registry.get("versions", {}).get(version_name, {})
        best_path = version_info.get("best_path")
        if best_path and os.path.exists(best_path):
            return best_path
        return None

    def has_active_model(self):
        return bool(self.get_active_model_path())

    def list_versions(self):
        versions = list(self.registry.get("versions", {}).values())
        versions.sort(key=lambda item: item.get("version_name", ""))
        return versions

    def allocate_next_version_name(self):
        """分配下一个版本号。"""
        max_index = 0
        for version_name in self.registry.get("versions", {}):
            if version_name.startswith("model_v"):
                try:
                    max_index = max(max_index, int(version_name.split("model_v", 1)[1]))
                except ValueError:
                    continue
        return f"model_v{max_index + 1:04d}"

    def create_version_slot(self, version_name, sample_count, init_weights, train_args):
        """训练开始前预创建版本目录和元数据。"""
        version_dir = os.path.join(self.versions_root, version_name)
        os.makedirs(version_dir, exist_ok=True)

        version_info = {
            "version_name": version_name,
            "version_dir": version_dir,
            "status": "running",
            "created_at": current_timestamp(),
            "updated_at": current_timestamp(),
            "sample_count": int(sample_count or 0),
            "init_weights": init_weights,
            "best_path": os.path.join(version_dir, "best.pt"),
            "last_path": os.path.join(version_dir, "last.pt"),
            "metrics_summary_path": os.path.join(version_dir, "metrics_summary.json"),
            "train_log_path": os.path.join(version_dir, "train.log"),
            "train_args_path": os.path.join(version_dir, "train_args.yaml"),
            "metrics_summary": None,
            "stable": False,
            "active": False,
            "train_args": copy.deepcopy(train_args or {}),
        }
        self.registry["versions"][version_name] = version_info
        self.registry["last_training_status"] = "running"
        self.registry["last_training_message"] = f"训练中: {version_name}"
        self.save()
        return version_info

    def mark_training_failed(self, version_name, message):
        """训练失败，保留旧 active 模型。"""
        version_info = self.registry.get("versions", {}).get(version_name)
        if version_info:
            version_info["status"] = "failed"
            version_info["updated_at"] = current_timestamp()
            version_info["error_message"] = message
        self.registry["last_training_status"] = "failed"
        self.registry["last_training_message"] = message or "训练失败"
        self.save()

    def promote_version(self, version_name, metrics_summary=None):
        """将训练成功版本切换为 active。"""
        if version_name not in self.registry.get("versions", {}):
            raise ValueError(f"Unknown model version: {version_name}")

        current_active = self.registry.get("active_model_version")
        version_info = self.registry["versions"][version_name]
        version_info["status"] = "ready"
        version_info["stable"] = True
        version_info["active"] = True
        version_info["updated_at"] = current_timestamp()
        if metrics_summary is not None:
            version_info["metrics_summary"] = metrics_summary

        if current_active and current_active in self.registry["versions"] and current_active != version_name:
            self.registry["versions"][current_active]["active"] = False
            self.registry["versions"][current_active]["stable"] = True
            self.registry["previous_model_version"] = current_active

        self.registry["active_model_version"] = version_name
        self.registry["last_training_status"] = "idle"
        self.registry["last_training_message"] = f"当前模型: {version_name}"
        self.registry["last_successful_train_at"] = current_timestamp()
        self.save()
        return version_info

    def rollback_to_previous(self):
        """一键回退到上一稳定版本，只切换 active，不改动已有标注。"""
        active_version = self.registry.get("active_model_version")
        previous_version = self.registry.get("previous_model_version")

        if not previous_version or previous_version not in self.registry.get("versions", {}):
            return False, "没有可回退的历史稳定模型"

        if active_version and active_version in self.registry["versions"]:
            self.registry["versions"][active_version]["active"] = False
            self.registry["versions"][active_version]["stable"] = True

        self.registry["versions"][previous_version]["active"] = True
        self.registry["versions"][previous_version]["stable"] = True
        self.registry["active_model_version"] = previous_version
        self.registry["previous_model_version"] = active_version
        self.registry["last_training_status"] = "idle"
        self.registry["last_training_message"] = f"已回退到 {previous_version}"
        self.save()
        return True, previous_version
