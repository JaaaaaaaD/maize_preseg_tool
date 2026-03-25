# 项目级目录和元数据管理
import hashlib
import json
import os

from config import (
    DEFAULT_CLASS_NAMES,
    FIXED_VAL_MIN_COUNT,
    FIXED_VAL_RATIO,
    FIXED_VAL_SEED,
    AUTO_TRAIN_THRESHOLD,
)
from utils.annotation_schema import current_timestamp

LEGACY_AUTO_TRAIN_THRESHOLD = 20
LEGACY_FIXED_VAL_MIN_COUNT = 5


def _safe_name(name):
    """将项目名规整成适合目录名的格式。"""
    valid_chars = []
    for char in name or "project":
        if char.isalnum() or char in ("-", "_"):
            valid_chars.append(char)
        else:
            valid_chars.append("_")
    safe = "".join(valid_chars).strip("_")
    return safe or "project"


def get_source_root_from_images(image_paths):
    """基于当前图片集合计算项目来源目录。"""
    if not image_paths:
        return os.getcwd()
    common_path = os.path.commonpath(image_paths)
    if os.path.isfile(common_path):
        common_path = os.path.dirname(common_path)
    return os.path.abspath(common_path)


def build_project_id(source_root):
    """根据来源目录计算稳定 project_id。"""
    source_root = os.path.abspath(source_root)
    base_name = os.path.basename(source_root.rstrip("\\/")) or "project"
    digest = hashlib.sha1(source_root.encode("utf-8")).hexdigest()[:8]
    return f"{_safe_name(base_name)}_{digest}"


def get_project_paths(project_id):
    """返回项目关键路径。"""
    # 使用当前工作目录作为项目根目录
    project_root = os.path.join(os.getcwd(), f"project_{project_id}")
    return {
        "project_root": project_root,
        "metadata_path": os.path.join(project_root, "project_metadata.json"),
        "image_records_path": os.path.join(project_root, "image_records.json"),
        "models_root": os.path.join(project_root, "models"),
        "model_versions_root": os.path.join(project_root, "models", "versions"),
        "model_registry_path": os.path.join(project_root, "models", "registry.json"),
        "dataset_root": os.path.join(project_root, "dataset"),
        "split_manifest_path": os.path.join(project_root, "split_manifest.json"),
        "logs_root": os.path.join(project_root, "logs"),
        "exports_root": os.path.join(project_root, "exports"),
    }


def _ensure_project_layout(paths):
    """创建项目目录结构。"""
    # 不再创建PROJECTS_ROOT目录，直接创建项目相关目录
    os.makedirs(paths["project_root"], exist_ok=True)
    os.makedirs(paths["models_root"], exist_ok=True)
    os.makedirs(paths["model_versions_root"], exist_ok=True)
    os.makedirs(paths["dataset_root"], exist_ok=True)
    os.makedirs(paths["logs_root"], exist_ok=True)
    os.makedirs(paths["exports_root"], exist_ok=True)


def _default_project_metadata(project_id, source_root, class_names):
    """返回默认项目元数据。"""
    return {
        "project_id": project_id,
        "project_name": os.path.basename(source_root.rstrip("\\/")) or project_id,
        "source_root": source_root,
        "class_names": list(class_names or DEFAULT_CLASS_NAMES),
        "active_model_version": None,
        "previous_model_version": None,
        "last_successful_train_at": None,
        "completed_image_count": 0,
        "dirty_completed_image_count": 0,
        "fixed_val_manifest": None,
        "bootstrap_done": False,
        "last_training_status": "idle",
        "last_training_message": "",
        "fixed_val_ratio": FIXED_VAL_RATIO,
        "fixed_val_min_count": FIXED_VAL_MIN_COUNT,
        "fixed_val_seed": FIXED_VAL_SEED,
        "auto_train_threshold": AUTO_TRAIN_THRESHOLD,
        "created_at": current_timestamp(),
        "updated_at": current_timestamp(),
    }


def _safe_int(value, default=None):
    """尽量将值转成 int，失败时回退到默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json_file(path, default_value):
    """读取 json 文件，失败时返回默认值。"""
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default_value


def save_json_file(path, payload):
    """写入 json 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def ensure_project_for_images(image_paths, class_names=None):
    """根据当前图片集合准备项目目录与元数据。"""
    source_root = get_source_root_from_images(image_paths)
    project_id = build_project_id(source_root)
    paths = get_project_paths(project_id)
    _ensure_project_layout(paths)

    metadata = load_json_file(paths["metadata_path"], None)
    if not metadata:
        metadata = _default_project_metadata(project_id, source_root, class_names or DEFAULT_CLASS_NAMES)
        save_json_file(paths["metadata_path"], metadata)

    image_records = load_json_file(paths["image_records_path"], {})
    if not isinstance(image_records, dict):
        image_records = {}
        save_json_file(paths["image_records_path"], image_records)

    # 类别配置随项目保留；若旧项目没有该字段，补默认值。
    metadata.setdefault("class_names", list(class_names or DEFAULT_CLASS_NAMES))
    metadata.setdefault("source_root", source_root)
    metadata.setdefault("project_name", os.path.basename(source_root.rstrip("\\/")) or project_id)
    metadata.setdefault("fixed_val_ratio", FIXED_VAL_RATIO)
    metadata.setdefault("fixed_val_seed", FIXED_VAL_SEED)

    # 将旧项目的默认阈值迁移到当前版本；如果用户后续手工改过别的值，则保留其配置。
    current_min_val = _safe_int(metadata.get("fixed_val_min_count"))
    if current_min_val is None or current_min_val == LEGACY_FIXED_VAL_MIN_COUNT:
        metadata["fixed_val_min_count"] = FIXED_VAL_MIN_COUNT
    else:
        metadata.setdefault("fixed_val_min_count", FIXED_VAL_MIN_COUNT)

    current_train_threshold = _safe_int(metadata.get("auto_train_threshold"))
    if current_train_threshold is None or current_train_threshold == LEGACY_AUTO_TRAIN_THRESHOLD:
        metadata["auto_train_threshold"] = AUTO_TRAIN_THRESHOLD
    else:
        metadata.setdefault("auto_train_threshold", AUTO_TRAIN_THRESHOLD)

    metadata["updated_at"] = current_timestamp()
    save_json_file(paths["metadata_path"], metadata)

    return project_id, metadata, paths


def load_project(project_id):
    """按 project_id 加载项目。"""
    paths = get_project_paths(project_id)
    _ensure_project_layout(paths)
    metadata = load_json_file(paths["metadata_path"], _default_project_metadata(project_id, paths["project_root"], DEFAULT_CLASS_NAMES))
    image_records = load_json_file(paths["image_records_path"], {})
    return metadata, image_records, paths


def save_project_metadata(project_id, metadata):
    """保存项目元数据。"""
    paths = get_project_paths(project_id)
    metadata["updated_at"] = current_timestamp()
    save_json_file(paths["metadata_path"], metadata)


def load_image_records(project_id):
    """加载项目的图片记录。"""
    paths = get_project_paths(project_id)
    records = load_json_file(paths["image_records_path"], {})
    if not isinstance(records, dict):
        records = {}
    return records


def save_image_records(project_id, image_records):
    """保存项目图片记录。"""
    paths = get_project_paths(project_id)
    save_json_file(paths["image_records_path"], image_records)


def refresh_project_counters(project_id):
    """根据图片记录重算 completed / dirty 计数。"""
    metadata, image_records, _ = load_project(project_id)
    completed_count = 0
    dirty_completed_count = 0

    for record in image_records.values():
        if record.get("annotation_completed"):
            completed_count += 1
            if record.get("dirty_since_last_train"):
                dirty_completed_count += 1

    metadata["completed_image_count"] = completed_count
    metadata["dirty_completed_image_count"] = dirty_completed_count
    metadata["bootstrap_done"] = bool(metadata.get("active_model_version"))
    save_project_metadata(project_id, metadata)
    return metadata


def update_image_record(project_id, image_path, annotation_file, image_state, annotation_hash):
    """更新某张图片在项目里的持久化记录。"""
    image_path = os.path.abspath(image_path)
    metadata, image_records, _ = load_project(project_id)
    old_record = image_records.get(image_path, {})
    completed = bool(image_state.get("annotation_completed", False))
    last_trained_seen_hash = old_record.get("last_trained_seen_hash")
    dirty_since_last_train = completed and annotation_hash != last_trained_seen_hash

    image_records[image_path] = {
        "image_path": image_path,
        "image_name": os.path.basename(image_path),
        "annotation_file": annotation_file,
        "annotation_completed": completed,
        "last_modified_at": image_state.get("last_modified_at") or current_timestamp(),
        "annotation_hash": annotation_hash,
        "last_trained_seen_hash": last_trained_seen_hash,
        "dirty_since_last_train": dirty_since_last_train,
        "project_id": project_id,
    }
    save_image_records(project_id, image_records)
    metadata = refresh_project_counters(project_id)
    return image_records[image_path], metadata


def mark_training_started(project_id, message):
    """更新项目训练状态。"""
    metadata = refresh_project_counters(project_id)
    metadata["last_training_status"] = "running"
    metadata["last_training_message"] = message or "训练中"
    save_project_metadata(project_id, metadata)
    return metadata


def mark_training_failed(project_id, message):
    """记录训练失败，但不切换 active 模型。"""
    metadata = refresh_project_counters(project_id)
    metadata["last_training_status"] = "failed"
    metadata["last_training_message"] = message or "训练失败"
    save_project_metadata(project_id, metadata)
    return metadata


def mark_training_success(project_id, version_name, snapshot_hashes):
    """训练成功后，仅将训练快照内未变化的 completed 图片标记为 clean。"""
    metadata, image_records, _ = load_project(project_id)
    snapshot_hashes = snapshot_hashes or {}

    for image_path, trained_hash in snapshot_hashes.items():
        record = image_records.get(image_path)
        if not record:
            continue
        if record.get("annotation_hash") != trained_hash:
            continue
        record["last_trained_seen_hash"] = trained_hash
        record["dirty_since_last_train"] = False

    save_image_records(project_id, image_records)
    metadata = refresh_project_counters(project_id)
    metadata["active_model_version"] = version_name
    metadata["last_successful_train_at"] = current_timestamp()
    metadata["bootstrap_done"] = True
    metadata["last_training_status"] = "idle"
    metadata["last_training_message"] = f"当前模型: {version_name}"
    save_project_metadata(project_id, metadata)
    return metadata


def update_project_versions(project_id, active_model_version, previous_model_version):
    """同步项目元数据中的 active / previous 版本号。"""
    metadata = refresh_project_counters(project_id)
    metadata["active_model_version"] = active_model_version
    metadata["previous_model_version"] = previous_model_version
    metadata["bootstrap_done"] = bool(active_model_version)
    save_project_metadata(project_id, metadata)
    return metadata


def get_completed_records(project_id):
    """获取项目下已完成图片记录。"""
    _, image_records, _ = load_project(project_id)
    completed = []
    for record in image_records.values():
        if record.get("annotation_completed"):
            completed.append(record)
    completed.sort(key=lambda item: item.get("image_name", ""))
    return completed


def get_dirty_completed_records(project_id):
    """获取项目下 dirty 的已完成图片记录。"""
    return [record for record in get_completed_records(project_id) if record.get("dirty_since_last_train")]