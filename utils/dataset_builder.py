# 训练数据集构建工具
import hashlib
import os
import random
import shutil

from config import DEFAULT_CLASS_NAMES, FIXED_VAL_MIN_COUNT, FIXED_VAL_RATIO, FIXED_VAL_SEED
from utils.annotation_schema import current_timestamp
from utils.annotation_schema import normalize_formal_instance
from utils.data_manager import load_annotation_file
from utils.project_context import (
    get_completed_records,
    get_project_paths,
    load_json_file,
    load_project,
    save_json_file,
)


def _stable_dataset_stem(image_path):
    """为复制到数据集目录的文件生成稳定文件名。"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    digest = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()[:8]
    safe_base = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in base_name)
    return f"{safe_base}_{digest}"


def _reset_dataset_dirs(dataset_root):
    """重建数据集目录，确保删除旧标签残留。"""
    for relative_dir in ("images", "labels"):
        target_dir = os.path.join(dataset_root, relative_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(os.path.join(target_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "val"), exist_ok=True)


def _normalize_image_path(image_path):
    """统一图片路径格式，避免 Windows 正反斜杠导致 split 判断失效。"""
    if not image_path:
        return ""
    return os.path.normcase(os.path.normpath(os.path.abspath(str(image_path))))


def _resolve_val_manifest(project_id, completed_records, rebuild_split=False):
    """生成或复用固定验证集清单。"""
    paths = get_project_paths(project_id)
    metadata, _, _ = load_project(project_id)
    manifest = load_json_file(paths["split_manifest_path"], {})
    completed_paths = [os.path.abspath(record["image_path"]) for record in completed_records]
    completed_path_set = {_normalize_image_path(path) for path in completed_paths}

    if manifest and not rebuild_split:
        val_paths = {
            _normalize_image_path(path)
            for path in manifest.get("val_image_paths", [])
            if _normalize_image_path(path) in completed_path_set
        }
        manifest["val_image_paths"] = [
            path for path in manifest.get("val_image_paths", []) if _normalize_image_path(path) in val_paths
        ]
        return manifest, val_paths

    total_count = len(completed_paths)
    val_ratio = float(metadata.get("fixed_val_ratio", FIXED_VAL_RATIO))
    min_val = int(metadata.get("fixed_val_min_count", FIXED_VAL_MIN_COUNT))
    seed = int(metadata.get("fixed_val_seed", FIXED_VAL_SEED))

    if total_count <= 1:
        val_count = 0
    else:
        val_count = max(min_val, int(round(total_count * val_ratio)))
        val_count = min(val_count, total_count - 1)
        val_count = max(1, val_count)

    rng = random.Random(seed)
    shuffled = list(completed_paths)
    rng.shuffle(shuffled)
    val_paths = sorted(shuffled[:val_count])
    normalized_val_paths = {_normalize_image_path(path) for path in val_paths}

    manifest = {
        "project_id": project_id,
        "created_at": manifest.get("created_at") if manifest and not rebuild_split else current_timestamp(),
        "updated_at": current_timestamp(),
        "seed": seed,
        "val_ratio": val_ratio,
        "min_val_count": min_val,
        "val_image_paths": val_paths,
    }
    save_json_file(paths["split_manifest_path"], manifest)
    return manifest, normalized_val_paths


def _normalize_polygon_line(polygon, image_width, image_height):
    """将 polygon 转成 YOLO segmentation txt 一行的坐标串。"""
    coords = []
    for point in polygon:
        x_norm = max(0.0, min(1.0, float(point[0]) / float(image_width)))
        y_norm = max(0.0, min(1.0, float(point[1]) / float(image_height)))
        coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
    return coords


def _write_yolo_label(label_path, instances, image_width, image_height, class_names):
    """写出 YOLO segmentation 标签。

    Ultralytics 的 txt 标签以“单个 polygon 一行”为最稳妥写法。若一个正式实例包含多个
    polygon，这里会展开为多行、同 class_id 的记录，保证训练兼容性优先。
    """
    lines = []
    for raw_instance in instances or []:
        instance = normalize_formal_instance(raw_instance, class_names, raw_instance.get("id", 0))
        class_id = int(instance.get("class_id", 0))
        for polygon in instance.get("polygons", []):
            if len(polygon) < 3:
                continue
            coords = _normalize_polygon_line(polygon, image_width, image_height)
            if len(coords) >= 6:
                lines.append(" ".join([str(class_id)] + coords))

    with open(label_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def _write_data_yaml(data_yaml_path, dataset_root, class_names):
    """生成 Ultralytics data.yaml。"""
    image_root = os.path.join(dataset_root, "images")
    content = [
        f"path: {dataset_root}",
        f"train: {os.path.relpath(os.path.join(image_root, 'train'), dataset_root).replace(os.sep, '/')}",
        f"val: {os.path.relpath(os.path.join(image_root, 'val'), dataset_root).replace(os.sep, '/')}",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for index, name in enumerate(class_names):
        content.append(f"  {index}: {name}")

    with open(data_yaml_path, "w", encoding="utf-8") as file:
        file.write("\n".join(content) + "\n")


def build_project_dataset(project_id, rebuild_split=False):
    """从已完成正式标注构建项目训练集。"""
    completed_records = get_completed_records(project_id)
    paths = get_project_paths(project_id)
    dataset_root = paths["dataset_root"]
    os.makedirs(dataset_root, exist_ok=True)

    if not completed_records:
        raise RuntimeError("当前项目没有已完成图片，无法构建训练集")

    manifest, val_paths = _resolve_val_manifest(project_id, completed_records, rebuild_split=rebuild_split)
    _reset_dataset_dirs(dataset_root)

    data_yaml_path = os.path.join(dataset_root, "data.yaml")
    snapshot_hashes = {}
    export_index = []
    class_names = None
    train_count = 0
    val_count = 0

    for record in completed_records:
        annotation = load_annotation_file(record.get("annotation_file"))
        if not annotation:
            continue

        image_path = annotation["image_path"]
        if not os.path.exists(image_path):
            continue

        class_names = list(annotation.get("class_names") or DEFAULT_CLASS_NAMES)
        split = "val" if _normalize_image_path(image_path) in val_paths else "train"
        train_count += 1 if split == "train" else 0
        val_count += 1 if split == "val" else 0

        snapshot_hashes[image_path] = record.get("annotation_hash")
        stem = _stable_dataset_stem(image_path)
        image_ext = os.path.splitext(image_path)[1] or ".jpg"
        target_image_path = os.path.join(dataset_root, "images", split, f"{stem}{image_ext}")
        target_label_path = os.path.join(dataset_root, "labels", split, f"{stem}.txt")

        shutil.copy2(image_path, target_image_path)
        from PIL import Image as PILImage

        with PILImage.open(image_path) as image:
            width, height = image.size

        _write_yolo_label(target_label_path, annotation["plants"], width, height, class_names)

        export_index.append(
            {
                "image_path": image_path,
                "split": split,
                "copied_image_path": target_image_path,
                "label_path": target_label_path,
                "annotation_file": record.get("annotation_file"),
            }
        )

    class_names = list(class_names or DEFAULT_CLASS_NAMES)
    _write_data_yaml(data_yaml_path, dataset_root, class_names)

    return {
        "project_id": project_id,
        "dataset_root": dataset_root,
        "data_yaml_path": data_yaml_path,
        "completed_count": len(export_index),
        "train_count": train_count,
        "val_count": val_count,
        "snapshot_hashes": snapshot_hashes,
        "class_names": class_names,
        "split_manifest": manifest,
        "export_index": export_index,
    }
