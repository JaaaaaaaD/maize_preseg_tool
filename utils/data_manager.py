# 数据管理工具
import json
import os
import shutil
from datetime import datetime

from config import ANNOTATION_DIR, DEFAULT_CLASS_NAMES, VERSION
from utils.annotation_schema import (
    compute_annotation_hash,
    ensure_plant_groups,
    next_instance_id,
    next_plant_group_id,
    normalize_formal_instance,
    normalize_image_state,
)
from utils.helpers import calculate_polygon_area
from utils.project_context import get_completed_records


def _safe_file_stem(raw_name):
    """将文件 stem 转成适合 Windows 的安全文件名。"""
    raw_name = raw_name or "annotation"
    invalid_chars = '<>:"/\\|?*'
    safe = []
    for char in raw_name:
        if char in invalid_chars:
            safe.append("_")
        else:
            safe.append(char)
    sanitized = "".join(safe).strip()
    return sanitized or "annotation"


def get_auto_save_path(image_path):
    """获取自动保存路径。"""
    if not image_path:
        return None
    image_name = _safe_file_stem(os.path.splitext(os.path.basename(image_path))[0])
    path_hash = abs(hash(os.path.abspath(image_path))) % 10000
    return os.path.join(ANNOTATION_DIR, f"{image_name}_{path_hash}.maize")


def _build_project_payload(
    image_path,
    plants,
    current_plant_id,
    plant_groups=None,
    image_state=None,
    project_id=None,
    class_names=None,
    ignored_regions=None,
):
    """构造统一的保存 payload。"""
    class_names = list(class_names or DEFAULT_CLASS_NAMES)
    normalized_plants = []
    for index, plant in enumerate(plants or [], start=1):
        normalized_plants.append(normalize_formal_instance(plant, class_names, index))

    normalized_state = normalize_image_state(image_path, image_state)
    normalized_groups = ensure_plant_groups(normalized_plants, plant_groups or [])

    return {
        "image_path": image_path,
        "project_id": project_id,
        "class_names": class_names,
        "plants": normalized_plants,
        "plant_groups": normalized_groups,
        "current_plant_id": next_instance_id(normalized_plants, current_plant_id),
        "next_plant_group_id": next_plant_group_id(normalized_groups),
        "image_state": normalized_state,
        "ignored_regions": ignored_regions or [],
        "annotation_hash": compute_annotation_hash(normalized_plants, normalized_groups, normalized_state),
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "is_auto_save": True,
    }


def save_current_annotation(
    image_path,
    plants,
    current_plant_id,
    plant_groups=None,
    image_state=None,
    project_id=None,
    class_names=None,
    ignored_regions=None,
):
    """保存当前标注。"""
    save_path = get_auto_save_path(image_path)
    if not save_path:
        return False, None, None

    try:
        os.makedirs(ANNOTATION_DIR, exist_ok=True)
        payload = _build_project_payload(
            image_path,
            plants,
            current_plant_id,
            plant_groups=plant_groups,
            image_state=image_state,
            project_id=project_id,
            class_names=class_names,
            ignored_regions=ignored_regions,
        )
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return True, save_path, payload
    except Exception as error:
        print(f"保存标注失败: {error}")
        return False, None, None


def _normalize_loaded_payload(payload, image_path=None, class_names=None):
    """兼容旧数据结构并返回统一格式。"""
    class_names = list(payload.get("class_names") or class_names or DEFAULT_CLASS_NAMES)

    plants = []
    for index, plant in enumerate(payload.get("plants", []), start=1):
        plants.append(normalize_formal_instance(plant, class_names, index))

    plant_groups = ensure_plant_groups(plants, payload.get("plant_groups", []))
    image_state = normalize_image_state(image_path or payload.get("image_path"), payload.get("image_state"))
    annotation_hash = payload.get("annotation_hash") or compute_annotation_hash(plants, plant_groups, image_state)

    return {
        "image_path": image_path or payload.get("image_path"),
        "project_id": payload.get("project_id"),
        "class_names": class_names,
        "plants": plants,
        "plant_groups": plant_groups,
        "current_plant_id": next_instance_id(plants, payload.get("current_plant_id", 1)),
        "next_plant_group_id": next_plant_group_id(plant_groups),
        "image_state": image_state,
        "annotation_hash": annotation_hash,
        "version": payload.get("version", VERSION),
    }


def load_annotation_file(save_path, class_names=None):
    """按 .maize 路径加载标注。"""
    if not save_path or not os.path.exists(save_path):
        return None

    try:
        with open(save_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return _normalize_loaded_payload(payload, class_names=class_names)
    except Exception as error:
        print(f"加载标注失败: {error}")
        return None


def load_current_annotation(image_path, class_names=None):
    """加载当前标注。"""
    load_path = get_auto_save_path(image_path)
    if not load_path or not os.path.exists(load_path):
        return None
    annotation = load_annotation_file(load_path, class_names=class_names)
    if annotation:
        annotation["image_path"] = image_path
    return annotation


def export_simple_json(image_path, plants, plant_groups=None, image_state=None, export_path=None, class_names=None, ignored_regions=None):
    """导出为扩展简单 JSON 格式。"""
    if not image_path:
        return None

    try:
        payload = _build_project_payload(
            image_path,
            plants,
            current_plant_id=1,
            plant_groups=plant_groups,
            image_state=image_state,
            class_names=class_names,
            ignored_regions=ignored_regions,
        )
        
        if not export_path:
            base_name = _safe_file_stem(os.path.splitext(os.path.basename(image_path))[0])
            export_dir = ANNOTATION_DIR
            export_path = os.path.join(export_dir, f"{base_name}_annotation.json")
        else:
            export_dir = os.path.dirname(export_path)
        
        os.makedirs(export_dir, exist_ok=True)

        with open(export_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return export_path
    except Exception as error:
        print(f"导出 JSON 失败: {error}")
        return None


def export_coco_format(
    image_path,
    plants,
    image_width,
    image_height,
    export_path=None,
    class_names=None,
    ignored_regions=None,
):
    """导出为 COCO 格式。

    这里按“正式实例”为 annotation 单位；若一个实例包含多个 polygon，则写成一个
    segmentation 数组列表，保持与正式层的实例概念一致。
    """
    if not image_path:
        return None

    class_names = list(class_names or DEFAULT_CLASS_NAMES)

    try:
        coco_data = {
            "info": {
                "description": "Maize Plant Multi-Class Instance Segmentation Dataset",
                "version": VERSION,
                "year": datetime.now().year,
                "contributor": "Maize Preseg Tool",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "licenses": [],
            "images": [
                {
                    "id": 1,
                    "file_name": os.path.basename(image_path),
                    "width": image_width,
                    "height": image_height,
                    "date_captured": "",
                    "license": 0,
                    "coco_url": "",
                    "flickr_url": "",
                }
            ],
            "annotations": [],
            "categories": [
                {
                    "id": class_id + 1,
                    "name": class_name,
                    "supercategory": "maize_part",
                }
                for class_id, class_name in enumerate(class_names)
            ],
            "ignored_regions": [],
        }

        annotation_id = 1
        for plant in plants or []:
            segmentation = []
            x_coords = []
            y_coords = []
            total_area = 0.0

            for polygon in plant.get("polygons", []):
                if len(polygon) < 3:
                    continue
                segmentation.append([coord for point in polygon for coord in (point[0], point[1])])
                x_coords.extend([point[0] for point in polygon])
                y_coords.extend([point[1] for point in polygon])
                total_area += float(calculate_polygon_area(polygon))

            if not segmentation:
                continue

            x_min = min(x_coords)
            y_min = min(y_coords)
            width = max(x_coords) - x_min
            height = max(y_coords) - y_min

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": int(plant.get("class_id", 0)) + 1,
                    "segmentation": segmentation,
                    "area": float(total_area),
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0,
                    "attributes": {
                        "instance_id": plant.get("id", 0),
                        "class_name": plant.get("class_name"),
                        "source": plant.get("source"),
                        "owner_plant_id": plant.get("owner_plant_id"),
                    },
                }
            )
            annotation_id += 1

        # 添加忽略区域
        for region in ignored_regions or []:
            segmentation = []
            x_coords = []
            y_coords = []
            total_area = 0.0

            if len(region) >= 3:
                segmentation.append([coord for point in region for coord in (point[0], point[1])])
                x_coords.extend([point[0] for point in region])
                y_coords.extend([point[1] for point in region])
                total_area += float(calculate_polygon_area(region))

                if segmentation:
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    width = max(x_coords) - x_min
                    height = max(y_coords) - y_min

                    coco_data["ignored_regions"].append({
                        "segmentation": segmentation,
                        "area": float(total_area),
                        "bbox": [x_min, y_min, width, height]
                    })

        if not export_path:
            base_name = _safe_file_stem(os.path.splitext(os.path.basename(image_path))[0])
            export_dir = ANNOTATION_DIR
            export_path = os.path.join(export_dir, f"{base_name}_coco.json")
        else:
            export_dir = os.path.dirname(export_path)
        
        os.makedirs(export_dir, exist_ok=True)

        with open(export_path, "w", encoding="utf-8") as file:
            json.dump(coco_data, file, ensure_ascii=False, indent=2)

        return export_path
    except Exception as error:
        print(f"导出 COCO 格式失败: {error}")
        return None


def export_completed_annotations(project_id, export_dir=None):
    """批量导出项目下所有已完成图片的简单 JSON。"""
    completed_records = get_completed_records(project_id)
    if not completed_records:
        return None, 0

    try:
        export_dir = export_dir or os.path.join(ANNOTATION_DIR, f"completed_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(export_dir, exist_ok=True)
        exported_count = 0

        for record in completed_records:
            annotation = load_annotation_file(record.get("annotation_file"))
            if not annotation:
                continue
            json_path = export_simple_json(
                annotation["image_path"],
                annotation["plants"],
                plant_groups=annotation.get("plant_groups"),
                image_state=annotation.get("image_state"),
                export_dir=export_dir,
                class_names=annotation.get("class_names"),
            )
            if json_path:
                exported_count += 1

        return export_dir, exported_count
    except Exception as error:
        print(f"批量导出失败: {error}")
        return None, 0


def copy_completed_annotations_to_dir(project_id, export_dir):
    """将已完成图片的 .maize 与 JSON 一并导出。"""
    completed_records = get_completed_records(project_id)
    if not completed_records:
        return None, 0

    try:
        os.makedirs(export_dir, exist_ok=True)
        exported_count = 0
        for record in completed_records:
            annotation_file = record.get("annotation_file")
            if annotation_file and os.path.exists(annotation_file):
                shutil.copy2(annotation_file, os.path.join(export_dir, os.path.basename(annotation_file)))
            annotation = load_annotation_file(annotation_file)
            if not annotation:
                continue
            json_path = export_simple_json(
                annotation["image_path"],
                annotation["plants"],
                plant_groups=annotation.get("plant_groups"),
                image_state=annotation.get("image_state"),
                export_dir=export_dir,
                class_names=annotation.get("class_names"),
            )
            if json_path:
                exported_count += 1
        return export_dir, exported_count
    except Exception as error:
        print(f"导出项目标注失败: {error}")
        return None, 0