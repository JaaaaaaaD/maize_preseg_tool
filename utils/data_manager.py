# 数据管理工具

import json
import os
from datetime import datetime
from config import ANNOTATION_DIR, VERSION

def get_auto_save_path(image_path):
    """获取自动保存路径"""
    if not image_path:
        return None
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    path_hash = abs(hash(image_path)) % 10000
    return os.path.join(ANNOTATION_DIR, f"{image_name}_{path_hash}.maize")

def save_current_annotation(image_path, plants, current_plant_id):
    """保存当前标注"""
    save_path = get_auto_save_path(image_path)
    if not save_path:
        return False

    try:
        # 确保保存目录存在
        os.makedirs(ANNOTATION_DIR, exist_ok=True)
        
        project_data = {
            "image_path": image_path,
            "plants": plants,
            "current_plant_id": current_plant_id,
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": VERSION,
            "is_auto_save": True
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存标注失败: {e}")
        return False

def load_current_annotation(image_path):
    """加载当前标注"""
    load_path = get_auto_save_path(image_path)
    if not load_path or not os.path.exists(load_path):
        return None

    try:
        with open(load_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)

        # 兼容旧格式
        for plant in project_data.get("plants", []):
            if "points" in plant and "polygons" not in plant:
                plant["polygons"] = [plant["points"]]
                plant["total_area"] = plant.get("area", 0)

        return {
            "plants": project_data.get("plants", []),
            "current_plant_id": project_data.get("current_plant_id", 1)
        }
    except Exception as e:
        print(f"加载标注失败: {e}")
        return None

def export_simple_json(image_path, plants):
    """导出为简单JSON格式"""
    if not image_path or not plants:
        return None

    try:
        export_data = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "plants": plants,
            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": VERSION
        }

        # 生成导出路径
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        export_path = os.path.join(ANNOTATION_DIR, f"{base_name}_annotation.json")

        # 确保保存目录存在
        os.makedirs(ANNOTATION_DIR, exist_ok=True)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return export_path
    except Exception as e:
        print(f"导出JSON失败: {e}")
        return None

def export_coco_format(image_path, plants, image_width, image_height):
    """导出为COCO格式"""
    if not image_path or not plants:
        return None

    try:
        coco_data = {
            "info": {
                "description": "Maize Plant Annotations",
                "version": VERSION,
                "year": datetime.now().year,
                "contributor": "Maize Preseg Tool",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                    "flickr_url": ""
                }
            ],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "maize_plant",
                    "supercategory": "plant"
                }
            ]
        }

        annotation_id = 1
        for plant in plants:
            for polygon in plant.get("polygons", []):
                # 转换多边形格式
                segmentation = []
                for point in polygon:
                    segmentation.extend([point[0], point[1]])

                # 计算边界框
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x_min = min(x_coords)
                y_min = min(y_coords)
                width = max(x_coords) - x_min
                height = max(y_coords) - y_min

                annotation = {
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": plant.get("total_area", 0),
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0,
                    "attributes": {
                        "plant_id": plant.get("id", 0)
                    }
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

        # 生成导出路径
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        export_path = os.path.join(ANNOTATION_DIR, f"{base_name}_coco.json")

        # 确保保存目录存在
        os.makedirs(ANNOTATION_DIR, exist_ok=True)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        return export_path
    except Exception as e:
        print(f"导出COCO格式失败: {e}")
        return None

def export_annotated_images(image_paths, annotation_status):
    """批量导出已标注的图片"""
    annotated_images = [path for path in image_paths if annotation_status.get(path, False)]
    if not annotated_images:
        return None, 0

    try:
        # 创建导出目录
        export_dir = os.path.join(ANNOTATION_DIR, f"annotated_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(export_dir, exist_ok=True)

        exported_count = 0
        for image_path in annotated_images:
            # 加载标注
            annotation_data = load_current_annotation(image_path)
            if not annotation_data or not annotation_data["plants"]:
                continue

            # 导出为JSON
            json_path = export_simple_json(image_path, annotation_data["plants"])
            if json_path:
                # 复制JSON文件到导出目录
                import shutil
                dest_json = os.path.join(export_dir, os.path.basename(json_path))
                shutil.copy2(json_path, dest_json)
                exported_count += 1

        return export_dir, exported_count
    except Exception as e:
        print(f"批量导出失败: {e}")
        return None, 0