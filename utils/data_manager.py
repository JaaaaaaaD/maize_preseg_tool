# 数据管理工具
import json
import os
import shutil
from datetime import datetime
import tempfile

from config import DEFAULT_CLASS_NAMES, VERSION
from utils.annotation_schema import (
    compute_annotation_hash,
    next_instance_id,
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


def _build_project_payload(
    image_path,
    plants,
    current_plant_id,
    image_state=None,
    project_id=None,
    class_names=None,
    ignored_regions=None,
):
    """构造统一的保存 payload。"""
    class_names = list(class_names or DEFAULT_CLASS_NAMES)
    normalized_plants = []
    for index, plant in enumerate(plants or [], start=1):
        normalized_plants.append(normalize_formal_instance(plant, index))

    normalized_state = normalize_image_state(image_path, image_state)

    # 提取图片名称（不包含路径）
    image_name = os.path.basename(image_path)
    
    return {
        "image_path": image_path,
        "image_name": image_name,
        "project_id": project_id,
        "class_names": class_names,
        "plants": normalized_plants,
        "current_plant_id": next_instance_id(normalized_plants, current_plant_id),
        "image_state": normalized_state,
        "ignored_regions": ignored_regions or [],
        "annotation_hash": compute_annotation_hash(normalized_plants, normalized_state),
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
    }


def save_annotation_manually(
    image_path,
    plants,
    image_width,
    image_height,
    export_path,
    class_names=None,
    ignored_regions=None,
    image_state=None,
    current_plant_id=None,
    project_id=None
):
    """手动保存标注为COCO格式。"""
    if not image_path or not export_path:
        return False, None, None

    try:
        # 构建COCO格式数据
        coco_data = _build_coco_format(
            image_path,
            plants,
            image_width,
            image_height,
            class_names=class_names,
            ignored_regions=ignored_regions,
            plant_groups=plant_groups,
            image_state=image_state,
            current_plant_id=current_plant_id,
            project_id=project_id
        )

        # 创建导出目录
        export_dir = os.path.dirname(export_path)
        os.makedirs(export_dir, exist_ok=True)

        # 备份旧文件
        _backup_old_file(export_path)

        # 原子写入：先写临时文件，再重命名
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir=export_dir)
        temp_path = temp_file.name
        try:
            json.dump(coco_data, temp_file, ensure_ascii=False, indent=2)
            temp_file.close()
            # 重命名临时文件为目标文件
            os.replace(temp_path, export_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

        # 构建返回的payload
        payload = _build_project_payload(
            image_path,
            plants,
            current_plant_id or 1,
            plant_groups=plant_groups,
            image_state=image_state,
            project_id=project_id,
            class_names=class_names,
            ignored_regions=ignored_regions,
        )

        return True, export_path, payload
    except Exception as error:
        print(f"保存标注失败: {error}")
        return False, None, None


def _backup_old_file(file_path):
    """备份旧文件，保留最近3个版本。"""
    if not os.path.exists(file_path):
        return

    try:
        # 备份目录
        backup_dir = os.path.join(os.path.dirname(file_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)

        # 生成备份文件名
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        ext = os.path.splitext(base_name)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{name_without_ext}_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_name)

        # 复制文件到备份目录
        shutil.copy2(file_path, backup_path)

        # 清理旧备份，只保留最近3个
        backup_files = sorted(
            [f for f in os.listdir(backup_dir) if f.startswith(name_without_ext) and f.endswith(ext)],
            key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)),
            reverse=True
        )

        for old_backup in backup_files[3:]:
            try:
                os.remove(os.path.join(backup_dir, old_backup))
            except:
                pass
    except Exception as e:
        print(f"备份文件失败: {e}")


def _build_coco_format(
    image_path,
    plants,
    image_width,
    image_height,
    class_names=None,
    ignored_regions=None,
    image_state=None,
    current_plant_id=None,
    project_id=None
):
    """构建标准COCO格式数据。"""
    class_names = list(class_names or DEFAULT_CLASS_NAMES)

    # 提取图片名称（不包含路径）
    image_name = os.path.basename(image_path)
    
    # 生成唯一的图片ID（基于图片路径的哈希值）
    image_id = abs(hash(image_path)) % 1000000
    
    # 构建COCO数据结构
    coco_data = {
        "info": {
            "description": "Maize Plant Multi-Class Instance Segmentation Dataset",
            "version": VERSION,
            "year": datetime.now().year,
            "contributor": "Maize Preseg Tool",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "custom": {
                "annotation_completed": image_state.get("annotation_completed", False) if image_state else False,
                "image_name": image_name,
                "tool_version": VERSION,
                "last_modified_at": image_state.get("last_modified_at") if image_state else datetime.now().isoformat(),
                "project_id": project_id,
                "current_plant_id": current_plant_id or 1
            }
        },
        "licenses": [],
        "images": [
            {
                "id": image_id,
                "file_name": image_name,
                "width": image_width,
                "height": image_height,
                "date_captured": "",
                "license": 0,
                "coco_url": "",
                "flickr_url": "",
                "attributes": {
                    "original_path": image_path,
                    "image_state": image_state or {}
                }
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
        "custom_extensions": {
            "class_names": class_names,
            "project_id": project_id
        }
    }

    annotation_id = 1
    
    # 添加植物实例标注
    for plant in plants or []:
        segmentation = []
        x_coords = []
        y_coords = []
        total_area = 0.0

        for polygon in plant.get("polygons", []):
            if len(polygon) < 3:
                continue
            # 确保多边形闭合
            if polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
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
                "image_id": image_id,
                "category_id": 1,  # 默认类别
                "segmentation": segmentation,
                "area": float(total_area),
                "bbox": [x_min, y_min, width, height],
                "iscrowd": 0,
                "attributes": {
                    "instance_id": plant.get("id", 0),
                    "source": plant.get("source"),
                    "owner_plant_id": plant.get("owner_plant_id"),
                },
            }
        )
        annotation_id += 1

    # 添加忽略区域（使用iscrowd=1的annotation格式）
    for region in ignored_regions or []:
        if len(region) < 3:
            continue
        # 确保多边形闭合
        if region[0] != region[-1]:
            region.append(region[0])
        
        segmentation = [[coord for point in region for coord in (point[0], point[1])]]
        x_coords = [point[0] for point in region]
        y_coords = [point[1] for point in region]
        total_area = float(calculate_polygon_area(region))

        x_min = min(x_coords)
        y_min = min(y_coords)
        width = max(x_coords) - x_min
        height = max(y_coords) - y_min

        coco_data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 使用第一个类别作为默认
                "segmentation": segmentation,
                "area": float(total_area),
                "bbox": [x_min, y_min, width, height],
                "iscrowd": 1,  # 标记为忽略区域
                "attributes": {
                    "type": "ignored_region"
                },
            }
        )
        annotation_id += 1

    return coco_data


def load_annotation_from_coco(coco_path, class_names=None):
    """从COCO文件加载标注。"""
    if not coco_path or not os.path.exists(coco_path):
        # 尝试从备份恢复
        backup_path = _try_restore_from_backup(coco_path)
        if backup_path:
            coco_path = backup_path
        else:
            return None

    try:
        with open(coco_path, "r", encoding="utf-8") as file:
            coco_data = json.load(file)
        
        # 验证COCO格式
        if not _validate_coco_format(coco_data):
            return None
        
        # 转换COCO格式为内部格式
        plants = []
        ignored_regions = []
        
        for ann in coco_data.get("annotations", []):
            if ann.get("iscrowd") == 1:
                # 处理忽略区域
                segmentation = ann.get("segmentation", [])
                for seg in segmentation:
                    if len(seg) >= 6:
                        polygon = []
                        for i in range(0, len(seg), 2):
                            polygon.append((seg[i], seg[i+1]))
                        if len(polygon) >= 3:
                            ignored_regions.append(polygon)
            else:
                # 处理正常实例
                segmentation = ann.get("segmentation", [])
                polygons = []
                
                for seg in segmentation:
                    if len(seg) >= 6:
                        polygon = []
                        for i in range(0, len(seg), 2):
                            polygon.append((seg[i], seg[i+1]))
                        if len(polygon) >= 3:
                            polygons.append(polygon)
                
                if not polygons:
                    continue
                
                # 创建植物实例
                plant = {
                    "id": ann.get("attributes", {}).get("instance_id", len(plants) + 1),
                    "polygons": polygons,
                    "bbox": ann.get("bbox", []),
                    "area": ann.get("area", 0),
                    "iscrowd": ann.get("iscrowd", 0),
                    "source": ann.get("attributes", {}).get("source"),
                    "owner_plant_id": ann.get("attributes", {}).get("owner_plant_id")
                }
                plants.append(plant)
        
        # 获取图片信息
        image_info = coco_data.get("images", [{}])[0]
        image_path = image_info.get("attributes", {}).get("original_path")
        
        # 获取自定义扩展信息
        class_names = coco_data.get("custom_extensions", {}).get("class_names", class_names)
        project_id = coco_data.get("custom_extensions", {}).get("project_id")
        
        # 获取图像状态
        image_state = image_info.get("attributes", {}).get("image_state", {})
        # 从info.custom中获取标注完成状态
        if "info" in coco_data and "custom" in coco_data["info"]:
            custom_info = coco_data["info"]["custom"]
            image_state["annotation_completed"] = custom_info.get("annotation_completed", False)
            image_state["last_modified_at"] = custom_info.get("last_modified_at", datetime.now().isoformat())
        
        # 构建内部格式payload
        internal_payload = {
            "image_path": image_path,
            "project_id": project_id,
            "class_names": class_names,
            "plants": plants,
            "current_plant_id": coco_data.get("info", {}).get("custom", {}).get("current_plant_id", 1),
            "image_state": image_state,
            "ignored_regions": ignored_regions,
        }
        
        return _normalize_loaded_payload(internal_payload, class_names=class_names)
    except Exception as error:
        print(f"加载COCO标注失败: {error}")
        return None


def _try_restore_from_backup(file_path):
    """尝试从备份恢复文件。"""
    if not file_path:
        return None
    
    backup_dir = os.path.join(os.path.dirname(file_path), "backups")
    if not os.path.exists(backup_dir):
        return None
    
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    
    # 查找备份文件
    backup_files = sorted(
        [f for f in os.listdir(backup_dir) if f.startswith(name_without_ext) and f.endswith(ext)],
        key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)),
        reverse=True
    )
    
    if backup_files:
        latest_backup = os.path.join(backup_dir, backup_files[0])
        try:
            # 复制备份文件到原位置
            shutil.copy2(latest_backup, file_path)
            print(f"从备份恢复文件: {latest_backup}")
            return file_path
        except:
            return None
    
    return None


def _validate_coco_format(coco_data):
    """验证COCO格式的有效性。"""
    try:
        # 检查必要字段
        if not isinstance(coco_data, dict):
            return False
        
        if "images" not in coco_data or not isinstance(coco_data["images"], list):
            return False
        
        if "annotations" not in coco_data or not isinstance(coco_data["annotations"], list):
            return False
        
        if "categories" not in coco_data or not isinstance(coco_data["categories"], list):
            return False
        
        # 检查images字段
        for image in coco_data["images"]:
            if not isinstance(image, dict):
                return False
            if "id" not in image or "file_name" not in image:
                return False
            if "width" not in image or "height" not in image:
                return False
        
        # 检查annotations字段
        for ann in coco_data["annotations"]:
            if not isinstance(ann, dict):
                return False
            if "id" not in ann or "image_id" not in ann:
                return False
            if "category_id" not in ann or "segmentation" not in ann:
                return False
        
        return True
    except:
        return False


def _normalize_loaded_payload(payload, image_path=None, class_names=None):
    """兼容旧数据结构并返回统一格式。"""
    class_names = list(payload.get("class_names") or class_names or DEFAULT_CLASS_NAMES)

    plants = []
    for index, plant in enumerate(payload.get("plants", []), start=1):
        plants.append(normalize_formal_instance(plant, index))

    image_state = normalize_image_state(image_path or payload.get("image_path"), payload.get("image_state"))
    annotation_hash = payload.get("annotation_hash") or compute_annotation_hash(plants, image_state)

    return {
        "image_path": image_path or payload.get("image_path"),
        "project_id": payload.get("project_id"),
        "class_names": class_names,
        "plants": plants,
        "current_plant_id": next_instance_id(plants, payload.get("current_plant_id", 1)),
        "image_state": image_state,
        "ignored_regions": payload.get("ignored_regions", []),
        "annotation_hash": annotation_hash,
        "version": VERSION,
    }


def load_annotation_file(file_path, class_names=None):
    """加载标注文件（兼容旧接口）。"""
    return load_annotation_from_coco(file_path, class_names=class_names)


def batch_export_annotations(
    export_dir,
    image_paths,
    coco_container=None,
    class_names=None,
    progress_callback=None
):
    """批量导出标注为COCO格式。"""
    if not image_paths or not export_dir:
        return 0, 0, 0

    try:
        os.makedirs(export_dir, exist_ok=True)
        exported_count = 0
        skipped_count = 0
        error_count = 0
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            if progress_callback and not progress_callback(i + 1, total, f"导出: {os.path.basename(image_path)}"):
                break

            try:
                # 优先从coco_container获取标注数据
                if coco_container and image_path in coco_container:
                    annotation = coco_container[image_path]
                else:
                    # 尝试从文件加载标注数据
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    coco_path = os.path.join(os.path.dirname(image_path), f"{base_name}_coco.json")
                    annotation = load_annotation_from_coco(coco_path)
                
                if not annotation:
                    error_count += 1
                    continue
                
                # 检查是否已标注
                if not annotation.get("image_state", {}).get("annotation_completed", False):
                    skipped_count += 1
                    continue
                
                # 构建导出路径，使用原图片名+anno
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                export_path = os.path.join(export_dir, f"{base_name}_anno.json")
                
                # 获取图片实际宽度和高度
                from PIL import Image
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except:
                    width, height = 0, 0
                
                # 保存标注
                success, _, _ = save_annotation_manually(
                    image_path,
                    annotation.get("plants", []),
                    width,
                    height,
                    export_path,
                    class_names=class_names or annotation.get("class_names"),
                    ignored_regions=annotation.get("ignored_regions"),
                    plant_groups=annotation.get("plant_groups"),
                    image_state=annotation.get("image_state"),
                    current_plant_id=annotation.get("current_plant_id")
                )
                
                if success:
                    exported_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"导出 {image_path} 失败: {e}")
                error_count += 1

        return exported_count, skipped_count, error_count
    except Exception as error:
        print(f"批量导出失败: {error}")
        return 0, 0, len(image_paths)


def batch_import_annotations(
    import_dir,
    image_paths,
    project_id,
    conflict_strategy="skip",
    progress_callback=None
):
    """批量导入COCO格式标注。"""
    if not import_dir or not image_paths:
        return {"imported": 0, "skipped": 0, "errors": 0}

    try:
        imported_count = 0
        skipped_count = 0
        error_count = 0
        total = len(image_paths)

        # 收集所有COCO文件
        coco_files = [f for f in os.listdir(import_dir) if f.endswith(".json")]
        
        # 创建临时COCO容器
        temp_coco_container = {}

        for i, image_path in enumerate(image_paths):
            if progress_callback and not progress_callback(i + 1, total, f"导入: {os.path.basename(image_path)}"):
                break

            try:
                # 查找对应的COCO文件
                image_name = os.path.basename(image_path)
                base_name = os.path.splitext(image_name)[0]
                coco_file = None

                for f in coco_files:
                    if (f.startswith(base_name) and f.endswith("_coco.json")) or f.startswith(base_name) and f.endswith("_anno.json"):
                        coco_file = os.path.join(import_dir, f)
                        break

                if not coco_file:
                    skipped_count += 1
                    continue

                # 加载COCO文件
                annotation = load_annotation_from_coco(coco_file)
                if not annotation:
                    error_count += 1
                    continue

                # 将标注数据存储到临时COCO容器
                temp_coco_container[image_path] = annotation
                imported_count += 1

            except Exception as e:
                print(f"导入 {image_path} 失败: {e}")
                error_count += 1

        return {"imported": imported_count, "skipped": skipped_count, "errors": error_count, "temp_coco_container": temp_coco_container}
    except Exception as error:
        print(f"批量导入失败: {error}")
        return {"imported": 0, "skipped": 0, "errors": len(image_paths), "temp_coco_container": {}}