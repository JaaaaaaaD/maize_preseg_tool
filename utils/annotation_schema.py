# 标注数据结构辅助函数
import beifen
import copy
import hashlib
import json
from datetime import datetime

from utils.helpers import calculate_polygon_area, get_plant_color


def current_timestamp():
    """返回统一的时间戳字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clone_polygons(polygons):
    """深拷贝多边形坐标，统一转成 float tuple。"""
    cloned = []
    for polygon in polygons or []:
        points = []
        for point in polygon or []:
            if len(point) < 2:
                continue
            points.append((float(point[0]), float(point[1])))
        if points:
            cloned.append(points)
    return cloned


def normalize_polygon(points):
    """规范化单个多边形，去掉脏点并确保闭合。"""
    polygon = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        polygon.append((float(point[0]), float(point[1])))

    if len(polygon) < 3:
        return []

    # 连续重复点会让拖拽与导出更难处理，这里只保留顺序唯一点。
    deduped = []
    for point in polygon:
        if not deduped or deduped[-1] != point:
            deduped.append(point)

    if len(deduped) < 3:
        return []

    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])

    if calculate_polygon_area(deduped) <= 1:
        return []

    return deduped


def normalize_polygons(polygons):
    """规范化多 polygon 实例。"""
    normalized = []
    for polygon in polygons or []:
        valid_polygon = normalize_polygon(polygon)
        if valid_polygon:
            normalized.append(valid_polygon)
    return normalized


def calculate_total_polygon_area(polygons):
    """计算一个实例全部 polygon 的面积总和。"""
    total_area = 0.0
    for polygon in polygons or []:
        total_area += float(calculate_polygon_area(polygon))
    return float(total_area)





def next_instance_id(instances, current_hint=1):
    """计算下一个正式实例 id。"""
    max_id = int(current_hint or 1) - 1
    for instance in instances or []:
        try:
            max_id = max(max_id, int(instance.get("id", 0)))
        except (TypeError, ValueError):
            continue
    return max_id + 1


def make_image_state(image_path, annotation_completed=False):
    """创建图片级状态。"""
    return {
        "image_path": image_path,
        "annotation_completed": bool(annotation_completed),
        "last_modified_at": current_timestamp(),
        "last_trained_seen_hash": None,
        "dirty_since_last_train": bool(annotation_completed),
    }


def make_formal_instance(
    instance_id,
    polygons,
    source="manual",
    origin_model_version=None,
    origin_confidence=None,
    created_at=None,
    updated_at=None,
    color=None,
):
    """创建正式实例对象。"""
    polygons = normalize_polygons(polygons)
    now = current_timestamp()
    return {
        "id": int(instance_id),
        "polygons": polygons,
        "color": list(color or get_plant_color(int(instance_id))),
        "total_area": calculate_total_polygon_area(polygons),
        "source": source or "manual",
        "origin_model_version": origin_model_version,
        "origin_confidence": origin_confidence,
        "confirmed": True,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
    }


def make_candidate_instance(
    candidate_id,
    polygons,
    confidence=None,
    model_version=None,
):
    """创建候选实例对象。"""
    return {
        "candidate_id": str(candidate_id),
        "polygons": normalize_polygons(polygons),
        "confidence": None if confidence is None else float(confidence),
        "model_version": model_version,
        "selected": False,
    }


def normalize_formal_instance(instance, fallback_id):
    """兼容旧版 .maize 的正式实例结构。"""
    instance_id = int(instance.get("id", fallback_id))
    polygons = instance.get("polygons")
    if polygons is None and instance.get("points"):
        polygons = [instance.get("points")]
    polygons = normalize_polygons(polygons or [])
    created_at = instance.get("created_at") or current_timestamp()
    updated_at = instance.get("updated_at") or created_at

    normalized = {
        "id": instance_id,
        "polygons": polygons,
        "color": list(instance.get("color") or get_plant_color(instance_id)),
        "total_area": float(instance.get("total_area") or calculate_total_polygon_area(polygons)),
        "source": instance.get("source") or "manual",
        "origin_model_version": instance.get("origin_model_version"),
        "origin_confidence": instance.get("origin_confidence"),
        "confirmed": bool(instance.get("confirmed", True)),
        "created_at": created_at,
        "updated_at": updated_at,
        "labels": instance.get("labels", []),
        "preannotation_record_id": instance.get("preannotation_record_id"),
    }
    return normalized


def normalize_candidate_instance(candidate, fallback_index):
    """兼容候选层对象。"""
    return make_candidate_instance(
        candidate.get("candidate_id", f"cand_{fallback_index:04d}"),
        candidate.get("polygons", []),
        confidence=candidate.get("confidence"),
        model_version=candidate.get("model_version"),
    )


def normalize_image_state(image_path, image_state):
    """兼容图片级状态结构。"""
    state = copy.deepcopy(image_state or {})
    if not state:
        return make_image_state(image_path, annotation_completed=False)

    state.setdefault("image_path", image_path)
    state.setdefault("annotation_completed", False)
    state.setdefault("last_modified_at", current_timestamp())
    state.setdefault("last_trained_seen_hash", None)
    state.setdefault("dirty_since_last_train", bool(state.get("annotation_completed", False)))
    return state


def touch_instance(instance, source_override=None):
    """更新实例更新时间。"""
    instance["updated_at"] = current_timestamp()
    if source_override:
        instance["source"] = source_override
    instance["total_area"] = calculate_total_polygon_area(instance.get("polygons", []))
    return instance





def serialize_annotation_payload(instances, image_state):
    """序列化为稳定的 hash 输入。"""
    payload = {
        "instances": [],
        "annotation_completed": bool((image_state or {}).get("annotation_completed", False)),
    }

    for instance in sorted(instances or [], key=lambda item: int(item.get("id", 0))):
        payload["instances"].append(
            {
                "id": int(instance.get("id", 0)),
                "polygons": [
                    [[round(point[0], 2), round(point[1], 2)] for point in polygon]
                    for polygon in instance.get("polygons", [])
                ],
                "source": instance.get("source"),
                "origin_model_version": instance.get("origin_model_version"),
                "origin_confidence": instance.get("origin_confidence"),
            }
        )

    return payload


def compute_annotation_hash(instances, image_state):
    """计算当前正式标注的稳定 hash。"""
    payload = serialize_annotation_payload(instances, image_state)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
