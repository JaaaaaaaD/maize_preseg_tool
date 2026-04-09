import copy
import json
import os

from utils.annotation_schema import clone_polygons, current_timestamp

CORRECTION_RECORD_SCHEMA_VERSION = 2
SEMANTIC_ACTION_AUTO = "__auto__"
SEMANTIC_ACTIONS = (
    "keep",
    "trim",
    "split",
    "merge",
    "relabel",
    "ignore",
    "reject",
)
REASON_CODES = (
    "weak_stem_support",
    "cross_neighbor_region",
    "isolated_fragment",
    "occlusion_too_heavy",
    "boundary_refined_only",
)
SEMANTIC_ACTION_LABELS = {
    SEMANTIC_ACTION_AUTO: "Auto",
    "keep": "Keep",
    "trim": "Trim",
    "split": "Split",
    "merge": "Merge",
    "relabel": "Relabel",
    "ignore": "Ignore",
    "reject": "Reject",
}
REASON_CODE_LABELS = {
    "weak_stem_support": "Weak Stem Support",
    "cross_neighbor_region": "Cross Neighbor Region",
    "isolated_fragment": "Isolated Fragment",
    "occlusion_too_heavy": "Occlusion Too Heavy",
    "boundary_refined_only": "Boundary Refined Only",
}

GEOMETRY_REFINEMENT_EVENTS = {"add_vertex", "delete_vertex", "drag_vertex"}
TRIM_EVENTS = {"add_hole", "delete_staging_polygon"}
SPLIT_EVENTS = {"split_staging_polygon"}
RELABEL_EVENTS = {"update_staging_label"}
IGNORE_EVENTS = {"candidate_ignored", "instance_ignored"}
REJECT_EVENTS = {"candidate_rejected", "delete_instance"}
MERGE_EVENTS = {"proposal_merged", "merge_staging_polygon"}


def normalize_semantic_action(action, allow_auto=False):
    if action is None:
        return SEMANTIC_ACTION_AUTO if allow_auto else None
    value = str(action).strip().lower()
    if not value:
        return SEMANTIC_ACTION_AUTO if allow_auto else None
    if value == SEMANTIC_ACTION_AUTO and allow_auto:
        return value
    if value in SEMANTIC_ACTIONS:
        return value
    return None


def normalize_reason_code(reason_code):
    if reason_code is None:
        return None
    value = str(reason_code).strip()
    if not value:
        return None
    return value


def normalize_reason_codes(reason_codes):
    normalized = []
    for reason_code in reason_codes or []:
        value = normalize_reason_code(reason_code)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def normalize_event_log(entries):
    events = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        event_type = entry.get("event_type") or entry.get("action")
        if not event_type:
            continue
        normalized = {
            "timestamp": entry.get("timestamp") or current_timestamp(),
            "event_type": str(event_type),
            "details": copy.deepcopy(entry.get("details") or {}),
        }
        reason_code = normalize_reason_code(entry.get("reason_code"))
        if reason_code:
            normalized["reason_code"] = reason_code
        semantic_action = normalize_semantic_action(entry.get("semantic_action"))
        if semantic_action:
            normalized["semantic_action"] = semantic_action
        events.append(normalized)
    return events


def infer_semantic_action_from_parts(status, event_log, final_polygons):
    normalized_status = str(status or "").strip().lower()
    event_types = {entry.get("event_type") for entry in event_log or [] if entry.get("event_type")}

    if normalized_status == "ignored":
        return "ignore"
    if normalized_status == "rejected":
        return "reject"
    if normalized_status == "merged":
        return "merge"

    if event_types & IGNORE_EVENTS:
        return "ignore"
    if event_types & MERGE_EVENTS:
        return "merge"
    if event_types & REJECT_EVENTS and not final_polygons:
        return "reject"
    if event_types & SPLIT_EVENTS:
        return "split"
    if event_types & TRIM_EVENTS:
        return "trim"
    if event_types & RELABEL_EVENTS and not (event_types & GEOMETRY_REFINEMENT_EVENTS):
        return "relabel"
    if event_types & GEOMETRY_REFINEMENT_EVENTS:
        return "keep"
    if event_types & RELABEL_EVENTS:
        return "relabel"
    return "keep"


def normalize_record(record):
    source = copy.deepcopy(record or {})
    created_at = source.get("created_at") or current_timestamp()
    original_polygons = clone_polygons(source.get("original_polygons") or [])
    final_polygons = clone_polygons(source.get("final_polygons") or [])
    event_log = normalize_event_log(source.get("event_log") or source.get("operations") or [])
    reason_codes = normalize_reason_codes(source.get("reason_codes") or [])
    active_reason_code = normalize_reason_code(source.get("active_reason_code"))
    if active_reason_code and active_reason_code not in reason_codes:
        reason_codes.append(active_reason_code)

    semantic_action = normalize_semantic_action(source.get("semantic_action"))
    semantic_action_source = str(source.get("semantic_action_source") or "").strip().lower()
    if semantic_action_source not in ("auto", "manual"):
        semantic_action_source = "manual" if semantic_action else "auto"
    if not semantic_action:
        semantic_action_source = "auto"
        semantic_action = infer_semantic_action_from_parts(
            source.get("status"),
            event_log,
            final_polygons,
        )
    elif semantic_action_source == "auto":
        semantic_action = infer_semantic_action_from_parts(
            source.get("status"),
            event_log,
            final_polygons,
        )
        semantic_action_source = "auto"

    normalized = {
        "record_id": str(source.get("record_id") or ""),
        "image_path": source.get("image_path"),
        "created_at": created_at,
        "updated_at": source.get("updated_at") or created_at,
        "model_path": source.get("model_path"),
        "model_type": source.get("model_type"),
        "roi_box": copy.deepcopy(source.get("roi_box") or []),
        "candidate_id": source.get("candidate_id"),
        "confidence": source.get("confidence"),
        "original_polygons": original_polygons,
        "final_polygons": final_polygons,
        "formal_instance_id": source.get("formal_instance_id"),
        "status": str(source.get("status") or "accepted"),
        "semantic_action": semantic_action,
        "semantic_action_source": semantic_action_source,
        "reason_codes": reason_codes,
        "active_reason_code": active_reason_code,
        "event_log": event_log,
    }
    if normalized["formal_instance_id"] is not None:
        try:
            normalized["formal_instance_id"] = int(normalized["formal_instance_id"])
        except (TypeError, ValueError):
            normalized["formal_instance_id"] = None
    return normalized


def serialize_record(record):
    payload = normalize_record(record)
    payload["operations"] = copy.deepcopy(payload["event_log"])
    return payload


def normalize_records(records):
    normalized = []
    for record in records or []:
        if isinstance(record, dict):
            normalized.append(normalize_record(record))
    return normalized


def append_event(record, event_type, details=None, reason_code=None, semantic_action=None):
    normalized_record = normalize_record(record)
    event = {
        "timestamp": current_timestamp(),
        "event_type": str(event_type),
        "details": copy.deepcopy(details or {}),
    }
    normalized_reason = normalize_reason_code(reason_code) or normalized_record.get("active_reason_code")
    if normalized_reason:
        event["reason_code"] = normalized_reason
        if normalized_reason not in normalized_record["reason_codes"]:
            normalized_record["reason_codes"].append(normalized_reason)
    normalized_action = normalize_semantic_action(semantic_action)
    if normalized_action:
        event["semantic_action"] = normalized_action
    normalized_record["event_log"].append(event)
    normalized_record["updated_at"] = event["timestamp"]
    if normalized_record.get("semantic_action_source") != "manual":
        normalized_record["semantic_action"] = infer_semantic_action_from_parts(
            normalized_record.get("status"),
            normalized_record.get("event_log"),
            normalized_record.get("final_polygons"),
        )
        normalized_record["semantic_action_source"] = "auto"
    normalized_record["operations"] = copy.deepcopy(normalized_record["event_log"])
    record.clear()
    record.update(normalized_record)
    return event


def set_active_reason(record, reason_code):
    normalized_record = normalize_record(record)
    normalized_reason = normalize_reason_code(reason_code)
    normalized_record["active_reason_code"] = normalized_reason
    if normalized_reason and normalized_reason not in normalized_record["reason_codes"]:
        normalized_record["reason_codes"].append(normalized_reason)
    normalized_record["updated_at"] = current_timestamp()
    record.clear()
    record.update(normalized_record)
    return normalized_reason


def set_semantic_action(record, semantic_action, source="manual"):
    normalized_record = normalize_record(record)
    normalized_action = normalize_semantic_action(semantic_action)
    if not normalized_action:
        normalized_record["semantic_action"] = infer_semantic_action_from_parts(
            normalized_record.get("status"),
            normalized_record.get("event_log"),
            normalized_record.get("final_polygons"),
        )
        normalized_record["semantic_action_source"] = "auto"
    else:
        normalized_record["semantic_action"] = normalized_action
        normalized_record["semantic_action_source"] = "manual" if source == "manual" else "auto"
    normalized_record["updated_at"] = current_timestamp()
    record.clear()
    record.update(normalized_record)
    return normalized_record["semantic_action"]


def set_status(record, status):
    normalized_record = normalize_record(record)
    normalized_record["status"] = str(status or normalized_record.get("status") or "accepted")
    normalized_record["updated_at"] = current_timestamp()
    if normalized_record.get("semantic_action_source") != "manual":
        normalized_record["semantic_action"] = infer_semantic_action_from_parts(
            normalized_record.get("status"),
            normalized_record.get("event_log"),
            normalized_record.get("final_polygons"),
        )
        normalized_record["semantic_action_source"] = "auto"
    record.clear()
    record.update(normalized_record)
    return normalized_record["status"]


def load_records_from_file(path, image_path=None):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return []

    raw_records = payload.get("records", []) if isinstance(payload, dict) else payload
    records = normalize_records(raw_records)
    if image_path:
        for record in records:
            if not record.get("image_path"):
                record["image_path"] = image_path
    return records


def save_records_to_file(path, image_path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    normalized_records = [serialize_record(record) for record in records or []]
    payload = {
        "schema_version": CORRECTION_RECORD_SCHEMA_VERSION,
        "image_path": image_path,
        "updated_at": current_timestamp(),
        "record_count": len(normalized_records),
        "records": normalized_records,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def next_record_counter(records, default_value=1):
    max_counter = int(default_value or 1) - 1
    for record in records or []:
        record_id = str(record.get("record_id") or "")
        if not record_id.startswith("pre_"):
            continue
        suffix = record_id.split("_", 1)[-1]
        try:
            max_counter = max(max_counter, int(suffix))
        except ValueError:
            continue
    return max_counter + 1
