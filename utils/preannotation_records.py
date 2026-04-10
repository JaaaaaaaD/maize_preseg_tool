import copy
import json
import os

from utils.annotation_schema import clone_polygons, current_timestamp

CORRECTION_RECORD_SCHEMA_VERSION = 6
DEFAULT_REGION_LABEL = "stem"
REASON_CODES = (
    "occluded_by_left_plant",
    "occluded_by_right_plant",
    "occluded_by_background",
    "neighbor_false_positive",
    "background_false_positive",
    "wrong_fragment",
    "ear_stem_segmentation",
    "non_target_leaf_in_instance",
    "occluded_by_this_instance",
)
REASON_CODE_LABELS = {
    "occluded_by_left_plant": "Occluded By Left Plant",
    "occluded_by_right_plant": "Occluded By Right Plant",
    "occluded_by_background": "Occluded By Background",
    "neighbor_false_positive": "Neighbor False Positive",
    "background_false_positive": "Background False Positive",
    "wrong_fragment": "Wrong Fragment",
    "ear_stem_segmentation": "Ear Stem Segmentation",
    "non_target_leaf_in_instance": "Non Target Leaf In Instance",
    "occluded_by_this_instance": "Occluded By This Instance",
}
VALID_STATUSES = {"accepted", "modified", "ignored", "rejected", "merged"}

GEOMETRY_REFINEMENT_EVENTS = {"add_vertex", "delete_vertex", "drag_vertex"}
TRIM_EVENTS = {"add_hole", "delete_staging_polygon", "delete_removal_region"}
SPLIT_EVENTS = {"split_staging_polygon"}
RELABEL_EVENTS = {"update_staging_label"}
POLYGON_MERGE_EVENTS = {"merge_staging_polygon"}
IGNORE_EVENTS = {"candidate_ignored", "instance_ignored"}
REJECT_EVENTS = {"candidate_rejected", "delete_instance"}
MERGED_STATUS_EVENTS = {"proposal_merged"}
MODIFICATION_EVENTS = (
    GEOMETRY_REFINEMENT_EVENTS
    | TRIM_EVENTS
    | SPLIT_EVENTS
    | RELABEL_EVENTS
    | POLYGON_MERGE_EVENTS
)


def _signed_polygon_area(polygon):
    points = list(polygon or [])
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return area / 2.0


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


def normalize_labels(labels, polygons):
    outer_polygon_count = sum(1 for polygon in polygons or [] if _signed_polygon_area(polygon) < 0)
    if outer_polygon_count <= 0 and polygons:
        outer_polygon_count = len(polygons)
    normalized = []
    for label in list(labels or [])[:outer_polygon_count]:
        value = str(label or "").strip() or DEFAULT_REGION_LABEL
        normalized.append(value)
    while len(normalized) < outer_polygon_count:
        normalized.append(DEFAULT_REGION_LABEL)
    return normalized


def make_annotation_state(polygons, labels=None):
    normalized_polygons = clone_polygons(polygons or [])
    return {
        "polygons": normalized_polygons,
        "labels": normalize_labels(labels, normalized_polygons),
    }


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
        events.append(normalized)
    return events


def normalize_reason_segments(segments):
    normalized_segments = []
    for index, segment in enumerate(segments or []):
        if not isinstance(segment, dict):
            continue
        start_source = segment.get("start_state") or {}
        end_source = segment.get("end_state") or {}
        start_state = make_annotation_state(
            segment.get("start_polygons", start_source.get("polygons")),
            segment.get("start_labels", start_source.get("labels")),
        )
        end_state = make_annotation_state(
            segment.get("end_polygons", end_source.get("polygons")),
            segment.get("end_labels", end_source.get("labels")),
        )
        event_log = normalize_event_log(segment.get("event_log") or segment.get("events") or [])
        if not start_state["polygons"] and end_state["polygons"]:
            start_state = make_annotation_state(end_state["polygons"], end_state["labels"])
        if not end_state["polygons"] and start_state["polygons"] and not event_log:
            end_state = make_annotation_state(start_state["polygons"], start_state["labels"])
        normalized_segments.append(
            {
                "segment_id": str(segment.get("segment_id") or f"seg_{index + 1:04d}"),
                "reason_code": normalize_reason_code(segment.get("reason_code")),
                "started_at": segment.get("started_at") or (event_log[0]["timestamp"] if event_log else current_timestamp()),
                "updated_at": segment.get("updated_at") or segment.get("ended_at") or current_timestamp(),
                "start_polygons": start_state["polygons"],
                "start_labels": start_state["labels"],
                "end_polygons": end_state["polygons"],
                "end_labels": end_state["labels"],
                "event_log": event_log,
            }
        )
    return normalized_segments


def infer_status_from_record(record):
    event_types = {entry.get("event_type") for entry in record.get("event_log", []) if entry.get("event_type")}
    final_polygons = record.get("final_polygons") or []

    if event_types & IGNORE_EVENTS:
        return "ignored"
    if event_types & MERGED_STATUS_EVENTS:
        return "merged"
    if event_types & REJECT_EVENTS and not final_polygons:
        return "rejected"
    if event_types & MODIFICATION_EVENTS:
        return "modified"
    return "accepted"


def normalize_status(status, record):
    value = str(status or "").strip().lower()
    if value in {"ignored", "rejected", "merged"}:
        return value
    inferred = infer_status_from_record(record)
    if value == "modified" and inferred == "accepted":
        return "modified"
    if value in VALID_STATUSES:
        return value if value in {"ignored", "rejected", "merged"} else inferred
    return inferred


def normalize_record(record):
    source = copy.deepcopy(record or {})
    created_at = source.get("created_at") or current_timestamp()
    original_state = make_annotation_state(
        source.get("original_polygons") or [],
        source.get("original_labels"),
    )
    final_state = make_annotation_state(
        source.get("final_polygons") or original_state["polygons"],
        source.get("final_labels"),
    )
    event_log = normalize_event_log(source.get("event_log") or source.get("operations") or [])
    reason_codes = normalize_reason_codes(source.get("reason_codes") or [])
    active_reason_code = normalize_reason_code(source.get("active_reason_code"))
    if active_reason_code and active_reason_code not in reason_codes:
        reason_codes.append(active_reason_code)

    reason_segments = normalize_reason_segments(source.get("reason_segments") or source.get("reason_chains") or [])
    active_reason_segment_index = source.get("active_reason_segment_index")
    try:
        active_reason_segment_index = int(active_reason_segment_index)
    except (TypeError, ValueError):
        active_reason_segment_index = None
    if active_reason_segment_index is not None and not (0 <= active_reason_segment_index < len(reason_segments)):
        active_reason_segment_index = None

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
        "original_polygons": original_state["polygons"],
        "original_labels": original_state["labels"],
        "final_polygons": final_state["polygons"],
        "final_labels": final_state["labels"],
        "formal_instance_id": source.get("formal_instance_id"),
        "reason_codes": reason_codes,
        "active_reason_code": active_reason_code,
        "event_log": event_log,
        "reason_segments": reason_segments,
        "active_reason_segment_index": active_reason_segment_index,
    }
    normalized["status"] = normalize_status(source.get("status"), normalized)
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


def _append_event_to_record(normalized_record, event_type, details=None, reason_code=None, timestamp=None):
    event = {
        "timestamp": timestamp or current_timestamp(),
        "event_type": str(event_type),
        "details": copy.deepcopy(details or {}),
    }
    normalized_reason = normalize_reason_code(reason_code) or normalized_record.get("active_reason_code")
    if normalized_reason:
        event["reason_code"] = normalized_reason
        if normalized_reason not in normalized_record["reason_codes"]:
            normalized_record["reason_codes"].append(normalized_reason)
    normalized_record["event_log"].append(event)
    normalized_record["updated_at"] = event["timestamp"]
    normalized_record["operations"] = copy.deepcopy(normalized_record["event_log"])
    return event


def append_event(record, event_type, details=None, reason_code=None):
    normalized_record = normalize_record(record)
    event = _append_event_to_record(normalized_record, event_type, details=details, reason_code=reason_code)
    normalized_record["status"] = infer_status_from_record(normalized_record)
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


def set_status(record, status):
    normalized_record = normalize_record(record)
    normalized_record["status"] = normalize_status(status, normalized_record)
    normalized_record["updated_at"] = current_timestamp()
    record.clear()
    record.update(normalized_record)
    return normalized_record["status"]


def set_annotation_state(record, prefix, polygons, labels=None):
    normalized_record = normalize_record(record)
    state = make_annotation_state(polygons, labels)
    normalized_record[f"{prefix}_polygons"] = state["polygons"]
    normalized_record[f"{prefix}_labels"] = state["labels"]
    normalized_record["updated_at"] = current_timestamp()
    if prefix == "final":
        normalized_record["status"] = infer_status_from_record(normalized_record)
    record.clear()
    record.update(normalized_record)
    return state


def close_active_reason_segment(record, end_polygons=None, end_labels=None):
    normalized_record = normalize_record(record)
    index = normalized_record.get("active_reason_segment_index")
    if index is None:
        record.clear()
        record.update(normalized_record)
        return None
    if 0 <= index < len(normalized_record["reason_segments"]):
        state = make_annotation_state(
            normalized_record.get("final_polygons") if end_polygons is None else end_polygons,
            normalized_record.get("final_labels") if end_labels is None else end_labels,
        )
        segment = normalized_record["reason_segments"][index]
        segment["end_polygons"] = state["polygons"]
        segment["end_labels"] = state["labels"]
        segment["updated_at"] = current_timestamp()
    normalized_record["active_reason_segment_index"] = None
    normalized_record["updated_at"] = current_timestamp()
    record.clear()
    record.update(normalized_record)
    return normalized_record


def sync_active_reason_segment(record, polygons=None, labels=None):
    normalized_record = normalize_record(record)
    index = normalized_record.get("active_reason_segment_index")
    if index is None or not (0 <= index < len(normalized_record["reason_segments"])):
        record.clear()
        record.update(normalized_record)
        return normalized_record
    state = make_annotation_state(
        normalized_record.get("final_polygons") if polygons is None else polygons,
        normalized_record.get("final_labels") if labels is None else labels,
    )
    segment = normalized_record["reason_segments"][index]
    segment["end_polygons"] = state["polygons"]
    segment["end_labels"] = state["labels"]
    segment["updated_at"] = current_timestamp()
    normalized_record["updated_at"] = current_timestamp()
    record.clear()
    record.update(normalized_record)
    return normalized_record


def append_reasoned_event(
    record,
    event_type,
    details=None,
    reason_code=None,
    before_polygons=None,
    before_labels=None,
    after_polygons=None,
    after_labels=None,
):
    normalized_record = normalize_record(record)
    event = _append_event_to_record(
        normalized_record,
        event_type,
        details=details,
        reason_code=reason_code,
    )

    before_state = make_annotation_state(
        normalized_record.get("final_polygons") if before_polygons is None else before_polygons,
        normalized_record.get("final_labels") if before_labels is None else before_labels,
    )
    after_state = make_annotation_state(
        normalized_record.get("final_polygons") if after_polygons is None else after_polygons,
        normalized_record.get("final_labels") if after_labels is None else after_labels,
    )
    segment_reason = normalize_reason_code(reason_code) or normalized_record.get("active_reason_code")
    segment_index = normalized_record.get("active_reason_segment_index")

    if segment_index is None or not (0 <= segment_index < len(normalized_record["reason_segments"])):
        segment_index = None
    elif normalized_record["reason_segments"][segment_index].get("reason_code") != segment_reason:
        active_segment = normalized_record["reason_segments"][segment_index]
        active_segment["end_polygons"] = before_state["polygons"]
        active_segment["end_labels"] = before_state["labels"]
        active_segment["updated_at"] = event["timestamp"]
        segment_index = None

    if segment_index is None:
        normalized_record["reason_segments"].append(
            {
                "segment_id": f"seg_{len(normalized_record['reason_segments']) + 1:04d}",
                "reason_code": segment_reason,
                "started_at": event["timestamp"],
                "updated_at": event["timestamp"],
                "start_polygons": before_state["polygons"],
                "start_labels": before_state["labels"],
                "end_polygons": before_state["polygons"],
                "end_labels": before_state["labels"],
                "event_log": [],
            }
        )
        segment_index = len(normalized_record["reason_segments"]) - 1
        normalized_record["active_reason_segment_index"] = segment_index

    segment = normalized_record["reason_segments"][segment_index]
    segment["event_log"].append(copy.deepcopy(event))
    segment["end_polygons"] = after_state["polygons"]
    segment["end_labels"] = after_state["labels"]
    segment["updated_at"] = event["timestamp"]

    normalized_record["final_polygons"] = after_state["polygons"]
    normalized_record["final_labels"] = after_state["labels"]
    normalized_record["status"] = infer_status_from_record(normalized_record)
    normalized_record["operations"] = copy.deepcopy(normalized_record["event_log"])
    record.clear()
    record.update(normalized_record)
    return event


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
