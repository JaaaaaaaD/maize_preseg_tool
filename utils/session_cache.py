import hashlib
import json
import os
import tempfile
from datetime import datetime


SESSION_FILE_NAME = "session.json"
LAST_APP_SESSION_FILE_NAME = "last_app_session.json"


def _write_json_atomic(file_path, payload):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".json.tmp",
        dir=os.path.dirname(file_path),
        encoding="utf-8",
    )
    temp_path = temp_file.name
    try:
        json.dump(payload, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _read_json(file_path, default_value=None):
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default_value


def _image_key(image_path):
    normalized = os.path.abspath(image_path or "")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"draft_{digest}.json"


def get_autosave_root(project_root):
    return os.path.join(project_root, "autosave")


def save_session(autosave_root, image_paths, current_image_index, import_path=None):
    payload = {
        "image_paths": [os.path.abspath(path) for path in (image_paths or [])],
        "current_image_index": int(current_image_index or 0),
        "import_path": import_path or "",
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json_atomic(os.path.join(autosave_root, SESSION_FILE_NAME), payload)


def load_session(autosave_root):
    return _read_json(os.path.join(autosave_root, SESSION_FILE_NAME), default_value=None)


def save_image_draft(autosave_root, image_path, payload):
    if not image_path:
        return
    draft_file = os.path.join(autosave_root, _image_key(image_path))
    draft_payload = dict(payload or {})
    draft_payload["image_path"] = os.path.abspath(image_path)
    draft_payload["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_json_atomic(draft_file, draft_payload)


def load_image_draft(autosave_root, image_path):
    if not image_path:
        return None
    draft_file = os.path.join(autosave_root, _image_key(image_path))
    return _read_json(draft_file, default_value=None)


def _get_last_app_session_path(base_dir):
    return os.path.join(base_dir, LAST_APP_SESSION_FILE_NAME)


def save_last_app_session(base_dir, payload):
    if not base_dir:
        return
    _write_json_atomic(_get_last_app_session_path(base_dir), payload or {})


def load_last_app_session(base_dir):
    if not base_dir:
        return None
    return _read_json(_get_last_app_session_path(base_dir), default_value=None)
