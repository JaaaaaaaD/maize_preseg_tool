"""独立的项目训练入口。

这个脚本由 GUI 通过 QProcess 启动，不在主线程中直接训练。
stdout 使用约定格式输出，便于界面解析状态与进度：

- STATUS|stage|message
- PROGRESS|current_epoch|total_epoch|message
- RESULT|success|version_name|best_path|metrics_summary_path
- RESULT|failure|message
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def emit_status(stage, message):
    print(f"STATUS|{stage}|{message}", flush=True)


def emit_progress(current_epoch, total_epoch, message):
    print(f"PROGRESS|{current_epoch}|{total_epoch}|{message}", flush=True)


def emit_result_success(version_name, best_path, metrics_summary_path):
    print(f"RESULT|success|{version_name}|{best_path}|{metrics_summary_path}", flush=True)


def emit_result_failure(message):
    print(f"RESULT|failure|{message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model for one project")
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--data_yaml", required=True)
    parser.add_argument("--init_weights", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def detect_nvidia_gpu():
    """通过 nvidia-smi 粗略检测本机是否有 NVIDIA GPU。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
            check=False,
        )
        return result.returncode == 0 and bool((result.stdout or "").strip())
    except Exception:
        return False


def inspect_torch_runtime():
    """读取当前训练运行时的 torch / CUDA 状态，便于在 GUI 中明确展示。"""
    runtime = {
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "nvidia_gpu_detected": detect_nvidia_gpu(),
    }
    try:
        import torch

        runtime["torch_version"] = getattr(torch, "__version__", None)
        runtime["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        runtime["cuda_available"] = bool(torch.cuda.is_available())
        runtime["cuda_device_count"] = int(torch.cuda.device_count())
        for index in range(runtime["cuda_device_count"]):
            try:
                runtime["cuda_device_names"].append(torch.cuda.get_device_name(index))
            except Exception:
                runtime["cuda_device_names"].append(f"cuda:{index}")
    except Exception:
        pass
    return runtime


def resolve_device(device_arg, runtime):
    """将 auto 解析成可用设备；若显式请求 GPU 但不可用，则直接失败。"""
    if device_arg == "auto":
        return (0 if runtime["cuda_available"] else "cpu"), None

    normalized = str(device_arg).strip().lower()
    gpu_requested = normalized in {"cuda", "cuda:0", "0"} or normalized.startswith("cuda:")
    if gpu_requested and not runtime["cuda_available"]:
        torch_version = runtime.get("torch_version") or "unknown"
        torch_cuda_version = runtime.get("torch_cuda_version") or "None"
        if runtime.get("nvidia_gpu_detected"):
            return None, (
                "检测到 NVIDIA 显卡，但当前 PyTorch 不是可用的 CUDA 运行时。"
                f" torch={torch_version}, torch.version.cuda={torch_cuda_version}。"
                " 请安装 CUDA 版 torch 后再训练。"
            )
        return None, (
            "当前请求使用 GPU，但 PyTorch 未检测到可用 CUDA 设备。"
            f" torch={torch_version}, torch.version.cuda={torch_cuda_version}。"
        )
    return device_arg, None


def install_callbacks(model, total_epochs):
    """注册训练进度回调。"""
    try:
        model.add_callback("on_train_start", lambda trainer: emit_status("train_start", "开始训练"))
        model.add_callback(
            "on_train_epoch_end",
            lambda trainer: emit_progress(
                int(getattr(trainer, "epoch", 0)) + 1,
                total_epochs,
                f"Epoch {int(getattr(trainer, 'epoch', 0)) + 1}/{total_epochs}",
            ),
        )
        model.add_callback("on_train_end", lambda trainer: emit_status("train_end", "训练结束，正在整理结果"))
    except Exception:
        emit_status("callback_warning", "当前 ultralytics 版本未启用细粒度回调，进度将按阶段显示")


def collect_metrics(train_result, version_name, init_weights):
    """提取训练结果摘要。"""
    metrics_summary = {
        "version_name": version_name,
        "init_weights": init_weights,
        "results_dict": {},
    }

    results_dict = getattr(train_result, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics_summary["results_dict"] = results_dict
    else:
        for attr_name in ("box", "seg"):
            value = getattr(train_result, attr_name, None)
            if value is not None:
                metrics_summary[attr_name] = str(value)
    return metrics_summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    emit_status("bootstrap", "正在导入训练依赖")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        emit_result_failure("未安装 ultralytics，无法训练 YOLO11 segmentation 模型")
        raise SystemExit(1) from error

    runtime = inspect_torch_runtime()
    emit_status(
        "runtime",
        "Torch 运行时: "
        f"torch={runtime.get('torch_version') or 'unknown'}, "
        f"torch.version.cuda={runtime.get('torch_cuda_version') or 'None'}, "
        f"cuda_available={runtime.get('cuda_available')}, "
        f"device_count={runtime.get('cuda_device_count', 0)}",
    )
    if runtime.get("cuda_device_names"):
        emit_status("runtime", f"CUDA 设备: {', '.join(runtime['cuda_device_names'])}")
    elif runtime.get("nvidia_gpu_detected"):
        emit_status("runtime", "检测到 NVIDIA 显卡，但当前 Python 环境中的 torch 无法访问 CUDA")

    device, device_error = resolve_device(args.device, runtime)
    if device_error:
        emit_result_failure(device_error)
        raise SystemExit(1)
    emit_status("prepare", f"数据集: {args.data_yaml}")
    emit_status("prepare", f"初始化权重: {args.init_weights}")
    emit_status("prepare", f"设备: {device}")

    model = YOLO(args.init_weights)
    install_callbacks(model, args.epochs)

    try:
        train_result = model.train(
            data=args.data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=os.path.dirname(args.output_dir),
            name=os.path.basename(args.output_dir),
            exist_ok=True,
            task="segment",
            verbose=True,
        )
    except Exception as error:
        emit_result_failure(f"训练异常: {error}")
        raise SystemExit(1) from error

    emit_status("collect", "整理训练产物")
    weights_dir = os.path.join(args.output_dir, "weights")
    best_source = os.path.join(weights_dir, "best.pt")
    last_source = os.path.join(weights_dir, "last.pt")
    best_target = os.path.join(args.output_dir, "best.pt")
    last_target = os.path.join(args.output_dir, "last.pt")

    if os.path.exists(best_source):
        shutil.copy2(best_source, best_target)
    if os.path.exists(last_source):
        shutil.copy2(last_source, last_target)

    metrics_summary = collect_metrics(train_result, args.run_name, args.init_weights)
    metrics_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, ensure_ascii=False, indent=2)

    if not os.path.exists(best_target):
        emit_result_failure("训练结束但未找到 best.pt")
        raise SystemExit(1)

    emit_result_success(args.run_name, best_target, metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
