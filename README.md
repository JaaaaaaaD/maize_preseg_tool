# 玉米标注与项目级预标注工具

这是一个基于 PyQt5 的桌面标注工具，面向玉米实例分割场景，当前主线已经升级为：

- 人工标注
- 图片级“已完成”管理
- 项目级自动训练
- 打开未标注图时自动预标注
- 候选层人工接受/修正
- 模型版本管理与回退

当前入口文件是 `main.py`。

## 核心特性

- 一个正式实例可包含多个 polygon
- 类别挂在暂存区域（多边形）上，每个区域可以有独立的标签（stem、leaf、ear）
- 实例本身不带有类别，使用 -1 作为 none 占位
- 候选层与正式层分离，AI 结果不会直接写入正式标注
- 训练在独立进程中执行，不阻塞 GUI
- 推理在后台线程执行，不阻塞切图
- 支持项目级 YOLO segmentation 数据集导出
- 支持 active 模型和 previous 模型回退
- 支持撤销/重做操作，包括撤销删除植株

## 当前主流程

1. 加载同一目录下的一批图片，程序自动进入或创建一个项目。
2. 在左侧主画布中手工标注，先选择区域标签（stem、leaf、ear），然后绘制多边形，`Enter` 暂存一个区域，`Shift+Enter` 保存一个正式实例。
3. 右侧属性面板可查看实例信息，包括 `source / origin_model_version / origin_confidence`。
4. 图片确认无误后，点击“标记当前图片为已完成”。
5. 当自上次成功训练以来 `dirty completed >= 5` 时，自动启动一轮后台训练。
6. 训练成功后，后续打开“无正式实例且未完成”的图片时会自动执行预标注。
7. AI 候选先进入候选层；你可以接受、删除、拖点修改、改类别。
8. 保存后的最终正式标注才会进入训练集和导出结果。

## 标注一个图片的详细流程

### 1. 加载图片
- 点击左侧工具栏中的“批量加载图片”按钮，选择包含图片的目录。
- 程序会自动创建一个项目，并加载目录中的所有图片。
- 使用左侧工具栏中的“上一张”和“下一张”按钮（或左右方向键）切换图片。

### 2. 开始标注
- **选择标签**：在左侧工具栏的“区域标签”下拉框中选择当前要标注的区域类型（stem、leaf、ear）。
- **绘制多边形**：在左侧画布上点击鼠标左键添加多边形的顶点，完成后会自动闭合多边形。
- **暂存区域**：按 `Enter` 键或点击“暂存当前区域”按钮，将当前绘制的多边形暂存为植株的一个部分。
- **添加去除区域**：点击“去除区域”按钮，然后在需要去除的区域绘制多边形，完成后按 `Enter` 键暂存。
- **保存整株**：当完成一个植株的所有区域标注后，按 `Shift+Enter` 键或点击“保存整株”按钮，将整个植株保存为正式实例。

### 3. 编辑和调整
- **撤销操作**：按 `Ctrl+Z` 键或点击“撤销”按钮，撤销上一步操作。
- **重做操作**：按 `Ctrl+Y` 键或点击“重做”按钮，重做上一步被撤销的操作。
- **删除植株**：在左侧“植株列表”中选择要删除的植株，然后点击“删除选中植株”按钮。
- **撤销删除**：点击“撤销删除植株”按钮，可以恢复最近删除的植株。

### 4. 查看和管理
- **右侧总览**：右侧画布显示当前图片的所有正式实例，每个实例的最左端会显示红色的 `plant ID`。
- **属性面板**：右侧属性面板显示当前选中实例的详细信息，包括类别、所属植株组等。
- **图片状态**：当图片标注完成后，点击“标记当前图片为已完成”按钮，将图片标记为已完成状态。
- **状态锁定**：当图片状态为已完成时，若要进行标注操作，会弹出对话框提示是否取消已完成状态。

### 5. 导入导出
- **导出当前图片**：点击右侧的“导出当前图片为 JSON”或“导出当前图片为 COCO”按钮，导出当前图片的标注。
- **批量导出**：点击“批量导出已完成图片”按钮，导出所有已完成图片的标注。
- **导入标注**：点击“批量导入标注”按钮，导入外部的标注文件。

## 常用快捷键

| 操作 | 快捷键 |
|------|--------|
| 暂存当前区域 | `Enter` |
| 保存整株 | `Shift+Enter` |
| 撤销 | `Ctrl+Z` |
| 重做 | `Ctrl+Y` |
| 删除选中植株 | `Delete` |
| 边缘吸附 | `Shift` |
| 批量加载图片 | `Ctrl+Shift+O` |
| 上一张 | `Left` |
| 下一张 | `Right` |
| 膨胀点选 | `G` |
| 忽略区域 | `I` |

## 界面结构

- 左侧：辅助功能、标注操作、实例列表、图片导航
- 中间左画布：主编辑画布，显示正式层、候选层和当前手工预览层
- 中间右画布：正式实例总览，不显示候选层，每个实例最左端显示红色的 plant ID
- 右侧：项目状态、属性面板、文件加载、导出、帮助

## 目录结构

当前仓库里的目录可以按“源码目录”和“运行期目录”来理解。

### 源码与仓库目录

```text
maize_preseg_tool/
├─ .git/                  # Git 仓库元数据
├─ .idea/                 # IDE 工程配置，不参与程序运行
│  └─ inspectionProfiles/ # IDE 检查配置
├─ components/            # PyQt 组件：画布、工具栏、帮助弹窗
├─ images_raw/            # 示例原始图片目录
├─ maize_annotations/     # 运行后生成的标注、项目状态、数据集
├─ models/                # 模型相关封装和项目模型 registry 逻辑
├─ services/              # 后台推理和训练管理服务
├─ test/                  # 你的测试图片目录；加载后会形成一个项目
├─ tools/                 # 独立脚本，如项目训练入口
├─ ui/                    # 右侧属性面板等 UI 模块
├─ utils/                 # 数据持久化、数据集构建、项目上下文、图像处理
└─ __pycache__/           # Python 字节码缓存
```

各目录职责：

- `.git/`：版本控制目录。
- `.idea/`：PyCharm/IDEA 的本地工程配置。
- `components/`：核心交互组件。
- `images_raw/`：仓库内示例图片。
- `maize_annotations/`：程序运行后自动生成的数据目录。
- `models/`：模型封装、模型注册表和版本管理逻辑。
- `services/`：推理线程、训练进程调度等后台服务。
- `test/`：你当前用于测试标注和训练的图片目录。
- `tools/`：可独立运行的脚本，当前主要是 `train_project_model.py`。
- `ui/`：较独立的界面模块。
- `utils/`：通用工具和项目/数据集逻辑。
- `__pycache__/`：运行时自动生成，可忽略。

### `components/` 目录

```text
components/
├─ help_dialog.py   # 帮助弹窗
├─ image_label.py   # 主画布/总览画布，负责绘制、候选层、顶点编辑、缩放拖动
└─ toolbars.py      # 左右工具栏和按钮创建
```

### `models/` 目录

```text
models/
├─ project_model_registry.py # 项目级模型版本管理、active/previous 回退
└─ sam_model.py              # SAM 模型封装
```

### `services/` 目录

```text
services/
├─ inference_service.py # 单图预标注推理与后台 QThread worker
└─ training_manager.py  # 自动训练阈值检查、QProcess 训练管理、进度解析
```

### `tools/` 目录

```text
tools/
└─ train_project_model.py # 独立训练入口，供 GUI 后台进程调用
```

### `ui/` 目录

```text
ui/
└─ annotation_properties_panel.py # 右侧项目状态、属性编辑面板
```

### `utils/` 目录

```text
utils/
├─ annotation_schema.py # 正式实例/候选实例/图片状态兼容结构
├─ auxiliary_algorithms.py
├─ data_manager.py      # .maize 读写、JSON/COCO 导出
├─ dataset_builder.py   # 项目级 YOLO 数据集构建、固定验证集 manifest
├─ helpers.py           # 通用辅助函数
├─ image_processor.py   # 预处理、边缘图和吸附辅助
└─ project_context.py   # 项目目录、metadata、image_records 管理
```

### 运行期目录 `maize_annotations/`

程序运行后会自动生成：

```text
maize_annotations/
├─ projects/      # 单张图片对应的 .maize 标注文件
└─ project_state/ # 项目级状态、模型、数据集、导出
```

### `maize_annotations/project_state/<project_id>/` 目录

每个项目会有一个独立目录：

```text
maize_annotations/project_state/<project_id>/
├─ dataset/            # 当前项目构建出的训练集
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
│     ├─ train/
│     └─ val/
├─ exports/            # 批量导出已完成图片时生成的导出目录
├─ logs/               # 项目级日志目录
├─ models/             # 当前项目的模型目录
│  └─ versions/
│     └─ model_v000x/  # 每次训练的独立模型版本目录
├─ image_records.json  # 图片级完成状态、dirty 状态、annotation hash
├─ project_metadata.json # 项目配置、计数、active 模型版本
└─ split_manifest.json # 固定验证集清单
```

`model_v000x/` 下面通常包含：

```text
model_v000x/
├─ best.pt
├─ last.pt
├─ metrics_summary.json
├─ train.log
└─ train_args.yaml
```

## 数据组织

项目状态目录：

```text
maize_annotations/project_state/<project_id>/
```

其中主要包含：

- `project_metadata.json`：项目配置、训练阈值、active 模型版本、计数信息
- `image_records.json`：图片级完成状态、dirty 状态、annotation hash
- `split_manifest.json`：固定验证集清单
- `models/`：模型 registry 和版本目录
- `dataset/`：当前项目构建出的 YOLO 数据集

单图正式标注保存于：

```text
maize_annotations/projects/*.maize
```

## 训练与验证集规则

- 默认自动训练阈值：`5`
- 固定验证集比例：`20%`
- 最小验证集数量：`1`
- 新增已完成图片默认进 `train`
- 只有点击“重建验证集”时才会重建 `split_manifest`
- 训练初始化权重：
  - 若已有稳定模型，用上一个 stable `best.pt`
  - 否则用 `yolo11n-seg.pt`

## 预标注规则

自动预标注只在以下条件同时满足时触发：

- 当前项目已有 active 模型
- 当前图片没有正式实例
- 当前图片不是“已完成”

另外，右侧提供了“对当前图执行AI预标注”按钮：

- 可手动对当前图再次跑一遍预标注
- 即使当前图已有正式实例，也可以查看候选层
- 不会覆盖正式层

## 安装

基础安装：

```bash
pip install -r requirements.txt
```

启动：

```bash
python main.py
```

### GPU 训练说明

如果要用显卡训练，必须安装 CUDA 版 PyTorch。仅执行 `pip install -r requirements.txt` 可能会装到 CPU 版。

当前这台机器已验证可用的一种安装方式是：

```bash
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

训练脚本会在启动时输出 `torch / CUDA` 运行时状态；如果检测到你请求 GPU 但当前环境里的 torch 不能用 CUDA，会直接报错，而不是静默退回 CPU。

## 导出

支持：

- 当前图片简单 JSON
- 当前图片 COCO JSON
- 项目级 YOLO segmentation 数据集
- 批量导出已完成图片对应 JSON

YOLO 导出目录：

```text
dataset/
├─ images/train
├─ images/val
├─ labels/train
├─ labels/val
└─ data.yaml
```

YOLO 导出只包含：

- 正式实例
- 已完成图片
- 类别和 polygon



## 主要依赖

- `PyQt5`
- `Pillow`
- `numpy`
- `opencv-python`
- `torch`
- `ultralytics`
- `PyYAML`

## 目前已知情况

- 候选层显示和正式层保存已经分离
- 训练、推理、模型回退主流程已接通
- 旧版 `.maize` 会自动补默认字段，保持兼容
- SAM 和区域生长保留入口，但不是当前推荐主流程

如果你只是第一次上手，建议直接看 [使用说明.md](使用说明.md)。