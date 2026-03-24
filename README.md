# 玉米植株预标注工具

这是一个基于 PyQt5 的桌面标注工具，面向玉米植株实例分割场景。程序以“手工勾边 + 边缘吸附 + 自动保存”为主线，支持批量浏览图片、按植株组织多区域多边形、导出 JSON / COCO 标注结果，并预留了区域生长与 SAM 分割接口。

当前运行入口是 `main.py`，核心界面由左右工具栏和中间双画布组成：

- 左侧工具栏：辅助功能、标注操作、植株管理、图片导航
- 中间画布：左侧为主标注视图，右侧为整图总览视图
- 右侧工具栏：文件加载、导出、帮助与标注状态切换

## 1. 项目定位

这个仓库更接近“半自动标注桌面工具”而不是训练脚本：

- 输入是本地图片
- 处理核心是交互式植株区域标注
- 中间结果会自动保存为项目文件
- 最终结果可导出为简单 JSON 或 COCO 格式

推荐把它理解为：给玉米单株做多区域实例标注的 GUI 工具。

## 2. 当前可用能力

以当前 `main.py` 实际接线为准，主流程中可稳定使用的能力包括：

- 批量加载图片和单张加载
- 左右方向键切换图片
- 图片预处理缓存
- 边缘吸附辅助点选
- 手工添加多边形顶点
- `Enter` 暂存当前区域
- `Shift+Enter` 将多个区域合并保存为一株植株
- `Ctrl+Z` 撤销当前点或当前暂存区域，或撤销整株新增/删除
- 植株列表选择与删除
- 切图时自动保存当前标注
- 重新打开图片时自动恢复已有标注
- 导出当前图片的简单 JSON
- 导出当前图片的 COCO JSON
- 批量导出“已标记为已标注”的图片对应 JSON

## 3. 程序主流程

### 3.1 启动阶段

程序从 `main.py` 启动：

1. 创建 `QApplication`
2. 初始化 `MainWindow`
3. 构建 UI、快捷键、状态栏
4. 自动创建标注目录 `./maize_annotations/projects`

### 3.2 加载图片

加载图片后会进入以下链路：

1. `load_batch_images()` 或 `load_single_image()` 读取文件路径
2. `goto_image()` 切到目标图片
3. 若当前图片有未保存修改，则先执行自动保存
4. 通过 `utils.helpers.load_image()` 加载图片
5. 通过 `utils.image_processor.preprocess_image()` 做预处理
6. 将 `(foreground_mask, edge_map)` 放入 `preprocess_cache`
7. 左右两个 `ImageLabel` 同步加载图像
8. 通过 `utils.data_manager.load_current_annotation()` 恢复已有标注
9. 更新植株列表、状态栏、标注状态按钮

### 3.3 标注阶段

当前最完整的标注主线是“手工多边形 + 边缘吸附”：

1. 鼠标左键逐点添加顶点
2. 鼠标移动时尝试根据边缘图寻找吸附点
3. `Enter` 调用 `save_current_polygon()` 暂存一个闭合区域
4. 一个植株可由多个区域组成
5. `Shift+Enter` 调用 `confirm_preview_and_save()`，将多个区域合并成一个植株对象
6. 新植株会加入 `plants` 列表，并同步到右侧总览视图

### 3.4 保存与切图

- 当前图片只要发生修改，`annotation_changed` 就会置为 `True`
- 在切换到下一张图片前，`goto_image()` 会触发自动保存
- 自动保存文件后缀为 `.maize`
- 再次打开同一张图片时，会按图片路径自动匹配并恢复标注

### 3.5 导出阶段

当前有三类导出：

- 简单 JSON：导出当前图片的植株标注
- COCO JSON：导出当前图片为 COCO 风格实例分割格式
- 批量导出已标注：把当前会话中被手工标记为“已标注”的图片，对应 JSON 复制到一个时间戳目录

## 4. 标注数据结构

程序内部的单株标注结构大致如下：

```json
{
  "id": 1,
  "polygons": [
    [[x1, y1], [x2, y2], [x3, y3], [x1, y1]]
  ],
  "color": [r, g, b, a],
  "total_area": 1234.5
}
```

其中：

- `id`：植株编号
- `polygons`：该植株包含的一个或多个闭合区域
- `color`：界面展示颜色
- `total_area`：所有区域面积之和

## 5. 目录结构与职责

```text
maize_preseg_tool/
├─ main.py                    # 当前运行入口，主窗口和业务调度
├─ main_copy.py               # 历史/备份版主程序，不是当前入口
├─ config.py                  # 参数、快捷键、路径配置
├─ components/
│  ├─ image_label.py          # 图像显示、绘制、缩放、拖动、点选逻辑
│  ├─ toolbars.py             # 左右工具栏控件创建
│  └─ help_dialog.py          # 使用说明弹窗
├─ utils/
│  ├─ image_processor.py      # 预处理、边缘图和吸附辅助计算
│  ├─ data_manager.py         # 自动保存、恢复、JSON/COCO 导出
│  ├─ auxiliary_algorithms.py # 区域生长、mask 转多边形
│  └─ helpers.py              # 通用辅助函数
├─ models/
│  └─ sam_model.py            # SAM 模型封装
└─ images_raw/                # 示例原始图片
```

## 6. 模块调用关系

主调用链可以概括为：

```text
main.py
├─ components.toolbars.Toolbars
├─ components.image_label.ImageLabel
├─ components.help_dialog.HelpDialog
├─ utils.helpers.load_image
├─ utils.image_processor.preprocess_image
├─ utils.data_manager.save_current_annotation / load_current_annotation
├─ utils.data_manager.export_simple_json / export_coco_format / export_annotated_images
└─ models.sam_model.SamModel
```

其中最关键的职责分工是：

- `MainWindow` 负责调度状态、切图、自动保存、导出和工具栏事件
- `ImageLabel` 负责所有画布交互和绘制
- `data_manager` 负责把内存中的植株对象落盘
- `image_processor` 负责前景掩码和边缘图生成

## 7. 图像预处理逻辑

预处理位于 `utils/image_processor.py`，目标是给“边缘吸附”提供更稳定的候选边界：

- HSV 阈值提取绿色与暗色区域
- 形态学膨胀、闭运算、开运算
- 对 RGB 通道和灰度图分别做锐化与 Canny 边缘检测
- 合并多个边缘结果
- 用前景掩码过滤掉背景边缘

最终输出：

- `foreground_mask`
- `edge_map`

这两个结果会在切图时缓存到 `preprocess_cache`，避免重复计算。

## 8. 自动保存与导出文件

默认标注目录来自 `config.py`：

```python
ANNOTATION_DIR = "./maize_annotations/projects"
```

会产生以下文件：

- 自动保存项目：`{图片名}_{路径哈希}.maize`
- 简单导出：`{图片名}_annotation.json`
- COCO 导出：`{图片名}_coco.json`
- 批量导出目录：`annotated_export_YYYYMMDD_HHMMSS/`

## 9. 快捷键

当前快捷键来自 `config.py`：

| 功能 | 快捷键 |
| --- | --- |
| 暂存当前区域 | `Enter` |
| 保存整株 | `Shift+Enter` |
| 撤销 | `Ctrl+Z` |
| 删除选中植株 | `Delete` |
| 切换边缘吸附 | `Shift` |
| 批量加载图片 | `Ctrl+Shift+O` |
| 上一张图片 | `Left` |
| 下一张图片 | `Right` |
| 切换 SAM 分割 | `S` |
| 切换膨胀点选 | `G` |

## 10. 运行方式

仓库中没有 `requirements.txt`，需要手工准备环境。

基础依赖：

```bash
pip install pyqt5 pillow numpy opencv-python
```

如果需要尝试 SAM，再额外准备：

```bash
pip install torch
pip install segment-anything
```

然后启动：

```bash
python main.py
```

首次体验可以直接加载 `images_raw/` 下的样例图片。

## 11. 当前代码现状与已知限制

这部分很重要。下面是结合当前入口代码梳理出的真实状态。

### 11.1 推荐使用路径

当前最推荐、最完整的使用方式是：

- 批量加载图片
- 使用边缘吸附辅助的手工多边形标注
- 通过“暂存区域 -> 保存整株”组织单株
- 切图自动保存
- 导出 JSON / COCO

### 11.2 已接入口但尚未完全打通的部分

- `区域生长`：界面按钮和算法函数都在，但当前 `main.py` 运行路径里没有把鼠标点击真正接到 `perform_region_growing()`。
- `SAM 分割`：已有模型封装、模型加载入口和画布状态位，但 `components/image_label.py` 中当前点击链路与 `perform_sam_segmentation()` 的参数约定不一致，说明这条链路仍需联调。
- `撤销/重做状态按钮更新`：`update_undo_redo_state()` 仍是空实现。
- `save_undo_state()`：仍是空实现，说明更完整的撤销状态快照还没落地。

### 11.3 行为层面的注意点

- “标记为已标注/未标注”只保存在内存字典 `image_annotation_status` 中，当前不会持久化到磁盘。
- “批量导出已标注”实际导出的是对应 JSON 文件，不会复制原始图片。
- `main_copy.py` 中保留了更大体量的旧版逻辑，但当前程序入口并不使用它。

## 12. 如果后续要继续开发，建议优先处理

1. 把区域生长模式接到 `ImageLabel.mousePressEvent()` 的点击分支。
2. 修正 SAM 点击提示点与 `perform_sam_segmentation()` 的参数约定。
3. 补全撤销/重做状态管理。
4. 为依赖增加 `requirements.txt` 或 `environment.yml`。
5. 将“已标注”状态随项目文件一起持久化。

## 13. 一句话总结

这套程序的核心已经成型：它是一个以 PyQt5 为界面的玉米植株多区域标注工具，当前主打“手工标注 + 边缘吸附 + 自动保存 + JSON/COCO 导出”；而区域生长和 SAM 属于已接入口、但还需要继续打通的增强能力。
