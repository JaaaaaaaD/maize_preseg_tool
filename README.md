# 玉米实例分割标注与预标注工具

本项目是一个基于 `PyQt5` 的桌面标注系统，面向玉米图像的多实例、多区域分割场景。系统同时支持人工标注、SAM 辅助预标注、实例微调、批量图片管理、COCO 导入导出，以及预标注修正记录导出。

当前代码版本的定位不是“通用标注平台”，而是围绕玉米部位实例分割的一套专用工作流工具。默认类别为：

- `stem`
- `leaf`
- `ear`

## 系统目标

系统试图解决三件事：

1. 提供稳定的人工多边形标注能力
2. 将 SAM 预标注接入到人工修正流程中
3. 在批量图片场景下管理“未完成 / 已完成 / 可导出”的标注状态

## 系统架构

### 总体分层

当前代码大致分为 6 层：

1. 启动层
2. 主窗口业务层
3. 画布交互层
4. 模型与服务层
5. 数据持久化层
6. UI 组件层

### 目录结构

```text
maize_preseg_tool/
├─ app/                     # 主窗口业务逻辑，按职责拆成多个 mixin
├─ components/              # 画布、工具栏、帮助对话框等组件
├─ models/                  # SAM 管理与模型封装
├─ services/                # 训练与推理服务
├─ ui/                      # 右侧属性面板等 UI 模块
├─ utils/                   # 数据结构、导入导出、图像处理、状态机等工具
├─ config.py                # 全局配置和快捷键
├─ main.py                  # 程序入口
└─ README.md
```

### 入口与主窗口

- [main.py](/E:/PycharmProjects/maize_preseg_tool/main.py)
  - 创建 `QApplication`
  - 实例化主窗口 `MainWindow`

- [app/main_window.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window.py)
  - 通过多重继承拼装主窗口
  - 当前主窗口由以下模块组合：
    - `MainWindowBase`
    - `MainWindowProjectMixin`
    - `MainWindowAnnotationMixin`
    - `MainWindowIOMixin`
    - `MainWindowSamMixin`

这种结构的特点是：

- UI 初始化集中在基础类
- 标注、项目管理、SAM、IO 分别放在不同文件
- 逻辑可拆，但状态跨文件较多，理解时要按主流程串起来看

### 主窗口业务分工

- [app/main_window_base.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_base.py)
  - 创建三栏界面
  - 初始化画布、工具栏、右侧属性面板
  - 维护全局运行状态
  - 注册快捷键
  - 维护图片固定序号映射 `image_sequence_map`
  - 挂载轻量交互状态机 `InteractionStateMachine`

- [app/main_window_project.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_project.py)
  - 批量加载图片
  - 切换上一张 / 下一张
  - 标记图片已完成 / 未完成
  - 切图时保存当前图状态、加载目标图状态

- [app/main_window_annotation.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_annotation.py)
  - 人工标注流程
  - 植株实例管理
  - 微调、加点、删点、切割、删除暂存区域
  - 撤销 / 重做
  - 状态栏与交互按钮刷新

- [app/main_window_sam.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_sam.py)
  - SAM 模型加载
  - 训练触发
  - 框选预标注
  - 接受候选并进入微调
  - 预标注修正记录管理与导出

- [app/main_window_io.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_io.py)
  - 批量导入 / 导出 COCO 数据

### 画布交互层

- [components/image_label.py](/E:/PycharmProjects/maize_preseg_tool/components/image_label.py)
  - 是系统的核心交互组件
  - 左画布负责编辑
  - 右画布负责总览
  - 主要职责：
    - 图片显示、缩放、拖拽
    - 多边形绘制
    - 实例选择
    - 顶点拖拽
    - 加点 / 删点
    - 暂存区域切割
    - 候选层显示
    - 正式实例与暂存区域高亮
    - 将几何变更回传给主窗口

### UI 组件层

- [components/toolbars.py](/E:/PycharmProjects/maize_preseg_tool/components/toolbars.py)
  - 创建左侧与右侧各组按钮
- [ui/annotation_properties_panel.py](/E:/PycharmProjects/maize_preseg_tool/ui/annotation_properties_panel.py)
  - 右侧属性面板
  - 当前版本是简化实现，主要保留了按钮和面板容器接口

### 模型与服务层

- [models/sam_manager.py](/E:/PycharmProjects/maize_preseg_tool/models/sam_manager.py)
  - SAM 模型生命周期管理
- [services/sam_training_manager.py](/E:/PycharmProjects/maize_preseg_tool/services/sam_training_manager.py)
  - 训练流程封装
- [app/workers.py](/E:/PycharmProjects/maize_preseg_tool/app/workers.py)
  - 训练线程 worker

### 数据与工具层

- [utils/data_manager.py](/E:/PycharmProjects/maize_preseg_tool/utils/data_manager.py)
  - COCO 构建、导入、导出
  - 原子写入
  - 旧文件备份
- [utils/annotation_schema.py](/E:/PycharmProjects/maize_preseg_tool/utils/annotation_schema.py)
  - 内部标注结构标准化
- [utils/sam_utils.py](/E:/PycharmProjects/maize_preseg_tool/utils/sam_utils.py)
  - SAM 轮廓转多边形等辅助逻辑
- [utils/interaction_state.py](/E:/PycharmProjects/maize_preseg_tool/utils/interaction_state.py)
  - 轻量交互状态机

## 当前交互状态机

当前版本引入了一层轻量状态机，用于统一主交互状态。已定义的状态包括：

- `idle`
- `preannotation_box`
- `preannotation_candidate`
- `fine_tune`
- `fine_tune_add_vertex`
- `fine_tune_split_staging`
- `ignore_region`
- `removal_region`

状态机的作用：

- 给工具栏和状态栏提供统一状态源
- 收敛“候选未接受前不能继续交互”这类规则
- 避免多模式并存时的错误高亮和误操作

需要注意的是，当前实现仍是“状态机 + 旧布尔状态并行”的过渡结构，不是完全状态机驱动。

## 功能说明

### 1. 图片与项目管理

- 批量加载图片
- 上一张 / 下一张切换
- 显示当前进度
- 标记图片为已完成或未完成
- 按图片维护标注状态

### 2. 人工标注

- 左键逐点绘制多边形
- 支持一个植株实例下包含多个区域
- 区域支持单独标签
- 当前内置标签为：
  - `stem`
  - `leaf`
  - `ear`

### 3. 实例管理

- 保存当前区域
- 保存整株
- 删除选中植株
- 撤销删除植株
- 继续标注已有植株
- 右侧总览同步显示正式实例

### 4. 微调能力

针对正式实例，系统支持微调模式：

- 拖拽顶点
- 添加顶点
- 删除顶点
- 修改暂存区域标签
- 删除选中的暂存区域
- 切割选中的暂存区域
- 挖孔 / 去除局部区域

其中：

- 删除选中的暂存区域快捷键为 `Ctrl+D`
- 点击暂存区域时，左侧 `label` 下拉框会同步为该区域标签

### 5. 忽略区与去除区

- 忽略区用于表示训练时不应参与学习的区域
- 去除区主要用于对当前实例局部做裁切或挖孔

### 6. 撤销 / 重做

系统维护两套编辑栈：

- 普通标注栈
- 微调栈

支持：

- 撤销
- 重做
- 微调中的状态回滚

### 7. SAM 辅助预标注

当前 SAM 流程支持：

- 加载本地 SAM 权重
- 启动训练
- 框选 ROI 做预标注
- 在候选层查看结果
- 接受候选并转为正式实例
- 拒绝当前候选
- 接受后进入微调

当前约束是：

- 候选未被接受前，系统应限制用户继续执行手工标注或正式微调
- 只有接受候选后，候选区域才进入正式编辑链路

### 8. 批量导入导出

- 批量导入 COCO 标注
- 批量导出已完成图片的 COCO 标注
- 批量导出已完成图片的预标注修正记录

## 数据结构与文件组织

### 1. 内部运行容器

当前运行期主要依赖以下内存结构：

- `image_paths`
  - 当前批量加载的图片列表
- `image_sequence_map`
  - 图片路径到固定序号的映射
- `coco_container`
  - 以图片路径为键的标注容器
- `preannotation_adjustment_records`
  - 当前图片的预标注修正记录列表

### 2. COCO 标注文件

系统导出的标注文件是标准 COCO 的扩展形式，由 [utils/data_manager.py](/E:/PycharmProjects/maize_preseg_tool/utils/data_manager.py) 负责生成。

主要特点：

- `images` 中保留原图路径和图像状态
- `annotations` 中记录实例分割多边形
- 忽略区以 `iscrowd=1` 形式保存
- `info.custom` 中保存：
  - 是否已完成
  - 最后修改时间
  - 项目 ID
  - 当前 plant ID

批量导出时，文件名形如：

- `xxx_anno.json`

### 3. 预标注修正记录文件

当前版本将预标注修正记录按图片拆分存储，文件名规则为：

- `image_序号_correction.json`

这里的“序号”不是临时当前索引，而是图片在本次批量加载后由 `image_sequence_map` 分配的固定序号。

例如：

- 第一张图：`image_1_correction.json`
- 第二张图：`image_2_correction.json`

记录文件由 [app/main_window_sam.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_sam.py) 管理，单条记录通常包含：

- `record_id`
- `image_path`
- `created_at`
- `model_path`
- `model_type`
- `roi_box`
- `candidate_id`
- `confidence`
- `original_polygons`
- `final_polygons`
- `formal_instance_id`
- `status`
- `operations`

`operations` 用来描述该预标注实例在人工修正过程中发生的操作。

### 4. 当前 correction 的实现边界

需要明确一点：当前代码里 correction 记录已经按图片拆分并支持导出，但自动保存逻辑目前是关闭的。

也就是说：

- 内存中会维护预标注修正记录
- 导出功能只导出“已完成图片”的 correction 文件
- 若后续要把 correction 作为强审计链条使用，还需要继续加强“实时落盘”和“一致性校验”

## 标注流程

下面是建议按当前实现使用的标准流程。

### 流程 A：纯人工标注

1. 批量加载图片
2. 切到目标图片
3. 在左侧选择标签 `stem / leaf / ear`
4. 在左画布逐点绘制一个区域
5. 按 `Enter` 暂存当前区域
6. 继续绘制该植株的其他区域，重复暂存
7. 按 `Shift+Enter` 保存整株实例
8. 如有需要，继续标注下一株
9. 标记该图片为已完成
10. 批量导出已完成图片的 COCO 标注

### 流程 B：预标注辅助标注

1. 加载图片
2. 加载 SAM 模型
3. 点击“框选预标注”
4. 在图中框选 ROI
5. 系统生成候选实例
6. 检查候选结果
7. 若接受：
   - 接受候选
   - 系统将其写入正式实例列表
   - 自动进入微调模式
8. 若不接受：
   - 拒绝当前候选
9. 在微调模式下对实例进行修正
10. 修正后退出微调
11. 标记图片已完成
12. 批量导出 COCO 和 correction 记录

### 流程 C：正式实例微调

1. 选中一个正式实例
2. 进入微调模式
3. 选择需要的编辑动作：
   - 拖拽顶点
   - 添加顶点
   - 删除顶点
   - 切割暂存区域
   - 删除暂存区域
   - 修改暂存区域标签
   - 挖孔 / 去除局部
4. 保存微调结果
5. 退出微调模式

## 界面结构

当前主界面是三栏布局。

### 左侧栏

用于工具操作，主要包括：

- 辅助功能
- 标注操作
- 植株管理
- 图片导航

### 中间区域

- 左画布：主编辑画布
- 右画布：正式实例总览画布

### 右侧栏

主要包括：

- 属性面板
- 文件操作
- SAM 操作
- 导入导出
- 辅助按钮

## 快捷键

当前配置定义在 [config.py](/E:/PycharmProjects/maize_preseg_tool/config.py)。

- `Return`：暂存当前区域
- `Shift+Return`：保存整株
- `Ctrl+Z`：撤销
- `Ctrl+Y`：重做
- `Delete`：删除选中植株
- `Ctrl+D`：删除选中的暂存区域
- `Shift`：边缘吸附
- `Ctrl+Shift+O`：批量加载图片
- `Left`：上一张
- `Right`：下一张
- `I`：忽略区模式

## 运行方式

### 环境要求

- Python 3.10 及以上更稳妥
- Windows 桌面环境
- 若要使用训练 / 推理，建议配置 CUDA 版 PyTorch

### 依赖

项目当前依赖见 [requirements.txt](/E:/PycharmProjects/maize_preseg_tool/requirements.txt)，主要包括：

- `PyQt5`
- `Pillow`
- `numpy`
- `opencv-python`
- `torch`
- `ultralytics`
- `PyYAML`
- `segment-anything`

安装示例：

```bash
pip install -r requirements.txt
```

如果需要 GPU 训练或 GPU 推理，建议先按 PyTorch 官方命令安装匹配 CUDA 的 `torch / torchvision / torchaudio`，再补装其余依赖。

### 启动

```bash
python main.py
```

## 当前实现特点与限制

这部分不是设计目标，而是当前代码版本的真实情况。

### 已具备

- 三栏式桌面标注界面
- 人工实例分割标注
- 正式实例微调
- SAM 框选预标注
- 按图片固定序号管理 correction 文件
- 轻量交互状态机
- 批量导入 / 导出 COCO
- 批量导出已完成图片的 correction

### 仍需继续完善

- 右侧属性面板当前是简化版
- 状态机尚未完全取代旧布尔状态
- correction 记录的自动落盘链路还不完整
- 复杂模式切换仍然依赖多处同步

## 建议阅读顺序

如果你准备继续维护这个项目，建议按下面顺序读代码：

1. [main.py](/E:/PycharmProjects/maize_preseg_tool/main.py)
2. [app/main_window.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window.py)
3. [app/main_window_base.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_base.py)
4. [components/image_label.py](/E:/PycharmProjects/maize_preseg_tool/components/image_label.py)
5. [app/main_window_annotation.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_annotation.py)
6. [app/main_window_project.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_project.py)
7. [app/main_window_sam.py](/E:/PycharmProjects/maize_preseg_tool/app/main_window_sam.py)
8. [utils/data_manager.py](/E:/PycharmProjects/maize_preseg_tool/utils/data_manager.py)
9. [utils/interaction_state.py](/E:/PycharmProjects/maize_preseg_tool/utils/interaction_state.py)

这样能先建立主流程，再去看数据结构和边界细节。
