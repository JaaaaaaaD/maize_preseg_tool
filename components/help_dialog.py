from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QTextBrowser, QVBoxLayout


class ZoomableHelpText(QTextBrowser):
    """支持 Ctrl+滚轮 缩放字体的帮助文本区域。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_steps = 0
        self._min_zoom_steps = -8
        self._max_zoom_steps = 20
        self.viewport().installEventFilter(self)

    def _zoom_by_wheel_delta(self, delta_value):
        if delta_value > 0 and self._zoom_steps < self._max_zoom_steps:
            self.zoomIn(1)
            self._zoom_steps += 1
            return True
        if delta_value < 0 and self._zoom_steps > self._min_zoom_steps:
            self.zoomOut(1)
            self._zoom_steps -= 1
            return True
        return False

    def set_zoom_steps(self, target_steps):
        target_steps = max(self._min_zoom_steps, min(self._max_zoom_steps, int(target_steps)))
        while self._zoom_steps < target_steps:
            self.zoomIn(1)
            self._zoom_steps += 1
        while self._zoom_steps > target_steps:
            self.zoomOut(1)
            self._zoom_steps -= 1

    def get_zoom_percent(self):
        # QTextBrowser 每步 zoom 大约 20% 左右的视觉变化，这里给出可读近似值
        return max(40, int(round(100 * (1.2 ** self._zoom_steps))))

    def eventFilter(self, obj, event):
        if obj is self.viewport() and event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            delta_value = event.angleDelta().y()
            if delta_value == 0:
                delta_value = event.pixelDelta().y()
            if self._zoom_by_wheel_delta(delta_value):
                event.accept()
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta_value = event.angleDelta().y()
            if delta_value == 0:
                delta_value = event.pixelDelta().y()
            if self._zoom_by_wheel_delta(delta_value):
                event.accept()
            else:
                event.ignore()
            return
        super().wheelEvent(event)


class HelpDialog(QDialog):
    """使用说明弹窗。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("玉米标注与项目级预标注工具 - 使用说明")
        self.setMinimumSize(700, 520)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self._normal_zoom_steps = 0
        self._maximized_zoom_steps = 7
        self.init_ui()
        self.resize_to_available_screen()
        self._sync_zoom_with_window_state()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.help_text = ZoomableHelpText()
        self.help_text.setOpenExternalLinks(False)
        self.help_text.setReadOnly(True)
        self.help_text.setHtml(self._build_help_html())
        layout.addWidget(self.help_text)
        self.installEventFilter(self)
        self.help_text.installEventFilter(self)
        self.help_text.viewport().installEventFilter(self)
        self.label_zoom_hint = QLabel("缩放: 100%")
        self.label_zoom_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.label_zoom_hint)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier and hasattr(self, "help_text"):
            delta_value = event.angleDelta().y()
            if delta_value == 0:
                delta_value = event.pixelDelta().y()
            if self.help_text._zoom_by_wheel_delta(delta_value):
                self._update_zoom_hint()
                event.accept()
            return True
        return super().eventFilter(obj, event)

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self._sync_zoom_with_window_state()
        super().changeEvent(event)

    def _sync_zoom_with_window_state(self):
        if not hasattr(self, "help_text"):
            return
        if self.isMaximized():
            self.help_text.set_zoom_steps(self._maximized_zoom_steps)
        else:
            self.help_text.set_zoom_steps(self._normal_zoom_steps)
        self._update_zoom_hint()

    def _update_zoom_hint(self):
        if not hasattr(self, "label_zoom_hint") or not hasattr(self, "help_text"):
            return
        self.label_zoom_hint.setText(f"缩放: {self.help_text.get_zoom_percent()}%")

    @staticmethod
    def _build_help_html():
        return """
        <h2>玉米标注与项目级预标注工具 使用说明</h2>

        <h3>1. 推荐使用流程（主流程）</h3>
        <ol>
            <li><b>加载图片：</b>点击“批量加载图片”导入一组待标注图像。</li>
            <li><b>逐图标注：</b>左键加点形成区域，按 Enter 暂存区域，按 Shift+Enter 保存整株。</li>
            <li><b>切换图片：</b>用“上一张/下一张”或方向键切图，继续标注。</li>
            <li><b>标记完成：</b>当前图确认无误后，点击“标记当前图片为已完成”。</li>
            <li><b>批量导出：</b>使用“批量导出已完成(coco格式)”导出结果。</li>
        </ol>

        <h3>2. 当前界面主要区域</h3>
        <ul>
            <li><b>左侧工具区：</b>标注、导航、植株管理、计时与辅助操作。</li>
            <li><b>中间画布区：</b>左侧为编辑画布，右侧为正式实例总览。</li>
            <li><b>右侧工具区：</b>文件操作、SAM模型、导入导出与辅助入口。</li>
        </ul>

        <h3>3. 手动标注核心操作</h3>
        <ul>
            <li><b>绘制区域：</b>左键添加顶点，形成闭合区域后按 Enter 暂存。</li>
            <li><b>保存整株：</b>将当前暂存区域合并为一个正式实例（Shift+Enter）。</li>
            <li><b>撤销/重做：</b>支持 Ctrl+Z / Ctrl+Y。</li>
            <li><b>删除实例：</b>选中植株后按 Delete 或点击“删除选中植株”。</li>
            <li><b>继续标注：</b>选择植株后点击“继续标注选中植株”，可在原实例上继续修改。</li>
        </ul>

        <h3>4. 辅助与微调功能</h3>
        <ul>
            <li><b>边缘吸附：</b>“边缘吸附: 开启/关闭”，帮助顶点贴边。</li>
            <li><b>忽略区域：</b>按 I 或点击按钮切换，标记无需参与训练/评估的区域。</li>
            <li><b>去除区域：</b>去除当前实例中不应保留的部分。</li>
            <li><b>微调模式：</b>进入后可执行加点、删点、切分、合并、画笔建区、画笔删暂存等操作。</li>
            <li><b>透明预览：</b>“区域透明预览”便于观察底图纹理和边界。</li>
        </ul>

        <h3>5. 预标注（SAM）流程</h3>
        <ol>
            <li>先在右侧“SAM模型”加载模型（可按需训练）。</li>
            <li>点击“框选预标注”，在画布上框出目标区域。</li>
            <li>得到候选后可“接受候选并微调”、或“拒绝当前 proposal”、或“忽略当前 proposal”。</li>
            <li>如需分析微调过程，可导出/导入预标注调整记录，并恢复 selected record final。</li>
        </ol>

        <h3>6. 导入导出与项目恢复</h3>
        <ul>
            <li><b>批量导入数据：</b>从目录导入已有 COCO 标注。</li>
            <li><b>批量导出已完成：</b>导出所有“已完成”图片的 COCO 结果。</li>
            <li><b>自动缓存恢复：</b>若上次未完整结束，下次加载同一批图片时可恢复草稿与定位。</li>
        </ul>

        <h3>7. 常用快捷键</h3>
        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 100%;">
            <tr><th align="left">功能</th><th align="left">快捷键</th></tr>
            <tr><td>批量加载图片</td><td>Ctrl+Shift+O</td></tr>
            <tr><td>暂存当前区域</td><td>Enter</td></tr>
            <tr><td>保存整株</td><td>Shift+Enter</td></tr>
            <tr><td>撤销 / 重做</td><td>Ctrl+Z / Ctrl+Y</td></tr>
            <tr><td>上一张 / 下一张</td><td>Left / Right</td></tr>
            <tr><td>删除选中植株</td><td>Delete</td></tr>
            <tr><td>删除选中暂存区域</td><td>Ctrl+D</td></tr>
            <tr><td>切换忽略区域</td><td>I</td></tr>
            <tr><td>切换边缘吸附</td><td>Shift</td></tr>
        </table>

        <h3>8. 常见问题</h3>
        <ul>
            <li><b>候选无法微调：</b>先“接受候选并微调”，再进入对应微调工具。</li>
            <li><b>无法开始训练：</b>请先退出微调/候选状态，并确保有已完成图片。</li>
            <li><b>担心中途退出丢失：</b>切图、编辑和退出时都会自动写入草稿缓存。</li>
        </ul>
        """

    def resize_to_available_screen(self):
        """根据屏幕大小自适应弹窗尺寸。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(820, 700)
            return

        geometry = screen.availableGeometry()
        width = max(self.minimumWidth(), min(int(geometry.width() * 0.65), 1120))
        height = max(self.minimumHeight(), min(int(geometry.height() * 0.82), 900))
        width = min(width, geometry.width())
        height = min(height, geometry.height())
        self.resize(width, height)