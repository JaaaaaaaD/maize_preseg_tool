# 帮助对话框

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class HelpDialog(QDialog):
    """使用说明弹窗"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("玉米植株标注工具 使用说明")
        self.setGeometry(200, 200, 780, 680)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        help_text = QLabel()
        help_text.setWordWrap(True)
        help_text.setText("""
        <h2>玉米植株多区域标注工具 使用说明</h2>

        <h3>一、批量标注</h3>
        <ul>
            <li><b>批量加载图片</b>：点击「批量加载图片」或按 <b>Ctrl+Shift+O</b></li>
            <li><b>自动保存进度</b>：仅当标注修改时自动保存，切换图片无卡顿</li>
            <li><b>切换图片</b>：点击「上一张」「下一张」或按 <b>←</b> <b>→</b> 方向键</li>
        </ul>

        <h3>二、核心操作</h3>
        <ul>
            <li><b>边缘吸附</b>：默认开启，按 <b>Shift</b> 切换开关</li>
            <li><b>膨胀点选</b>：按 <b>G</b> 切换开关，点击图像自动膨胀选择相似区域</li>
            <li><b>SAM分割</b>：按 <b>S</b> 切换开关，点击图像使用AI智能分割</li>
            <li><b>绘制顶点</b>：鼠标左键点击</li>
            <li><b>暂存当前区域</b>：按 <b>Enter</b></li>
            <li><b>保存整株</b>：按 <b>Shift+Enter</b></li>
            <li><b>智能撤销</b>：按 <b>Ctrl+Z</b></li>
        </ul>

        <h3>三、图像浏览操作</h3>
        <ul>
            <li><b>缩放图像</b>：鼠标滚轮滚动</li>
            <li><b>拖动图像</b>：鼠标右键按下并拖动</li>
        </ul>
        """)
        layout.addWidget(help_text)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)