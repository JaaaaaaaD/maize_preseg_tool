# 帮助对话框

from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QScrollArea, QVBoxLayout

class HelpDialog(QDialog):
    """使用说明弹窗"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("玉米植株标注工具 使用说明")
        self.setMinimumSize(640, 480)
        self.init_ui()
        self.resize_to_available_screen()

    def init_ui(self):
        layout = QVBoxLayout(self)
        help_text = QLabel()
        help_text.setWordWrap(True)
        help_text.setText("""
        <h2>玉米植株多区域标注工具 使用说明</h2>

        <h3>一、界面介绍</h3>
        <p>本工具分为三个主要区域：</p>
        <ul>
            <li><b>左侧工具栏</b>：包含文件操作、导航、标注操作、辅助功能和植株管理等工具按钮</li>
            <li><b>中间图像显示区</b>：左侧为实时标注图像，右侧为全局概览图像</li>
            <li><b>右侧工具栏</b>：包含导出和辅助功能按钮</li>
        </ul>

        <h3>二、基本操作流程</h3>
        <ol>
            <li><b>加载图片</b>：点击「批量加载图片」或按 <b>Ctrl+Shift+O</b> 选择多张图片</li>
            <li><b>开始标注</b>：使用鼠标左键点击添加顶点，或使用辅助工具（边缘吸附、膨胀点选、忽略区域）</li>
            <li><b>暂存区域</b>：完成一个区域的标注后，按 <b>Enter</b> 暂存该区域</li>
            <li><b>保存整株</b>：标注完植株的所有区域后，按 <b>Shift+Enter</b> 保存整株</li>
            <li><b>切换图片</b>：点击「上一张」「下一张」或按 <b>←</b> <b>→</b> 方向键</li>
            <li><b>导出标注</b>：点击「导出当前JSON」或「导出当前COCO」保存标注结果</li>
        </ol>

        <h3>三、核心标注操作</h3>
        <ul>
            <li><b>边缘吸附</b>：默认开启，按 <b>Shift</b> 切换开关。开启后，鼠标移动到边缘附近时会自动吸附到边缘点，显示绿色圆圈</li>
            <li><b>膨胀点选</b>：按 <b>G</b> 切换开关，点击图像自动膨胀选择相似颜色区域</li>
            <li><b>忽略区域</b>：按 <b>I</b> 切换开关，绘制区域标记为忽略区域，会用磨砂灰色覆盖该区域</li>
            <li><b>去除区域</b>：按 <b>R</b> 切换开关，在主多边形内部绘制要去除的区域，会显示透明效果露出底层图像</li>
            <li><b>投影框</b>：点击「投影框」按钮，右侧画布会显示蓝色虚线方框，表示当前左侧画布所看到的区域</li>
            <li><b>AI辅助</b>：点击「AI辅助」按钮，可开启或关闭AI自动训练和预标注功能</li>

            <li><b>绘制顶点</b>：鼠标左键点击添加顶点，会显示空心红色小圆（普通标注）或灰色小圆（忽略区域），移动鼠标时会有红色虚线（普通标注）或灰色虚线（忽略区域）连接到当前鼠标位置</li>
            <li><b>暂存当前区域</b>：按 <b>Enter</b>，会自动连接起点和最后一个点形成闭合多边形</li>
            <li><b>保存整株</b>：按 <b>Shift+Enter</b>，将所有暂存的区域保存为一个完整植株，去除区域会自动从主多边形中挖空</li>
            <li><b>智能撤销</b>：按 <b>Ctrl+Z</b>，可撤销上一步操作</li>
        </ul>

        <h3>四、图像浏览操作</h3>
        <ul>
            <li><b>缩放图像</b>：鼠标滚轮滚动，向前滚动放大，向后滚动缩小</li>
            <li><b>拖动图像</b>：鼠标右键按下并拖动，可调整图像位置</li>
        </ul>

        <h3>五、植株管理</h3>
        <ul>
            <li><b>选择植株</b>：在「植株管理」下拉菜单中选择要操作的植株，右侧全局概览会高亮显示选中的植株</li>
            <li><b>删除植株</b>：选择植株后，点击「删除选中植株」或按 <b>Delete</b> 键</li>
            <li><b>标注状态</b>：点击「标记为已标注」/「标记为未标注」切换当前图片的标注状态</li>
        </ul>

        <h3>六、导出功能</h3>
        <ul>
            <li><b>导出当前JSON</b>：将当前图片的标注导出为JSON格式</li>
            <li><b>导出当前COCO</b>：将当前图片的标注导出为COCO格式，适用于目标检测和实例分割任务</li>
            <li><b>批量导出已标注</b>：导出所有标记为已标注的图片的标注数据</li>
        </ul>

        <h3>七、快捷键总结</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th style="padding: 8px; text-align: left;">功能</th>
                <th style="padding: 8px; text-align: left;">快捷键</th>
            </tr>
            <tr>
                <td style="padding: 8px;">批量加载图片</td>
                <td style="padding: 8px;">Ctrl+Shift+O</td>
            </tr>
            <tr>
                <td style="padding: 8px;">上一张图片</td>
                <td style="padding: 8px;">←</td>
            </tr>
            <tr>
                <td style="padding: 8px;">下一张图片</td>
                <td style="padding: 8px;">→</td>
            </tr>
            <tr>
                <td style="padding: 8px;">暂存当前区域</td>
                <td style="padding: 8px;">Enter</td>
            </tr>
            <tr>
                <td style="padding: 8px;">保存整株</td>
                <td style="padding: 8px;">Shift+Enter</td>
            </tr>
            <tr>
                <td style="padding: 8px;">撤销操作</td>
                <td style="padding: 8px;">Ctrl+Z</td>
            </tr>
            <tr>
                <td style="padding: 8px;">删除选中植株</td>
                <td style="padding: 8px;">Delete</td>
            </tr>
            <tr>
                <td style="padding: 8px;">切换边缘吸附</td>
                <td style="padding: 8px;">Shift</td>
            </tr>
            <tr>
                <td style="padding: 8px;">切换膨胀点选</td>
                <td style="padding: 8px;">G</td>
            </tr>
            <tr>
                <td style="padding: 8px;">切换忽略区域</td>
                <td style="padding: 8px;">I</td>
            </tr>
            <tr>
                <td style="padding: 8px;">切换去除区域</td>
                <td style="padding: 8px;">R</td>
            </tr>
            
        </table>

        <h3>八、常见问题解决</h3>
        <ul>
            <li><b>边缘吸附不工作</b>：检查Shift键是否被按下，或在工具栏中查看边缘吸附状态</li>

            <li><b>标注数据丢失</b>：软件会自动保存标注进度，切换图片时会自动保存</li>
            <li><b>图片拖动时鼠标与光标不重合</b>：这是正常现象，拖动完成后会自动校准</li>
        </ul>

        <h3>九、性能提示</h3>
        <ul>
            <li>仅当标注修改时才会自动保存，切换图片无卡顿</li>
            <li>预处理数据会被缓存，再次打开相同图片时速度更快</li>
            <li>对于大型图片，建议先缩小后再进行标注</li>
        </ul>
        """)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(help_text)
        layout.addWidget(scroll)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def resize_to_available_screen(self):
        """根据屏幕大小自适应弹窗尺寸。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(780, 680)
            return

        geometry = screen.availableGeometry()
        width = max(self.minimumWidth(), min(int(geometry.width() * 0.6), 980))
        height = max(self.minimumHeight(), min(int(geometry.height() * 0.75), 820))
        width = min(width, geometry.width())
        height = min(height, geometry.height())
        self.resize(width, height)