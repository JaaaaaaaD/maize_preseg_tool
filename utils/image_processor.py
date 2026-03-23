# 图像处理工具

import cv2
import numpy as np
from config import (
    LOWER_GREEN, UPPER_GREEN, LOWER_DARK, UPPER_DARK,
    CANNY_THRESHOLD1, CANNY_THRESHOLD2,
    KERNEL_DILATE, KERNEL_CLOSE, KERNEL_OPEN,
    UNSHARP_MASK_KERNEL, UNSHARP_MASK_SIGMA, UNSHARP_MASK_AMOUNT, UNSHARP_MASK_THRESHOLD
)

def unsharp_mask(image, kernel_size=UNSHARP_MASK_KERNEL, sigma=UNSHARP_MASK_SIGMA, 
                 amount=UNSHARP_MASK_AMOUNT, threshold=UNSHARP_MASK_THRESHOLD):
    """应用反锐化遮罩增强图像边界"""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def preprocess_image(pil_image):
    """预处理图像，生成前景掩码和边缘图"""
    try:
        img_np = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. 扩展HSV颜色范围，包含更多的绿色和暗色
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 主绿色范围
        lower_green = np.array(LOWER_GREEN)
        upper_green = np.array(UPPER_GREEN)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 暗绿色和黑色范围
        lower_dark = np.array(LOWER_DARK)
        upper_dark = np.array(UPPER_DARK)
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # 合并掩码
        foreground_mask = cv2.bitwise_or(mask_green, mask_dark)

        # 2. 增强形态学操作，填充过渡区域
        # 先进行膨胀操作，填充小的孔洞和过渡区域
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_DILATE)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_DILATE, kernel_dilate)
        # 再进行闭操作，连接相邻区域
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_CLOSE)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_close)
        # 最后进行开操作，去除噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_OPEN)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel_open)

        # 3. 多通道边缘检测，特别处理过渡像素
        r, g, b = cv2.split(img_bgr)

        # 对每个通道应用适度的锐化处理，避免放大噪声
        # 对绿色通道应用锐化
        sharpened_g = unsharp_mask(g)
        blurred_g = cv2.GaussianBlur(sharpened_g, (5, 5), 1.2)
        edges_g = cv2.Canny(blurred_g, 30, 80)

        # 对红色通道应用锐化
        sharpened_r = unsharp_mask(r)
        blurred_r = cv2.GaussianBlur(sharpened_r, (5, 5), 1.2)
        edges_r = cv2.Canny(blurred_r, 35, 90)

        # 对蓝色通道应用锐化
        sharpened_b = unsharp_mask(b)
        blurred_b = cv2.GaussianBlur(sharpened_b, (5, 5), 1.2)
        edges_b = cv2.Canny(blurred_b, 35, 90)

        # 4. 基于亮度的边缘检测，特别针对暗区域
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # 对灰度图像应用适度锐化
        sharpened_gray = unsharp_mask(gray)
        blurred_gray = cv2.GaussianBlur(sharpened_gray, (5, 5), 1.2)
        # 提高阈值，减少噪声
        edges_gray = cv2.Canny(blurred_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

        # 5. 合并所有边缘
        edges = cv2.bitwise_or(edges_g, edges_r)
        edges = cv2.bitwise_or(edges, edges_b)
        edges = cv2.bitwise_or(edges, edges_gray)

        # 6. 增强边缘，处理不连续边缘，同时过滤噪声
        # 先进行闭操作连接不连续的边缘
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        # 进行开操作去除小的噪声点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)
        
        # 适度膨胀操作增强边缘
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel_dilate)

        # 7. 使用前景掩码过滤边缘
        edge_map = cv2.bitwise_and(edges, edges, mask=foreground_mask)

        return foreground_mask, edge_map
    except Exception as e:
        print(f"preprocess_image error: {e}")
        return None, None

def calculate_snap_point(screen_pos, edge_map, color_image, snap_radius=8, roi_size=8, color_change_threshold=15):
    """计算边缘吸附点"""
    if edge_map is None or color_image is None:
        return None

    try:
        img_h, img_w = edge_map.shape
        if not (0 <= screen_pos[0] < img_w and 0 <= screen_pos[1] < img_h):
            return None

        # 计算ROI区域
        x1 = max(0, int(screen_pos[0] - roi_size // 2))
        y1 = max(0, int(screen_pos[1] - roi_size // 2))
        x2 = min(img_w, x1 + roi_size)
        y2 = min(img_h, y1 + roi_size)

        # 计算ROI内的颜色变化程度
        roi_color = color_image[y1:y2, x1:x2]
        # 计算颜色标准差，衡量颜色变化程度
        color_std = np.std(roi_color)
        # 设置颜色变化阈值，如果颜色变化太小，不认为是边界
        if color_std < color_change_threshold:
            return None
        
        # 计算ROI内占比最多的颜色作为背景色
        # 将ROI内的像素转换为一维数组
        roi_pixels = roi_color.reshape(-1, 3)
        # 计算每种颜色的出现次数
        unique_colors, counts = np.unique(roi_pixels, axis=0, return_counts=True)
        # 找到出现次数最多的颜色
        background_color = unique_colors[np.argmax(counts)]

        # 恢复使用snap_radius计算边缘检测的ROI
        x1_edge = max(0, int(screen_pos[0] - snap_radius))
        y1_edge = max(0, int(screen_pos[1] - snap_radius))
        x2_edge = min(img_w, int(screen_pos[0] + snap_radius + 1))
        y2_edge = min(img_h, int(screen_pos[1] + snap_radius + 1))

        roi_edges = edge_map[y1_edge:y2_edge, x1_edge:x2_edge]
        edge_points = np.column_stack(np.where(roi_edges > 0))

        if len(edge_points) == 0:
            return None

        # 筛选满足条件的边缘点
        valid_edge_points = []
        color_threshold = 30  # 颜色相似度阈值

        for point in edge_points:
            py, px = point
            abs_x = x1_edge + px
            abs_y = y1_edge + py

            # 检查该点是否在颜色图像范围内
            if 0 <= abs_x < color_image.shape[1] and 0 <= abs_y < color_image.shape[0]:
                # 获取当前点的颜色
                current_color = color_image[abs_y, abs_x]
                
                # 检查当前点是否为背景色，如果是则跳过
                current_color_diff = np.linalg.norm(current_color - background_color)
                if current_color_diff < color_threshold:
                    continue

                # 检查多个方向的直线上是否存在颜色相近的像素（非背景色）
                # 包括上下左右四个方向，以及45度和30度的方向
                directions = [
                    (0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
                    (1, 1), (1, -1), (-1, 1), (-1, -1),  # 45度方向
                    (1, 2), (2, 1), (-1, 2), (2, -1), (1, -2), (-2, 1), (-1, -2), (-2, -1)  # 30度方向
                ]
                has_similar_color = False

                for dx, dy in directions:
                    # 计算该方向上的最大步长，确保不超出ROI范围
                    max_step = min(10,  # 直线搜索长度
                                  (x2_edge - abs_x) // abs(dx) if dx != 0 else 10,
                                  (y2_edge - abs_y) // abs(dy) if dy != 0 else 10,
                                  (abs_x - x1_edge) // abs(dx) if dx != 0 else 10,
                                  (abs_y - y1_edge) // abs(dy) if dy != 0 else 10)

                    if max_step < 2:
                        continue

                    # 检查直线上的多个像素
                    similar_count = 0
                    for step in range(1, max_step + 1):
                        nx = abs_x + dx * step
                        ny = abs_y + dy * step

                        if 0 <= nx < color_image.shape[1] and 0 <= ny < color_image.shape[0]:
                            line_color = color_image[ny, nx]
                            line_color_diff = np.linalg.norm(line_color - current_color)
                            line_bg_diff = np.linalg.norm(line_color - background_color)

                            if line_color_diff < color_threshold and line_bg_diff >= color_threshold:
                                similar_count += 1
                                if similar_count >= 2:
                                    has_similar_color = True
                                    break
                        else:
                            break

                    if has_similar_color:
                        break

                if has_similar_color:
                    valid_edge_points.append((abs_x, abs_y))

        if not valid_edge_points:
            return None

        # 计算距离，找到最近的有效边缘点
        min_dist = float('inf')
        closest_point = None
        for point in valid_edge_points:
            dist = ((point[0] - screen_pos[0]) ** 2 + (point[1] - screen_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        return closest_point
    except Exception as e:
        print(f"calculate_snap_point error: {e}")
        return None