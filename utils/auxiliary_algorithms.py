# 辅助算法工具

import cv2
import numpy as np
from collections import deque
from config import REGION_GROWING_THRESHOLD

def perform_region_growing(color_image, seed_point, threshold=REGION_GROWING_THRESHOLD, progress_callback=None):
    """执行区域生长算法"""
    if color_image is None:
        return None
    
    try:
        # 初始化掩码
        height, width = color_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 种子点
        seed_x, seed_y = int(seed_point[0]), int(seed_point[1])
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            return None
        
        # 种子颜色
        seed_color = color_image[seed_y, seed_x]
        
        # 使用队列进行广度优先搜索
        queue = deque()
        queue.append((seed_y, seed_x))
        mask[seed_y, seed_x] = 255
        
        # 8邻域
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        # 计算总像素数用于进度显示
        total_pixels = height * width
        processed_pixels = 0
        
        while queue:
            y, x = queue.popleft()
            processed_pixels += 1
            
            # 更新进度
            if processed_pixels % 1000 == 0:
                progress = min(90, int(20 + (processed_pixels / total_pixels) * 70))
                if progress_callback:
                    progress_callback(progress)
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] == 0:
                    # 计算颜色差异
                    pixel_color = color_image[ny, nx]
                    color_diff = np.linalg.norm(seed_color - pixel_color)
                    
                    if color_diff < threshold:
                        mask[ny, nx] = 255
                        queue.append((ny, nx))
        
        if progress_callback:
            progress_callback(100)
        
        return mask
    except Exception as e:
        print(f"Region growing error: {e}")
        return None

def convert_mask_to_polygon(mask):
    """将掩码转换为多边形"""
    if mask is None:
        return []
    
    try:
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 选择最大的轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 简化轮廓
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 转换为多边形点
        polygon_points = [(point[0][0], point[0][1]) for point in approx]
        
        return polygon_points
    except Exception as e:
        print(f"Convert mask to polygon error: {e}")
        return []