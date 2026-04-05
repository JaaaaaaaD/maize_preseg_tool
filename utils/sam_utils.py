# SAM工具函数
import cv2
import numpy as np


def mask_to_polygons(mask, epsilon_ratio=0.005, pixel_interval=50):
    """
    将掩码转换为多边形，按轮廓每隔若干像素采样一个顶点。
    
    Args:
        mask: 二值掩码
        epsilon_ratio: 多边形近似参数
        pixel_interval: 点间隔，默认8像素
        
    Returns:
        多边形列表
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area <= 50:
            continue

        contour_points = contour.reshape(-1, 2)
        sampled_points = []
        step = max(1, int(pixel_interval))
        for index in range(0, len(contour_points), step):
            x, y = contour_points[index]
            sampled_points.append((float(x), float(y)))

        if len(contour_points) >= 3 and contour_points[-1].tolist() != contour_points[0].tolist():
            x, y = contour_points[-1]
            sampled_points.append((float(x), float(y)))

        # 确保多边形闭合
        if len(sampled_points) >= 3:
            if sampled_points[0] != sampled_points[-1]:
                sampled_points.append(sampled_points[0])
            polygons.append(sampled_points)
    
    return polygons

def process_sam_polygons(polygons):
    """
    处理SAM生成的多边形，确保与手动标注格式一致
    
    Args:
        polygons: SAM生成的多边形列表
        
    Returns:
        处理后的多边形列表
    """
    processed_polygons = []
    
    for polygon in polygons:
        # 确保多边形至少有3个点
        if len(polygon) < 3:
            continue
        
        # 确保多边形闭合
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        
        processed_polygons.append(polygon)
    
    return processed_polygons
