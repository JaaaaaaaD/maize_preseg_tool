import os
import re

def rename_wechat_images(folder_path):
    """
    将指定文件夹中的微信图片按顺序重命名
    格式：image_(序号)_(时间戳)
    """
    # 统计非"微信图片"开头的图片数量
    non_wechat_count = 0
    wechat_images = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 只处理文件，不处理目录
        if not os.path.isfile(file_path):
            continue
        
        # 检查是否为图片文件
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            continue
        
        # 统计非"微信图片"开头的图片
        if not filename.startswith('微信图片'):
            non_wechat_count += 1
        else:
            # 收集微信图片
            wechat_images.append(filename)
    
    print(f"非微信图片数量: {non_wechat_count}")
    print(f"微信图片数量: {len(wechat_images)}")
    
    # 处理微信图片重命名
    if wechat_images:
        # 按文件名排序（确保顺序）
        wechat_images.sort()
        
        # 开始重命名
        for i, filename in enumerate(wechat_images, start=1):
            # 提取时间戳
            # 微信图片的格式通常是：微信图片_20240101123456.jpg
            timestamp_match = re.search(r'微信图片_(\d{14})', filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
            else:
                # 如果没有找到时间戳，使用当前时间
                timestamp = '00000000000000'
            
            # 计算新序号
            new_index = non_wechat_count + i
            
            # 构建新文件名
            extension = os.path.splitext(filename)[1]
            new_filename = f"image_{new_index}_{timestamp}{extension}"
            
            # 构建完整路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # 重命名文件
            try:
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
            except Exception as e:
                print(f"重命名失败: {filename}, 错误: {e}")
    else:
        print("没有找到微信图片需要重命名")

if __name__ == "__main__":
    # 示例用法
    folder = input("请输入文件夹路径: ")
    if os.path.exists(folder) and os.path.isdir(folder):
        rename_wechat_images(folder)
    else:
        print("指定的文件夹不存在")