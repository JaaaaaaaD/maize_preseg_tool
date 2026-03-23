# SAM模型封装

import torch
from segment_anything import sam_model_registry, SamPredictor

class SamModel:
    def __init__(self, model_type="vit_b", device=None):
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self.model_loaded = False
    
    def load_model(self, model_path):
        """加载SAM模型"""
        try:
            sam = sam_model_registry[self.model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"加载SAM模型失败: {str(e)}")
            return False
    
    def set_image(self, image):
        """设置当前图像"""
        if not self.model_loaded:
            return False
        try:
            self.predictor.set_image(image)
            return True
        except Exception as e:
            print(f"设置图像失败: {str(e)}")
            return False
    
    def predict(self, point_coords, point_labels):
        """预测分割结果"""
        if not self.model_loaded:
            return None
        try:
            masks, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            return masks
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return None
    
    def is_loaded(self):
        """检查模型是否已加载"""
        return self.model_loaded
    
    def get_device(self):
        """获取当前设备"""
        return self.device