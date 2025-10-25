from importlib.util import set_loader
import os
import csv
import argparse
import torch
import numpy as np
import json
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torchvision import transforms

# 导入CrossViT模型
from models.crossvit import crossvit_18_224  # 默认使用crossvit_18_224模型
from models import create_model  # 从main.py逻辑中获取的模型创建函数

# 允许PIL加载可能被截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ================================================================
# 健壮的Dataset类 (修改自predict.py)
# ================================================================
class PredictDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # 直接读取文件夹中的所有图片文件
        try:
            IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            for img_file in os.listdir(data_dir):
                img_path = os.path.join(data_dir, img_file)
                if os.path.isfile(img_path) and img_file.lower().endswith(IMG_EXTENSIONS):
                    self.image_paths.append(img_path)
            
            if not self.image_paths:
                print(f"警告: 在目录 {data_dir} 中未找到任何有效的图片文件。")
        except Exception as e:
            print(f"严重错误: 无法读取测试目录 {data_dir}。错误: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            filename = os.path.basename(img_path)
            
            with Image.open(img_path) as image:
                image = image.convert('RGB')
            
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # 测试集没有label，返回None作为占位符
            return image_tensor, None, filename
        except Exception as e:
            print(f"警告：处理文件 {self.image_paths[idx]} 时出错。将跳过此文件。错误: {e}")
            return None, None, None


# ================================================================
# 健壮的Collate Function (修改自predict.py)
# ================================================================
def robust_collate_fn(batch):
    predict_batch = [b for b in batch if b[0] is not None]
    if not predict_batch:
        return torch.tensor([]), [], []
    images = torch.stack([b[0] for b in predict_batch])
    # 测试集没有标签，返回空列表
    labels = []
    filenames = [b[2] for b in predict_batch]
    return images, labels, filenames


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="CrossViT 花卉分类测试脚本")
    parser.add_argument("predict_data_dir", type=str, help="测试集图片文件夹的路径")
    parser.add_argument("output_csv_path", type=str, help="保存测试结果CSV的路径")
    parser.add_argument("--model_path", type=str, default="../model/best_model.pth", help="模型权重文件路径")
    parser.add_argument("--model_type", type=str, default="crossvit_18_224", help="模型类型")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像大小")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_classes", type=int, default=100, help="类别数量")
    parser.add_argument("--mapping_path", type=str, default="../model/class_to_idx.json", help="类别映射文件路径")
    parser.add_argument("--amp", action="store_true", help="使用自动混合精度")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 确保模型路径存在
    if not os.path.exists(args.model_path):
        # 尝试从默认位置查找
        default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_path)
        if os.path.exists(default_model_path):
            args.model_path = default_model_path
        else:
            print(f"错误: 模型文件 {args.model_path} 不存在")
            return

    # 加载类别映射
    class_to_idx = None
    idx_to_class = None
    if os.path.exists(args.mapping_path):
        try:
            with open(args.mapping_path, 'r', encoding='utf-8') as f:
                class_to_idx = json.load(f)
            # 创建反向映射
            idx_to_class = {v: int(k) for k, v in class_to_idx.items()}
            print(f"成功加载 {len(class_to_idx)} 个类别映射")
        except Exception as e:
            print(f"警告: 无法加载类别映射文件 {args.mapping_path}, 将尝试从文件夹名称自动生成: {e}")
    
    # 对于测试集，必须从映射文件加载类别信息，否则无法进行预测
    if class_to_idx is None:
        print("错误: 无法加载类别映射文件，无法进行预测。请提供有效的class_to_idx.json文件。")
        return

    # 数据预处理 - 与训练时保持一致
    transform_predict = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    predict_dataset = PredictDataset(data_dir=args.predict_data_dir, transform=transform_predict)
    predict_loader = DataLoader(
        predict_dataset,
        sampler=SequentialSampler(predict_dataset),
        batch_size=args.batch_size,
        num_workers=0,  # Windows环境推荐使用0
        pin_memory=True,
        collate_fn=robust_collate_fn
    )
    print(f"测试集已加载: 找到 {len(predict_dataset)} 张有效图片")

    # 加载CrossViT模型
    try:
        print(f"正在创建模型: {args.model_type}")
        model = create_model(
            args.model_type,
            pretrained=False,  # 我们将手动加载权重
            num_classes=args.num_classes
        )
        
        print(f"正在从 {args.model_path} 加载模型权重...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 处理不同格式的检查点
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("模型权重（从 'model' 键）加载成功")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型权重（从 'model_state_dict' 键）加载成功")
        else:
            # 尝试直接加载，如果模型结构有细微差异则忽略不匹配的键
            try:
                model.load_state_dict(checkpoint)
                print("模型权重直接加载成功")
            except:
                # 尝试忽略不匹配的键
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print("模型权重加载成功（忽略不匹配的键）")
        
        model.to(device)
        model.eval()
        print("模型已成功加载到设备并设为评估模式")

    except Exception as e:
        print(f"严重错误: 加载模型时发生错误: {e}")
        return

    # 运行验证
    predictions = []
    correct_predictions = 0
    total_samples = 0
    
    try:
        with torch.no_grad():           
            for images, labels, filenames in tqdm(predict_loader, desc="正在测试"):
                if not filenames:
                    continue

                images = images.to(device, non_blocking=True)
                
                # 进行推理
                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(images)
                
                # 计算概率和预测类别
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred_indices = torch.max(probabilities, dim=1)
                
                # 保存预测结果
                for i, (filename, pred_idx, conf) in enumerate(zip(filenames, pred_indices, confidence)):
                    # 将预测索引转换为类别ID
                    pred_class_id = idx_to_class[pred_idx.item()]
                    
                    # 保存结果 - 只包含filename、预测的id和confidence
                    predictions.append([
                        filename,
                        pred_class_id,
                        conf.item()
                    ])

    except Exception as e:
        print(f"严重错误: 在测试循环中发生错误: {e}")
        return

    print(f"测试过程完成，共处理 {len(predictions)} 个样本")

    # 保存结果到CSV
    try:
        os.makedirs(os.path.dirname(args.output_csv_path), exist_ok=True)
        headers = ['filename', 'category_id', 'confidence']

        with open(args.output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(predictions)

        print(f"测试完成！结果已保存至: {args.output_csv_path}")
        print(f"总共测试了 {len(predictions)} 张图片")

    except Exception as e:
        print(f"严重错误: 保存结果到 CSV 时出错: {e}")
        return


if __name__ == "__main__":
    main()