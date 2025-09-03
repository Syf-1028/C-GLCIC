import os
import json
import torch
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from Sobelmodels_test import CompletionNetwork
from utils import poisson_blend
import glob
import pandas as pd
from tqdm import tqdm


# ====================== 核心函数：Sobel边缘生成 ======================
def gen_sobel_edge(x, ksize=3, gaussian_sigma=1.0, low_threshold=50, high_threshold=150):
    """
    生成与Canny边缘图行为一致的Sobel边缘图（输入x为[B,C,H,W]的tensor，输出[B,1,H,W]的tensor）
    关键改进：添加高斯滤波去噪 + 双阈值二值化，对齐Canny的边缘质量
    参数说明：
        ksize: Sobel核大小（必须是1,3,5,7，默认3，与Canny常用配置匹配）
        gaussian_sigma: 高斯滤波标准差（默认1.0，与Canny去噪逻辑一致）
        low_threshold/high_threshold: 双阈值（默认50/150，直接复用Canny的阈值参数，确保筛选标准一致）
    """
    edges = []
    assert x.shape[1] == 3, f"输入图像必须是3通道(RGB)，实际{x.shape[1]}通道"
    assert ksize in (1, 3, 5, 7), f"Sobel核大小必须是1,3,5,7，实际{ksize}"

    for i, img in enumerate(x):
        try:
            # 1. 切断梯度 + 格式转换（与Canny完全对齐：detach→HWC→0-255→clip）
            img_detach = img.detach()  # 不影响原Tensor梯度
            img_np = img_detach.permute(1, 2, 0).cpu().numpy() * 255  # CHW→HWC，映射到像素范围
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # 防止数值溢出，确保cv2兼容

            # 2. 高斯滤波去噪（Canny的核心预处理步骤，Sobel必须补充，否则噪声严重）
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # 转灰度图（与Canny一致）
            gray_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=gaussian_sigma)  # 5×5核去噪（Canny默认逻辑）

            # 3. Sobel梯度计算（保留原有逻辑，但基于去噪后的灰度图）
            sobel_x = cv2.Sobel(gray_blur, cv2.CV_64F, dx=1, dy=0, ksize=ksize)  # x方向梯度
            sobel_y = cv2.Sobel(gray_blur, cv2.CV_64F, dx=0, dy=1, ksize=ksize)  # y方向梯度
            sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)  # 合并梯度幅值（边缘强度")

            # 4. 双阈值二值化（关键！对齐Canny的边缘筛选逻辑，去除弱边缘噪声）
            # 步骤1：将梯度幅值映射到0-255（确保阈值筛选范围与Canny一致）
            sobel_magnitude_norm = cv2.normalize(
                sobel_magnitude, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            # 步骤2：双阈值筛选（强边缘保留，弱边缘仅在与强边缘连通时保留，与Canny非极大值抑制逻辑等效）
            _, edge_binary = cv2.threshold(sobel_magnitude_norm, low_threshold, 255, cv2.THRESH_BINARY)  # 基础二值化
            # （可选）进一步优化：用形态学闭运算填补边缘间隙，与Canny边缘的连续性对齐
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edge_binary = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel)

            # 5. Tensor转换 + 归一化（与Canny完全对齐：0-1范围，单通道）
            edge_tensor = torch.from_numpy(edge_binary).unsqueeze(0).float() / 255.0  # 增加通道维度，归一化到0-1
            edges.append(edge_tensor)

        except Exception as e:
            # 异常处理（与Canny一致，确保批次生成不中断）
            print(f"处理第{i}张图像失败: {e}，使用全零边缘图")
            edge_tensor = torch.zeros(1, x.shape[2], x.shape[3]).float()
            edges.append(edge_tensor)

    # 最终格式对齐（CPU堆叠避免GPU碎片化，强制单通道）
    result_cpu = torch.stack(edges)
    result = result_cpu.to(x.device)
    assert result.shape[1] == 1, f"edge_map通道数错误！期望1，实际{result.shape[1]}"
    return result


# ====================== 核心函数：单图处理 ======================
def process_single_image(model, mpv, img_path, mask_dir, output_dir, params, device):
    try:
        # 加载并预处理图片
        img = Image.open(img_path).convert('RGB')
        img = transforms.Resize(params['img_size'])(img)
        img = transforms.RandomCrop((params['img_size'], params['img_size']))(img)
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0).to(device)  # 直接移到目标设备

        # 加载对应的mask文件
        filename = os.path.basename(img_path)
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            print(f"警告: 找不到对应的mask文件 {mask_path}，跳过此图片")
            return None

        # 加载mask图片
        mask_img = Image.open(mask_path).convert('L')
        mask_img = transforms.Resize(params['img_size'])(mask_img)
        mask_img = transforms.RandomCrop((params['img_size'], params['img_size']))(mask_img)
        mask = transforms.ToTensor()(mask_img)
        mask = torch.unsqueeze(mask, dim=0).to(device)

        # 确保mask是二值的
        mask = (mask > 0.5).float()

        # 生成Sobel边缘图（已在函数内确保与x同设备）
        edge_map = gen_sobel_edge(x, ksize=3)

        # 图像补全
        model.eval()
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask

            # 构建5通道输入
            input_main = torch.cat([x_mask, mask], dim=1)
            input_with_edge = torch.cat([input_main, edge_map], dim=1)

            output = model(input_with_edge)
            inpainted = poisson_blend(x_mask, output, mask)

            # # 保存结果图片
            # if params['save_results']:
            #     os.makedirs(output_dir, exist_ok=True)
            #     filename = os.path.basename(img_path)
            #     output_img_path = os.path.join(output_dir, filename)
            #
            #     # 统一移到CPU再拼接（保存图片无需GPU加速）
            #     x_cpu = x.cpu()
            #     x_mask_cpu = x_mask.cpu()
            #     edge_vis_cpu = edge_map.expand(-1, 3, -1, -1).cpu()  # 边缘图转为3通道并移到CPU
            #     inpainted_cpu = inpainted.cpu()
            #
            #     # 现在所有张量都在CPU，拼接无设备冲突
            #     imgs = torch.cat((x_cpu, x_mask_cpu, edge_vis_cpu, inpainted_cpu), dim=0)
            #     save_image(imgs, output_img_path, nrow=4)
            if params['save_results']:
                os.makedirs(output_dir, exist_ok=True)
                base_filename = os.path.splitext(filename)[0]

                # 保存边缘图
                edge_output_path = os.path.join(output_dir, f"{base_filename}_edge.png")
                save_image(edge_map, edge_output_path)

                # 保存修补后的图像
                inpainted_output_path = os.path.join(output_dir, f"{base_filename}_inpainted.png")
                save_image(inpainted, inpainted_output_path)

            return os.path.basename(img_path)

        return os.path.basename(img_path)

    except Exception as e:
        print(f"处理图片 {os.path.basename(img_path)} 出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# ====================== 主函数 ======================
def main():
    config = {
        'model_path': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/results/new_sobel/phase_3/model_cn_step90000',
        'config_path': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/results/new_sobel/config.json',
        'test_set_dir': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/images/mask_realimage',
        'mask_dir': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/images/mask_image',
        # 新增mask文件夹路径
        'output_dir': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/paper_pic/s-glcic/only_e_i',
        'img_size': 160,
        'save_results': True
    }

    # 路径验证
    for key in ['model_path', 'config_path', 'test_set_dir', 'mask_dir']:
        path = os.path.expanduser(config[key])
        if not os.path.exists(path):
            print(f"错误: 路径不存在 - {path}")
            return
    for key in ['model_path', 'config_path', 'test_set_dir', 'mask_dir', 'output_dir']:
        config[key] = os.path.expanduser(config[key])

    # 加载模型和配置
    with open(config['config_path'], 'r') as f:
        train_config = json.load(f)
    mpv = torch.tensor(train_config['mpv']).view(1, 3, 1, 1)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    mpv = mpv.to(device)

    # 加载模型
    model = CompletionNetwork(input_channels=5)
    state_dict = torch.load(config['model_path'], map_location='cpu')
    if 'conv1.weight' in state_dict:
        weight_shape = state_dict['conv1.weight'].shape
        print(f"预训练权重形状: {weight_shape}")
        if weight_shape[1] == 5:
            print("裁剪5通道预训练权重到4通道...")
            state_dict['conv1.weight'] = state_dict['conv1.weight'][:, :4, :, :]
        elif weight_shape[1] == 4:
            print("预训练权重已经是4通道，无需修改")
        else:
            raise ValueError(f"不支持的权重通道数: {weight_shape[1]}")
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 获取测试集图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(config['test_set_dir'], ext)))
    if not image_paths:
        print(f"在目录 {config['test_set_dir']} 中未找到图片")
        return
    print(f"找到 {len(image_paths)} 张图片，开始处理...")

    # 批量处理图片
    successful_count = 0

    for img_path in tqdm(image_paths, desc="处理图片"):
        filename = process_single_image(
            model=model,
            mpv=mpv,
            img_path=img_path,
            mask_dir=config['mask_dir'],
            output_dir=config['output_dir'],
            params=config,
            device=device
        )
        if filename:
            successful_count += 1

    # 输出处理结果
    print(f"\n处理完成！成功处理 {successful_count}/{len(image_paths)} 张图片")

    if config['save_results']:
        print(f"处理后的图片已保存到: {config['output_dir']}")


if __name__ == '__main__':
    main()