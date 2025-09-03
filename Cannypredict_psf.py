import os
import json
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import inception_v3
import numpy as np
from PIL import Image
from scipy import linalg
from Cannymodels_test import CompletionNetwork
from utils import poisson_blend, gen_input_mask
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob
import pandas as pd
from tqdm import tqdm


class FIDCalculator:
    def __init__(self, device='cpu'):
        self.device = device
        # 加载Inception v3模型
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.eval()
        self.model.to(device)

        # 移除辅助分类器
        self.model.aux_logits = False

    def get_activations(self, images):
        """提取Inception特征"""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # 确保图像格式正确 [batch_size, 3, height, width]
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        self.model.eval()
        with torch.no_grad():
            features = self.model(images)
            # 如果特征有空间维度，进行全局平均池化
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1))
            return features.squeeze(-1).squeeze(-1).cpu().numpy()

    def calculate_fid(self, real_images, fake_images):
        """计算FID分数"""
        # 提取特征
        real_activations = self.get_activations(real_images)
        fake_activations = self.get_activations(fake_images)

        # 计算均值和协方差
        mu_real = np.mean(real_activations, axis=0)
        sigma_real = np.cov(real_activations, rowvar=False)

        mu_fake = np.mean(fake_activations, axis=0)
        sigma_fake = np.cov(fake_activations, rowvar=False)

        # 计算Frechet距离
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = (np.dot(diff, diff) +
               np.trace(sigma_real) +
               np.trace(sigma_fake) -
               2 * np.trace(covmean))

        return fid


def preprocess_images_for_fid(images):
    """预处理图像用于FID计算"""
    if isinstance(images, list):
        images = np.array(images)

    # 确保图像范围在[0, 1]之间且格式正确
    images = np.clip(images, 0, 1)

    # 如果图像是灰度图，转换为RGB
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.shape[-1] == 4:  # RGBA
        images = images[..., :3]

    # 调整维度顺序为 [batch, channels, height, width]
    if images.shape[-1] == 3:
        images = images.transpose(0, 3, 1, 2)

    return images


def save_statistics_to_csv(results, fid_value, csv_path):
    """保存统计信息到CSV文件"""
    # 提取PSNR和SSIM值
    psnr_values = [result['psnr'] for result in results]
    ssim_values = [result['ssim'] for result in results]

    # 创建统计信息字典
    stats_data = {
        'metric': ['PSNR', 'SSIM', 'FID'],
        'mean': [
            np.mean(psnr_values) if psnr_values else 0,
            np.mean(ssim_values) if ssim_values else 0,
            fid_value if fid_value is not None else 'N/A'
        ],
        'max': [
            np.max(psnr_values) if psnr_values else 0,
            np.max(ssim_values) if ssim_values else 0,
            'N/A'  # FID没有最大值概念
        ],
        'min': [
            np.min(psnr_values) if psnr_values else 0,
            np.min(ssim_values) if ssim_values else 0,
            'N/A'  # FID没有最小值概念
        ],
        'std': [
            np.std(psnr_values) if psnr_values else 0,
            np.std(ssim_values) if ssim_values else 0,
            'N/A'  # FID没有标准差概念
        ],
        'count': [
            len(psnr_values),
            len(ssim_values),
            len(results) if fid_value is not None else 0
        ]
    }

    # 创建DataFrame并保存
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return stats_df


# ====================== 核心函数：Canny边缘生成 ======================
def gen_canny_edge(x, low_threshold=20, high_threshold=60):
    edges = []
    assert x.shape[1] == 3, f"输入图像必须是3通道(RGB)，实际{x.shape[1]}通道"

    for i, img in enumerate(x):
        try:
            img_detach = img.detach()
            img_np = img_detach.permute(1, 2, 0).cpu().numpy() * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, low_threshold, high_threshold)
            edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0
            edges.append(edge_tensor)

            if i == 0 and edge_tensor.max().item() < 0.01:
                print(f"警告：边缘图最大值={edge_tensor.max().item():.4f}，可能全黑，建议降低阈值")

        except Exception as e:
            print(f"处理第{i}张图像边缘失败: {e}，使用全零边缘图")
            edge_tensor = torch.zeros(1, x.shape[2], x.shape[3]).float()
            edges.append(edge_tensor)

    result_cpu = torch.stack(edges)
    result = result_cpu.to(x.device)
    assert result.shape[1] == 1, f"edge_map通道数错误！期望1，实际{result.shape[1]}"
    return result


# ====================== PSNR计算函数 ======================
def calculate_psnr(original, reconstructed):
    original = original.cpu().numpy().squeeze().transpose(1, 2, 0)
    reconstructed = reconstructed.cpu().numpy().squeeze().transpose(1, 2, 0)
    return psnr(original, reconstructed, data_range=1.0)


# ====================== SSIM计算函数 ======================
def calculate_ssim(original, reconstructed):
    """计算SSIM"""
    original = original.cpu().numpy().squeeze().transpose(1, 2, 0)
    reconstructed = reconstructed.cpu().numpy().squeeze().transpose(1, 2, 0)

    # 转换为灰度图像计算SSIM
    if original.shape[2] == 3:
        original_gray = np.dot(original[..., :3], [0.2989, 0.5870, 0.1140])
        reconstructed_gray = np.dot(reconstructed[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        original_gray = original
        reconstructed_gray = reconstructed

    return ssim(original_gray, reconstructed_gray, data_range=1.0)


# ====================== 核心函数：单图处理 ======================
def process_single_image(model, mpv, img_path, output_dir, params, device):
    try:
        # 加载并预处理图片
        img = Image.open(img_path).convert('RGB')
        img = transforms.Resize(params['img_size'])(img)
        img = transforms.RandomCrop((params['img_size'], params['img_size']))(img)
        x = transforms.ToTensor()(img)
        x = torch.unsqueeze(x, dim=0).to(device)  # 直接移到目标设备

        # 生成掩码并移到目标设备
        mask = gen_input_mask(
            shape=(1, 1, x.shape[2], x.shape[3]),
            hole_size=((params['hole_min_w'], params['hole_max_w']),
                       (params['hole_min_h'], params['hole_max_h'])),
            max_holes=params['max_holes']
        ).to(device)  # 生成后立即移到目标设备

        # 生成Canny边缘图（已在函数内确保与x同设备）
        edge_map = gen_canny_edge(x,
                                  low_threshold=params['canny_low'],
                                  high_threshold=params['canny_high'])

        # 图像补全
        model.eval()
        with torch.no_grad():
            x_mask = x - x * mask + mpv * mask

            # 构建5通道输入
            input_main = torch.cat([x_mask, mask], dim=1)
            input_with_edge = torch.cat([input_main, edge_map], dim=1)

            output = model(input_with_edge)
            inpainted = poisson_blend(x_mask, output, mask)

            # 计算PSNR和SSIM
            psnr_value = calculate_psnr(x.cpu(), inpainted.cpu())
            ssim_value = calculate_ssim(x.cpu(), inpainted.cpu())

            # 保存结果图片
            if params['save_results']:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(img_path)
                output_img_path = os.path.join(output_dir, filename)

                # 统一移到CPU再拼接（保存图片无需GPU加速）
                x_cpu = x.cpu()
                x_mask_cpu = x_mask.cpu()
                edge_vis_cpu = edge_map.expand(-1, 3, -1, -1).cpu()  # 边缘图转为3通道并移到CPU
                inpainted_cpu = inpainted.cpu()

                # 现在所有张量都在CPU，拼接无设备冲突
                imgs = torch.cat((x_cpu, x_mask_cpu, edge_vis_cpu, inpainted_cpu), dim=0)
                save_image(imgs, output_img_path, nrow=4)

        # 返回图像数据用于FID计算
        original_np = x.cpu().numpy().squeeze(0)  # [C, H, W]
        reconstructed_np = inpainted.cpu().numpy().squeeze(0)  # [C, H, W]

        return os.path.basename(img_path), psnr_value, ssim_value, original_np, reconstructed_np

    except Exception as e:
        print(f"处理图片 {os.path.basename(img_path)} 出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


# ====================== 主函数 ======================
def main():
    config = {
        'model_path': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/results/test/phase_3/model_cn_step90000',
        'config_path': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/results/test/config.json',
        'test_set_dir': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/datasets/celeba_hq_256/test',
        'output_dir': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/result_predict/canny_psf/img_result',
        'result_csv_path': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/result_predict/canny_psf/metrics.csv',
        'stats_csv_path': 'D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/result_predict/canny_psf/statistics.csv',
        'img_size': 160,
        'max_holes': 5,
        'hole_min_w': 24,
        'hole_max_w': 48,
        'hole_min_h': 24,
        'hole_max_h': 48,
        'canny_low': 20,
        'canny_high': 60,
        'save_results': True,
        'calculate_fid': True
    }

    # 路径验证
    for key in ['model_path', 'config_path', 'test_set_dir']:
        path = os.path.expanduser(config[key])
        if not os.path.exists(path):
            print(f"错误: 路径不存在 - {path}")
            return
    for key in ['model_path', 'config_path', 'test_set_dir', 'output_dir', 'result_csv_path', 'stats_csv_path']:
        config[key] = os.path.expanduser(config[key])

    # 加载模型和配置
    with open(config['config_path'], 'r') as f:
        train_config = json.load(f)
    mpv = torch.tensor(train_config['mpv']).view(1, 3, 1, 1)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    mpv = mpv.to(device)

    # 初始化FID计算器
    if config['calculate_fid']:
        fid_calculator = FIDCalculator(device='cpu')

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
    results = []
    successful_count = 0
    original_imgs_for_fid = []
    reconstructed_imgs_for_fid = []

    for img_path in tqdm(image_paths, desc="处理图片"):
        filename, psnr_val, ssim_val, original_np, reconstructed_np = process_single_image(
            model=model,
            mpv=mpv,
            img_path=img_path,
            output_dir=config['output_dir'],
            params=config,
            device=device
        )
        if filename and psnr_val is not None and ssim_val is not None:
            results.append({
                'filename': filename,
                'psnr': round(psnr_val, 4),
                'ssim': round(ssim_val, 4)
            })
            successful_count += 1

            # 收集FID计算数据
            if config['calculate_fid'] and original_np is not None and reconstructed_np is not None:
                original_imgs_for_fid.append(original_np)
                reconstructed_imgs_for_fid.append(reconstructed_np)

    # =============================================
    # 计算FID分数
    # =============================================
    fid_value = None
    if config['calculate_fid'] and original_imgs_for_fid and reconstructed_imgs_for_fid:
        try:
            print("计算FID分数...")

            # 预处理图像
            original_imgs_processed = preprocess_images_for_fid(original_imgs_for_fid)
            reconstructed_imgs_processed = preprocess_images_for_fid(reconstructed_imgs_for_fid)

            # 计算FID
            fid_value = fid_calculator.calculate_fid(original_imgs_processed, reconstructed_imgs_processed)
            print(f"FID分数: {fid_value:.4f}")

        except Exception as e:
            print(f"计算FID时出错: {e}")
            import traceback
            traceback.print_exc()
            fid_value = None

    # 保存结果
    if results:
        # 保存详细结果到CSV
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(config['result_csv_path']), exist_ok=True)
        df.to_csv(config['result_csv_path'], index=False, encoding='utf-8-sig')

        # 保存统计信息到CSV
        stats_df = save_statistics_to_csv(results, fid_value, config['stats_csv_path'])

        # 计算统计信息用于显示
        psnr_values = [result['psnr'] for result in results]
        ssim_values = [result['ssim'] for result in results]

        print(f"\n处理完成！成功处理 {successful_count}/{len(image_paths)} 张图片")
        print("\nPSNR统计:")
        print(f"  平均PSNR: {np.mean(psnr_values):.4f}")
        print(f"  最高PSNR: {np.max(psnr_values):.4f}")
        print(f"  最低PSNR: {np.min(psnr_values):.4f}")
        print(f"  PSNR标准差: {np.std(psnr_values):.4f}")

        print("\nSSIM统计:")
        print(f"  平均SSIM: {np.mean(ssim_values):.4f}")
        print(f"  最高SSIM: {np.max(ssim_values):.4f}")
        print(f"  最低SSIM: {np.min(ssim_values):.4f}")
        print(f"  SSIM标准差: {np.std(ssim_values):.4f}")

        if fid_value is not None:
            print(f"\nFID分数: {fid_value:.4f}")
            print("注: FID分数越低越好，表示生成图像与真实图像分布越接近")

        print(f"\n详细结果已保存到: {config['result_csv_path']}")
        print(f"统计信息已保存到: {config['stats_csv_path']}")

        if config['save_results']:
            print(f"处理后的图片已保存到: {config['output_dir']}")

        # 在控制台显示统计表格
        print("\n统计信息表格:")
        print(stats_df.to_string(index=False))

    else:
        print("没有成功处理任何图片")


if __name__ == '__main__':
    main()