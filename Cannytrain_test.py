'''
    此文件执行c-GLCIC图像补全的训练任务，共三个阶段
    Modified to include Canny edge operator
'''

import json
import os
import argparse
import math

from torch.utils.data import DataLoader
from torch.optim import Adadelta
from torch.nn import BCELoss, DataParallel, L1Loss
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
import cv2

from Canny_glcic.Cannymodels_test import CompletionNetwork, ContextDiscriminator
from datasets import ImageDataset
from losses import completion_network_loss

from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)
# 新增PSNR计算函数
def calculate_psnr(img1, img2):
    """
    计算两幅图像之间的PSNR值
    img1, img2: 范围在[0, 1]的张量
    """
    # 确保两个张量在同一设备上
    if img1.device != img2.device:
        img2 = img2.to(img1.device)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


# 修改record_psnr函数，确保处理CPU上的数值
def record_psnr(result_dir, step, psnr_value, phase):
    """
    将PSNR值记录到文件中
    """
    psnr_file = os.path.join(result_dir, 'psnr.txt')
    # 如果psnr_value是张量，转换为Python数值
    if torch.is_tensor(psnr_value):
        psnr_value = psnr_value.item() if psnr_value.dim() == 0 else float(psnr_value.cpu())
    with open(psnr_file, 'a') as f:
        f.write(f'{phase}, step {step}: PSNR = {psnr_value:.4f} dB\n')
'''
    设置默认路径
'''
DEFAULT_DATA_DIR = "D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/datasets/celeba_hq_256"  # 默认数据集路径
DEFAULT_RESULT_DIR = "D:/PycharmProjects/GLCIC-PyTorch-master/GLCIC-PyTorch-master/results/test"  # 默认结果保存路径
# 固定Canny边缘检测阈值（直接在代码中设置，无需命令行输入）
CANNY_LOW_THRESHOLD = 20    # 降低低阈值，保留更多弱边缘
CANNY_HIGH_THRESHOLD = 60   # 降低高阈值，适合人脸等低对比度图像
# 设置命令行参数
parser = argparse.ArgumentParser()
# parser.add_argument('data_dir')         # 输入数据目录 必须有
# parser.add_argument('result_dir')       # 输出结果保存目录 必须有
parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                   help='输入数据目录 (默认: %(default)s)')
parser.add_argument('--result_dir', type=str, default=DEFAULT_RESULT_DIR,
                   help='输出结果保存目录 (默认: %(default)s)')
parser.add_argument('--data_parallel', action='store_true')     # 并行处理
parser.add_argument('--recursive_search', action='store_true', default=False)       # 递归搜索数据目录
parser.add_argument('--init_model_cn', type=str, default=None)      # 模型初始化   生成网络预训练权重路径
parser.add_argument('--init_model_cd', type=str, default=None)      # 判别器预训练权重
parser.add_argument('--steps_1', type=int, default=50000)           # 三个训练阶段的步数  default控制代码迭代次数 仅训练生成网络
parser.add_argument('--steps_2', type=int, default=5500)           # 仅训练判别器
parser.add_argument('--steps_3', type=int, default=90000)          # 联合训练
parser.add_argument('--snaperiod_1', type=int, default=10000)       # 三个阶段的快照周期  default表示快照保存的间隔步数
parser.add_argument('--snaperiod_2', type=int, default=1100)
parser.add_argument('--snaperiod_3', type=int, default=10000)
parser.add_argument('--max_holes', type=int, default=3)             # 每张图最大孔洞数
parser.add_argument('--hole_min_w', type=int, default=48)           # 孔洞最小宽度
parser.add_argument('--hole_max_w', type=int, default=96)           # 孔洞最大宽度
parser.add_argument('--hole_min_h', type=int, default=48)           # 孔洞最小高度
parser.add_argument('--hole_max_h', type=int, default=96)           # 孔洞最大高度
parser.add_argument('--cn_input_size', type=int, default=160)       # 生成网络输入尺寸
parser.add_argument('--ld_input_size', type=int, default=96)        # 局部判别器输入尺寸
parser.add_argument('--bsize', type=int, default=16)                # 批处理参数
parser.add_argument('--bdivs', type=int, default=1)                 # 批次分割数，梯度累积
parser.add_argument('--num_test_completions', type=int, default=16)         # 测试时生成的样本数
parser.add_argument('--mpv', nargs=3, type=float, default=None)         # RGB=3
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--arc', type=str, choices=['celeba', 'places2'], default='celeba')     # 模型架构选择
parser.add_argument('--canny_low', type=int, default=50)            # Canny低阈值
parser.add_argument('--canny_high', type=int, default=150)          # Canny高阈值
parser.add_argument('--edge_weight', type=float, default=0.1)       # 边缘损失权重


def gen_canny_edge(x, low_threshold=50, high_threshold=150):
    """生成Canny边缘图（输入x为[B,C,H,W]的tensor，输出[B,1,H,W]的tensor）"""
    edges = []
    assert x.shape[1] == 3, f"输入图像必须是3通道(RGB)，实际{x.shape[1]}通道"
    for i, img in enumerate(x):
        try:
            # ========== 关键修改：用detach()切断梯度，再转numpy ==========
            # detach()：不影响原Tensor的梯度计算，仅复制一份无梯度的Tensor
            img_detach = img.detach()  # 切断梯度关联
            # 转HWC格式 + 映射到0-255（图像像素范围）
            img_np = img_detach.permute(1, 2, 0).cpu().numpy() * 255
            # 防止数值溢出（确保像素值在0-255之间，避免cv2报错）
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # 生成真实Canny边缘图（不再走全零fallback）
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, low_threshold, high_threshold)
            # 转Tensor并归一化到0-1（与模型输入范围匹配）
            edge_tensor = torch.from_numpy(edge).unsqueeze(0).float() / 255.0
            edges.append(edge_tensor)
        except Exception as e:
            print(f"处理第{i}张图像失败: {e}，使用全零边缘图")
            # 异常时生成全零边缘图，确保形状正确
            edge_tensor = torch.zeros(1, x.shape[2], x.shape[3]).float()
            edges.append(edge_tensor)

    # 堆叠并归一化（强制在CPU堆叠，避免GPU内存碎片化导致的形状异常）
    result_cpu = torch.stack(edges)
    # 转移到GPU并强制通道数为1
    result = result_cpu.to(x.device)
    # 关键断言：确保边缘图必为1通道
    assert result.shape[1] == 1, f"edge_map通道数错误！期望1，实际{result.shape[1]}"
    return result


def main(args):
    # ================================================
    # Preparation 检查GPU的可用性
    # ================================================
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    gpu = torch.device('cuda:0')

    # 创建PSNR记录文件
    psnr_file = os.path.join(args.result_dir, 'psnr.txt')
    with open(psnr_file, 'w') as f:
        f.write('Step,Phase,PSNR\n')


    # create result directory (if necessary)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))

    # load dataset
    trnsfm = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset... (it may take a few minutes)')
    train_dset = ImageDataset(
        os.path.join(args.data_dir, 'train'),
        trnsfm,
        recursive_search=args.recursive_search)
    test_dset = ImageDataset(
        os.path.join(args.data_dir, 'test'),
        trnsfm,
        recursive_search=args.recursive_search)
    train_loader = DataLoader(
        train_dset,
        batch_size=(args.bsize // args.bdivs),
        shuffle=True)

    # compute mpv (mean pixel value) of training dataset
    if args.mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(
            total=len(train_dset.imgpaths),
            desc='computing mean pixel value of training dataset...')
        for imgpath in train_dset.imgpaths:
            img = Image.open(imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(train_dset.imgpaths)
        pbar.close()
    else:
        mpv = np.array(args.mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))
    args_dict = vars(args)
    args_dict['mpv'] = mpv_json
    with open(os.path.join(
            args.result_dir, '../config.json'),
            mode='w') as f:
        json.dump(args_dict, f)

    # make mpv & alpha tensors
    mpv = torch.tensor(
        mpv.reshape(1, 3, 1, 1),
        dtype=torch.float32).to(gpu)
    alpha = torch.tensor(
        args.alpha,
        dtype=torch.float32).to(gpu)

    # ================================================
    # Training Phase 1
    # ================================================
    # load completion network (修改为5通道输入)
    model_cn = CompletionNetwork(input_channels=5)  # 需要修改models.py中的CompletionNetwork类
    if args.init_model_cn is not None:
        # 处理预训练权重通道不匹配问题
        state_dict = torch.load(args.init_model_cn, map_location='cpu')
        if 'conv1.weight' in state_dict:
            old_weight = state_dict['conv1.weight']
            new_weight = torch.zeros(64, 5, 5, 5)  # 新的5通道输入
            new_weight[:, :4, :, :] = old_weight  # 复制原有4通道权重
            state_dict['conv1.weight'] = new_weight
        model_cn.load_state_dict(state_dict)

    if args.data_parallel:
        model_cn = DataParallel(model_cn)
    model_cn = model_cn.to(gpu)
    opt_cn = Adadelta(model_cn.parameters())
    l1_loss = L1Loss()  # 用于边缘损失

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        for x in train_loader:

            # forward
            x = x.to(gpu)
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (x.shape[3], x.shape[2])),
                max_holes=args.max_holes,
            ).to(gpu)

            edge_map = gen_canny_edge(x, args.canny_low, args.canny_high)  # 生成边缘图

            x_mask = x - x * mask + mpv * mask

            input = torch.cat((x_mask, mask, edge_map), dim=1)  # 5通道输入

            output = model_cn(input)

            # 计算损失
            loss_pixel = completion_network_loss(x, output, mask)
            output_edge = gen_canny_edge(output, args.canny_low, args.canny_high)
            loss_edge = l1_loss(output_edge, edge_map)
            loss = loss_pixel + args.edge_weight * loss_edge

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description('phase 1 | train loss: %.5f (pixel: %.5f edge: %.5f)' %
                                    (loss.cpu(), loss_pixel.cpu(), loss_edge.cpu()))
                pbar.update()

                # test
                if pbar.n % args.snaperiod_1 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        x = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions).to(gpu)
                        mask = gen_input_mask(
                            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                            hole_size=(
                                (args.hole_min_w, args.hole_max_w),
                                (args.hole_min_h, args.hole_max_h)),
                            hole_area=gen_hole_area(
                                (args.ld_input_size, args.ld_input_size),
                                (x.shape[3], x.shape[2])),
                            max_holes=args.max_holes).to(gpu)
                        edge_map = gen_canny_edge(x, args.canny_low, args.canny_high)
                        x_mask = x - x * mask + mpv * mask
                        input = torch.cat((x_mask, mask, edge_map), dim=1)
                        output = model_cn(input)
                        completed = poisson_blend(x_mask, output, mask)

                        # [新增] 计算PSNR
                        psnr_value = calculate_psnr(x, completed)
                        print(f'Phase 1 - Step {pbar.n}: PSNR = {psnr_value:.4f} dB')
                        record_psnr(args.result_dir, pbar.n, psnr_value, 'phase_1')

                        # 保存结果（包括边缘图）
                        edge_vis = edge_map.expand(-1, 3, -1, -1)  # 转为3通道便于可视化
                        edge_vis = torch.clamp(edge_vis * 1.5, 0, 1)  # 增强边缘亮度
                        imgs = torch.cat((
                            x.cpu(),
                            x_mask.cpu(),
                            edge_vis.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_1',
                            'step%d.png' % pbar.n)
                        model_cn_path = os.path.join(
                            args.result_dir,
                            'phase_1',
                            'model_cn_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        if args.data_parallel:
                            torch.save(
                                model_cn.module.state_dict(),
                                model_cn_path)
                        else:
                            torch.save(
                                model_cn.state_dict(),
                                model_cn_path)
                    model_cn.train()
                if pbar.n >= args.steps_1:
                    break

    pbar.close()

    # ================================================
    # Training Phase 2  (类似修改，保持输入一致性)
    # ================================================
    # 注意：Phase 2和3需要保持相同的输入通道修改
    # Phase 2和3的代码结构与原版相同，只需确保：
    # 1. 输入使用5通道（RGB+Mask+Edge）
    # 2. 判别器输入可能需要相应调整（如果使用边缘信息）
    # ================================================
    # Training Phase 2
    # ================================================
    # load context discriminator
    model_cd = ContextDiscriminator(
        local_input_shape=(3, args.ld_input_size, args.ld_input_size),
        global_input_shape=(3, args.cn_input_size, args.cn_input_size),
        arc=args.arc)
    if args.init_model_cd is not None:
        model_cd.load_state_dict(torch.load(
            args.init_model_cd,
            map_location='cpu'))
    if args.data_parallel:
        model_cd = DataParallel(model_cd)
    model_cd = model_cd.to(gpu)
    opt_cd = Adadelta(model_cd.parameters())
    bceloss = BCELoss()

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2:
        for x in train_loader:
            # fake forward (使用5通道输入)
            x = x.to(gpu)
            hole_area_fake = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=hole_area_fake,
                max_holes=args.max_holes).to(gpu)
            edge_map = gen_canny_edge(x, args.canny_low, args.canny_high)  # 新增边缘图
            fake = torch.zeros((len(x), 1)).to(gpu)
            x_mask = x - x * mask + mpv * mask
            input_cn = torch.cat((x_mask, mask, edge_map), dim=1)  # 5通道输入
            output_cn = model_cn(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = model_cd((
                input_ld_fake.to(gpu),
                input_gd_fake.to(gpu)))
            loss_fake = bceloss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            real = torch.ones((len(x), 1)).to(gpu)
            input_gd_real = x
            input_ld_real = crop(input_gd_real, hole_area_real)
            output_real = model_cd((input_ld_real, input_gd_real))
            loss_real = bceloss(output_real, real)

            # reduce
            loss = (loss_fake + loss_real) / 2.

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cd.step()
                opt_cd.zero_grad()
                pbar.set_description('phase 2 | train loss: %.5f' % loss.cpu())
                pbar.update()

                # test
                if pbar.n % args.snaperiod_2 == 0:
                    model_cn.eval()
                    model_cd.eval()
                    with torch.no_grad():
                        x = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions).to(gpu)
                        mask = gen_input_mask(
                            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                            hole_size=(
                                (args.hole_min_w, args.hole_max_w),
                                (args.hole_min_h, args.hole_max_h)),
                            hole_area=gen_hole_area(
                                (args.ld_input_size, args.ld_input_size),
                                (x.shape[3], x.shape[2])),
                            max_holes=args.max_holes).to(gpu)
                        edge_map = gen_canny_edge(x, args.canny_low, args.canny_high)
                        x_mask = x - x * mask + mpv * mask
                        input = torch.cat((x_mask, mask, edge_map), dim=1)
                        output = model_cn(input)
                        completed = poisson_blend(x_mask, output, mask)

                        # [新增] 计算PSNR
                        psnr_value = calculate_psnr(x, completed)
                        print(f'Phase 2 - Step {pbar.n}: PSNR = {psnr_value:.4f} dB')
                        record_psnr(args.result_dir, pbar.n, psnr_value, 'phase_2')

                        # 保存结果（包含边缘图）
                        edge_vis = edge_map.expand(-1, 3, -1, -1)
                        edge_vis = torch.clamp(edge_vis * 1.5, 0, 1)  # 增强边缘亮度
                        imgs = torch.cat((
                            x.cpu(),
                            x_mask.cpu(),
                            edge_vis.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'step%d.png' % pbar.n)
                        model_cn_path = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'model_cn_step%d' % pbar.n)
                        model_cd_path = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        if args.data_parallel:
                            torch.save(
                                model_cn.module.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.module.state_dict(),
                                model_cd_path)
                        else:
                            torch.save(
                                model_cn.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.state_dict(),
                                model_cd_path)
                    model_cn.train()
                    model_cd.train()
                if pbar.n >= args.steps_2:
                    break

    pbar.close()
# ================================================
# Training Phase 3 (类似修改，保持输入一致性)
# ================================================
    # ================================================
    # Training Phase 3
    # ================================================
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_3)
    while pbar.n < args.steps_3:
        for x in train_loader:
            # forward model_cd (使用5通道输入)
            x = x.to(gpu)
            hole_area_fake = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=hole_area_fake,
                max_holes=args.max_holes).to(gpu)
            edge_map = gen_canny_edge(x, args.canny_low, args.canny_high)  # 新增边缘图

            # fake forward
            fake = torch.zeros((len(x), 1)).to(gpu)
            x_mask = x - x * mask + mpv * mask
            input_cn = torch.cat((x_mask, mask, edge_map), dim=1)  # 5通道输入
            output_cn = model_cn(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = model_cd((input_ld_fake, input_gd_fake))
            loss_cd_fake = bceloss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            real = torch.ones((len(x), 1)).to(gpu)
            input_gd_real = x
            input_ld_real = crop(input_gd_real, hole_area_real)
            output_real = model_cd((input_ld_real, input_gd_real))
            loss_cd_real = bceloss(output_real, real)

            # reduce
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.

            # backward model_cd
            loss_cd.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                # optimize
                opt_cd.step()
                opt_cd.zero_grad()

            # forward model_cn (带边缘损失)
            loss_cn_1 = completion_network_loss(x, output_cn, mask)
            output_edge = gen_canny_edge(output_cn, args.canny_low, args.canny_high)
            loss_cn_edge = l1_loss(output_edge, edge_map)  # 边缘一致性损失

            input_gd_fake = output_cn
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = model_cd((input_ld_fake, (input_gd_fake)))
            loss_cn_2 = bceloss(output_fake, real)

            # reduce
            loss_cn = (loss_cn_1 + args.edge_weight * loss_cn_edge + alpha * loss_cn_2) / 3.

            # backward model_cn
            loss_cn.backward()
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description(
                    'phase 3 | train loss (cd): %.5f (cn): %.5f (edge: %.5f)' % (
                        loss_cd.cpu(),
                        loss_cn.cpu(),
                        loss_cn_edge.cpu()))
                pbar.update()

                # test
                if pbar.n % args.snaperiod_3 == 0:
                    model_cn.eval()
                    model_cd.eval()
                    with torch.no_grad():
                        x = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions).to(gpu)
                        mask = gen_input_mask(
                            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                            hole_size=(
                                (args.hole_min_w, args.hole_max_w),
                                (args.hole_min_h, args.hole_max_h)),
                            hole_area=gen_hole_area(
                                (args.ld_input_size, args.ld_input_size),
                                (x.shape[3], x.shape[2])),
                            max_holes=args.max_holes).to(gpu)
                        edge_map = gen_canny_edge(x, args.canny_low, args.canny_high)
                        x_mask = x - x * mask + mpv * mask
                        input = torch.cat((x_mask, mask, edge_map), dim=1)
                        output = model_cn(input)
                        completed = poisson_blend(x_mask, output, mask)

                        # [新增] 计算PSNR
                        psnr_value = calculate_psnr(x, completed)
                        print(f'Phase 3 - Step {pbar.n}: PSNR = {psnr_value:.4f} dB')
                        record_psnr(args.result_dir, pbar.n, psnr_value, 'phase_3')

                        # 保存结果（包含边缘图）
                        edge_vis = edge_map.expand(-1, 3, -1, -1)
                        edge_vis = torch.clamp(edge_vis * 1.5, 0, 1)
                        imgs = torch.cat((
                            x.cpu(),
                            x_mask.cpu(),
                            edge_vis.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'step%d.png' % pbar.n)
                        model_cn_path = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'model_cn_step%d' % pbar.n)
                        model_cd_path = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        if args.data_parallel:
                            torch.save(
                                model_cn.module.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.module.state_dict(),
                                model_cd_path)
                        else:
                            torch.save(
                                model_cn.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.state_dict(),
                                model_cd_path)
                    model_cn.train()
                    model_cd.train()
                if pbar.n >= args.steps_3:
                    break

    pbar.close()
if __name__ == '__main__':

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn is not None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    if args.init_model_cd is not None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)
    main(args)