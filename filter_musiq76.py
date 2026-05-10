"""
filter_musiq76.py

从 LSDIR + FFHQ 合并路径中筛选 MUSIQ 分数 > 76 的高质量图像
输出：musiq76_paths.txt，每行一个图像路径

用法：
  python filter_musiq76.py \
    --input_txt /data/datasets/LSDIR/actual_image_paths.txt \
    --output_txt /data/datasets/LSDIR/musiq76_paths.txt \
    --threshold 76.0 \
    --device cuda \
    --gpu_ids 0,1,2,3

多卡并行说明：
  脚本会自动把图像列表按 GPU 数量切分
  每张卡处理一部分，最后合并结果
  
  单卡运行只需去掉 --gpu_ids 参数
"""

import os
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True,
                        help='图像路径列表（LSDIR+FFHQ合并）')
    parser.add_argument('--output_txt', type=str, required=True,
                        help='筛选后的高质量图像路径列表')
    parser.add_argument('--threshold', type=float, default=76.0,
                        help='MUSIQ 分数阈值，默认 76')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda 或 cpu')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='多卡并行时指定 GPU ID，如 0,1,2,3')
    return parser.parse_args()


def filter_worker(rank, gpu_id, paths_chunk, threshold, output_file, lock):
    """单个 GPU 的筛选进程"""
    import pyiqa
    device = f'cuda:{gpu_id}'
    musiq_metric = pyiqa.create_metric('musiq', device=device)

    qualified = []
    failed = 0

    for i, path in enumerate(tqdm(paths_chunk, desc=f'GPU{gpu_id}', position=rank)):
        try:
            score = musiq_metric(path).item()
            if score > threshold:
                qualified.append(path)
        except Exception as e:
            failed += 1

        if (i + 1) % 1000 == 0:
            print(f'[GPU{gpu_id}] 已处理 {i+1}/{len(paths_chunk)}，'
                  f'筛选出 {len(qualified)} 张，失败 {failed} 张')

    # 加锁写文件，避免多进程冲突
    with lock:
        with open(output_file, 'a') as f:
            for path in qualified:
                f.write(path + '\n')

    print(f'[GPU{gpu_id}] 完成，筛选出 {len(qualified)} 张，失败 {failed} 张')


def main():
    args = parse_args()

    # 读取输入路径
    with open(args.input_txt, 'r') as f:
        all_paths = [line.strip() for line in f.readlines() if line.strip()]
    print(f'共读取 {len(all_paths)} 张图像路径')

    # 确保输出目录存在，清空旧文件
    os.makedirs(os.path.dirname(os.path.abspath(args.output_txt)), exist_ok=True)
    if os.path.exists(args.output_txt):
        os.remove(args.output_txt)
        print(f'已清空旧的输出文件：{args.output_txt}')

    # 解析 GPU ID
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    print(f'使用 {num_gpus} 张 GPU：{gpu_ids}')

    if num_gpus == 1:
        # 单卡模式
        try:
            import pyiqa
        except ImportError:
            print('请先安装 pyiqa：pip install pyiqa')
            return

        device = f'cuda:{gpu_ids[0]}' if args.device == 'cuda' else 'cpu'
        musiq_metric = pyiqa.create_metric('musiq', device=device)
        print(f'MUSIQ 模型加载成功，设备：{device}')
        print(f'开始筛选，阈值 MUSIQ > {args.threshold}...')

        qualified = []
        failed = 0

        for i, path in enumerate(tqdm(all_paths)):
            try:
                score = musiq_metric(path).item()
                if score > args.threshold:
                    qualified.append(path)
            except Exception as e:
                failed += 1

            if (i + 1) % 1000 == 0:
                print(f'已处理 {i+1}/{len(all_paths)}，'
                      f'筛选出 {len(qualified)} 张，失败 {failed} 张')

        with open(args.output_txt, 'w') as f:
            for path in qualified:
                f.write(path + '\n')

        print(f'\n筛选完成！')
        print(f'总图像数：{len(all_paths)}')
        print(f'MUSIQ > {args.threshold} 的图像数：{len(qualified)}')
        print(f'失败图像数：{failed}')
        print(f'结果已保存到：{args.output_txt}')

    else:
        # 多卡模式：把路径列表切分给每张卡
        chunk_size = len(all_paths) // num_gpus
        chunks = []
        for i, gpu_id in enumerate(gpu_ids):
            if i == num_gpus - 1:
                # 最后一张卡处理剩余的
                chunks.append(all_paths[i * chunk_size:])
            else:
                chunks.append(all_paths[i * chunk_size: (i + 1) * chunk_size])

        for i, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
            print(f'GPU{gpu_id} 负责处理 {len(chunk)} 张图像')

        # 多进程并行
        lock = mp.Lock()
        processes = []
        for rank, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
            p = mp.Process(
                target=filter_worker,
                args=(rank, gpu_id, chunk, args.threshold, args.output_txt, lock)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 统计结果
        with open(args.output_txt, 'r') as f:
            total_qualified = len(f.readlines())

        print(f'\n所有 GPU 筛选完成！')
        print(f'总图像数：{len(all_paths)}')
        print(f'MUSIQ > {args.threshold} 的图像数：{total_qualified}')
        print(f'结果已保存到：{args.output_txt}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()