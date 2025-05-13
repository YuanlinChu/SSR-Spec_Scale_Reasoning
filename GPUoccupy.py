import subprocess
import time
import torch
import argparse

def get_gpu_utilization(gpu_id):
    """通过调用nvidia-smi命令行工具获取GPU利用率"""
    smi_output = subprocess.check_output(
        f"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id={gpu_id}",
        shell=True,
    )
    utilization = smi_output.decode("utf-8").strip()
    if utilization == "[N/A]":
        return 0
    return int(utilization)

def gpu_burn(gpu_ids, utilization_threshold):
    """GPU burn程序"""
    matrix_size = 40960
    
    # 创建多个线程来并行处理每个GPU
    import threading
    
    def burn_gpu(gpu_id):
        while True:
            utilization = get_gpu_utilization(gpu_id)
            if utilization < utilization_threshold:
                device = torch.device(f'cuda:{gpu_id}')
                # 持续进行矩阵乘法运算
                while get_gpu_utilization(gpu_id) < utilization_threshold:
                    torch.matmul(
                        torch.randn(matrix_size, matrix_size, device=device),
                        torch.randn(matrix_size, matrix_size, device=device)
                    )
            else:
                torch.cuda.empty_cache()
            time.sleep(2)
    
    # 为每个指定的GPU创建独立的线程
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(target=burn_gpu, args=(gpu_id,))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # 等待所有线程
    for t in threads:
        t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Burn Program')
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True, help='指定要占用的GPU ID列表，例如：0 1 3')
    parser.add_argument('--utilization_threshold', type=int, default=40, help='GPU utilization threshold to start burning (percentage)')
    args = parser.parse_args()

    gpu_burn(args.gpu_ids, args.utilization_threshold)


# 占用 GPU 0 和 GPU 2
# python GPUoccupy.py --gpu_ids 0 2 --utilization_threshold 40