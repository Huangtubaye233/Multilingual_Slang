#!/usr/bin/env python3
"""
GPU Memory Management Script
用于清理当前进程的GPU内存
"""

import torch
import gc
import os

def clear_gpu_memory():
    """清理当前进程的GPU内存"""
    if torch.cuda.is_available():
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        # 同步GPU
        torch.cuda.synchronize()
        
        # 获取内存信息
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        
        print(f"GPU内存已清理")
        print(f"当前分配: {allocated:.2f}GB")
        print(f"当前保留: {reserved:.2f}GB")
        
        return allocated, reserved
    else:
        print("CUDA不可用")
        return None, None

def get_gpu_info():
    """获取GPU信息"""
    if torch.cuda.is_available():
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  总内存: {props.total_memory / 1024**3:.2f}GB")
            
        # 当前设备信息
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"\n当前GPU {current_device}内存使用:")
        print(f"  分配: {allocated:.2f}GB")
        print(f"  保留: {reserved:.2f}GB") 
        print(f"  总计: {total:.2f}GB")
        print(f"  可用: {total - reserved:.2f}GB")

if __name__ == "__main__":
    print("=== GPU内存管理工具 ===")
    get_gpu_info()
    print("\n清理GPU内存...")
    clear_gpu_memory()
    print("\n清理完成后的状态:")
    get_gpu_info() 