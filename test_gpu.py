#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU Test Script
==============
A simple script to test if TensorFlow can detect and use the GPU.
"""

import tensorflow as tf
import os
import sys
import time
import numpy as np

def test_gpu():
    """Test GPU availability and performance."""
    print("TensorFlow version:", tf.__version__)
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all messages
    
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu}")
        
        # Configure memory growth to avoid allocating all memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"Memory growth configuration error: {e}")
    else:
        print("No GPUs found. Using CPU only.")
        
    # Test calculation performance on both CPU and GPU
    # A simple matrix multiplication test
    
    # Create large matrices
    matrix_size = 5000
    print(f"\nPerforming {matrix_size}x{matrix_size} matrix multiplication test...")
    
    # Create random matrices
    A = np.random.random((matrix_size, matrix_size)).astype(np.float32)
    B = np.random.random((matrix_size, matrix_size)).astype(np.float32)
    
    # Test on CPU
    print("\nCPU Test:")
    with tf.device('/CPU:0'):
        start_time = time.time()
        C_cpu = tf.matmul(A, B)
        # Force execution
        _ = C_cpu.numpy()
        cpu_time = time.time() - start_time
        print(f"  CPU Time: {cpu_time:.4f} seconds")
    
    # Test on GPU if available
    if gpus:
        print("\nGPU Test:")
        try:
            with tf.device('/GPU:0'):
                start_time = time.time()
                C_gpu = tf.matmul(A, B)
                # Force execution
                _ = C_gpu.numpy()
                gpu_time = time.time() - start_time
                print(f"  GPU Time: {gpu_time:.4f} seconds")
                
                if cpu_time > 0 and gpu_time > 0:
                    speedup = cpu_time / gpu_time
                    print(f"  GPU Speedup: {speedup:.2f}x faster than CPU")
        except Exception as e:
            print(f"  GPU Test failed: {e}")

    print("\nGPU Test complete.")

if __name__ == "__main__":
    test_gpu()
