import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper


def compute_source_statistics(source_loader, model, device, K):
    """
    计算源域特征均值和子空间基向量
    Args:
        source_loader: 源域数据加载器
        model: 模型对象（需要实现 feature_extractor 方法）
        device: 运行设备（CPU 或 GPU）
        K: 子空间的维度
    Returns:
        mu_s: 源域特征均值
        V_s: 子空间基向量矩阵
        eigvals: 对应的特征值
    """
    features = []
    for batch in source_loader:
        batch = batch.to(device)
        h = model.feature_extractor(batch)
        features.append(h.cpu().detach().numpy())
    features = np.vstack(features)

    # 计算均值和协方差矩阵
    mu_s = np.mean(features, axis=0)
    centered_features = features - mu_s
    Sigma_s = np.cov(centered_features, rowvar=False)
    
    print("协方差矩阵shape",Sigma_s.shape)  #(256, 256)

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(Sigma_s)
    sorted_indices = np.argsort(eigvals)[::-1]
    with open("cormatrix.txt",'w') as f:
        print( eigvals[sorted_indices],file=f)
    # print( eigvals[sorted_indices])
    eigvals = eigvals[sorted_indices][:K] # lambdas
    eigvecs = eigvecs[:, sorted_indices][:, :K]
    V_s = eigvecs.T

    return mu_s, V_s, eigvals

def compute_lambdas_and_weights(Sigma_s, V_s, model, K):
    """
    计算源域特征值（lambdas）和加权因子（weights）
    Args:
        Sigma_s: 源域协方差矩阵
        V_s: 子空间基向量矩阵
        model: 包含预测头的模型对象
        K: 子空间维度
    Returns:
        lambdas: 子空间的特征值
        weights: 各方向的加权因子
    """
    # 1. 计算特征值 lambdas
    eigvals, eigvecs = np.linalg.eigh(Sigma_s)
    sorted_indices = np.argsort(eigvals)[::-1]
    lambdas = eigvals[sorted_indices][:K]  # 前 K 个特征值

    # 2. 获取线性回归器权重 w
    w = model.pred_head[0].weight.detach().cpu().numpy()  # 提取预测头第一层的权重

    # 3. 计算加权因子 weights
    weights = 1 + np.abs(np.dot(w, V_s.T))  # 计算投影并加权

    return lambdas, weights


def project_to_subspace(z_t, mu_s, V_s):
    centered_z_t = z_t - mu_s
    projected_z_t = torch.matmul(centered_z_t, V_s.T)
    return projected_z_t

def compute_statistics(z_proj):
    mean = torch.mean(z_proj, dim=0)
    var = torch.var(z_proj, dim=0, unbiased=False)
    return mean, var


def weighted_kl_loss(mu_t, sigma_t, lambdas, weights):
    term1 = (mu_t**2) / lambdas
    term2 = lambdas / sigma_t
    term3 = sigma_t / lambdas
    kl_loss = 0.5 * (term1 + term2 + term3 - 2).sum()
    return (weights * kl_loss).sum()
