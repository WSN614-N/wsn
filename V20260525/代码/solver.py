#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def solve_reduction(model, fixed_dof, fixed_value, force_dof, force_value):
    """
    用缩减法求解总体刚度方程
    参数:
        fixed_dof: 约束自由度编号（1开始）
        fixed_value: 约束位移值
        force_dof: 载荷自由度编号（1开始）
        force_value: 载荷值
    """
    # 转换为0索引
    fixed_dof = np.array(fixed_dof) - 1
    force_dof = np.array(force_dof) - 1
    model.nd = len(fixed_dof)
    
    # 构建总体载荷向量
    model.f = np.zeros(model.neq, dtype=np.float64)
    model.f[force_dof] = force_value
    
    # 划分已知和未知自由度
    free_dof = np.setdiff1d(np.arange(model.neq), fixed_dof)
    
    # 分块矩阵
    K_FF = model.K[np.ix_(free_dof, free_dof)]
    K_EF = model.K[np.ix_(fixed_dof, free_dof)]
    f_F = model.f[free_dof]
    d_E = np.array(fixed_value, dtype=np.float64)
    
    # 求解未知位移
    d_F = np.linalg.solve(K_FF, f_F - K_EF.T @ d_E)
    
    # 重构完整位移向量
    model.d = np.zeros(model.neq, dtype=np.float64)
    model.d[fixed_dof] = d_E
    model.d[free_dof] = d_F
    
    # 计算约束反力
    model.reaction = model.K[fixed_dof, :] @ model.d - model.f[fixed_dof]
    
    # 检查缩减后矩阵是否非奇异
    det_KFF = np.linalg.det(K_FF)
    print(f"\n施加边界条件后缩减矩阵行列式：{det_KFF:.4e}")
    print(f"缩减矩阵非奇异：{abs(det_KFF) > 1e-9}")