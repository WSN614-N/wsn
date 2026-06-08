#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def truss_element_stiffness(x1, x2, E, A, ndof):
    """
    计算杆单元全局坐标系下的刚度矩阵
    参数:
        x1, x2: 节点坐标列表 [x,y,z]
        E: 弹性模量
        A: 横截面积
        ndof: 自由度数量(1/2/3)
    返回:
        Ke: 单元刚度矩阵
        L: 单元长度
        dir_cos: 方向余弦数组
    """
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    
    # 计算坐标差和长度
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1] if ndof >= 2 else 0.0
    dz = x2[2] - x1[2] if ndof == 3 else 0.0
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if L < 1e-12:
        raise ValueError("退化单元：两个节点重合")
    
    # 计算方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L
    dir_cos = np.array([cx, cy, cz]) if ndof == 3 else np.array([cx, cy]) if ndof == 2 else np.array([cx])
    
    # 构建刚度矩阵
    k_factor = E * A / L
    if ndof == 1:
        Ke = k_factor * np.array([[1, -1], [-1, 1]])
    elif ndof == 2:
        Ke = k_factor * np.array([
            [cx**2, cx*cy, -cx**2, -cx*cy],
            [cx*cy, cy**2, -cx*cy, -cy**2],
            [-cx**2, -cx*cy, cx**2, cx*cy],
            [-cx*cy, -cy**2, cx*cy, cy**2]
        ])
    elif ndof == 3:
        Ke = k_factor * np.array([
            [cx**2, cx*cy, cx*cz, -cx**2, -cx*cy, -cx*cz],
            [cx*cy, cy**2, cy*cz, -cx*cy, -cy**2, -cy*cz],
            [cx*cz, cy*cz, cz**2, -cx*cz, -cy*cz, -cz**2],
            [-cx**2, -cx*cy, -cx*cz, cx**2, cx*cy, cx*cz],
            [-cx*cy, -cy**2, -cy*cz, cx*cy, cy**2, cy*cz],
            [-cx*cz, -cy*cz, -cz**2, cx*cz, cy*cz, cz**2]
        ])
    else:
        raise ValueError(f"不支持的自由度数量 ndof = {ndof}")
    
    return Ke, L, dir_cos