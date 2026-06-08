#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from element import truss_element_stiffness

def generate_LM(model):
    """生成对号矩阵LM"""
    model.LM = np.zeros((model.nen * model.ndof, model.nel), dtype=int)
    for e in range(model.nel):
        for j in range(model.nen):
            global_node = model.IEN[e, j] - 1  # 转0索引
            for m in range(model.ndof):
                local_dof = j * model.ndof + m
                global_dof = global_node * model.ndof + m
                model.LM[local_dof, e] = global_dof

def assemble_global_stiffness(model):
    """组装总体刚度矩阵"""
    model.neq = model.nnp * model.ndof
    model.K = np.zeros((model.neq, model.neq), dtype=np.float64)
    model.leng = np.zeros(model.nel)
    
    for e in range(model.nel):
        # 获取单元节点坐标
        node1 = model.IEN[e, 0] - 1
        node2 = model.IEN[e, 1] - 1
        
        if model.ndof == 1:
            x1 = [model.x[node1]]
            x2 = [model.x[node2]]
        elif model.ndof == 2:
            x1 = [model.x[node1], model.y[node1]]
            x2 = [model.x[node2], model.y[node2]]
        else:
            x1 = [model.x[node1], model.y[node1], model.z[node1]]
            x2 = [model.x[node2], model.y[node2], model.z[node2]]
        
        # 计算单元刚度矩阵
        Ke, L, _ = truss_element_stiffness(x1, x2, model.E[e], model.CArea[e], model.ndof)
        model.leng[e] = L
        
        # 直接组装到总体刚度矩阵
        for i in range(2 * model.ndof):
            global_i = model.LM[i, e]
            for j in range(2 * model.ndof):
                global_j = model.LM[j, e]
                model.K[global_i, global_j] += Ke[i, j]

def check_stiffness_properties(model):
    """检查总体刚度矩阵性质（作业要求）"""
    print("总体刚度矩阵性质检查")
    
    # 1. 对称性检查
    is_symmetric = np.allclose(model.K, model.K.T)
    print(f"1. 对称性：{is_symmetric}")
    
    # 2. 奇异性检查（行列式）
    det_K = np.linalg.det(model.K)
    print(f"2. 施加边界条件前行列式：{det_K:.4e}")
    print(f"   矩阵奇异：{abs(det_K) < 1e-9}")
    
    # 3. 对角元非负性检查
    is_diag_nonnegative = np.all(np.diag(model.K) >= -1e-9)
    print(f"3. 对角元非负：{is_diag_nonnegative}")
    
    # 4. 稀疏性检查（非零元素比例）
    non_zero_count = np.count_nonzero(np.abs(model.K) > 1e-9)
    sparsity = 1 - non_zero_count / (model.neq ** 2)
    print(f"4. 稀疏度：{sparsity:.2%}")