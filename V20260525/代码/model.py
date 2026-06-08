#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class FEModel:
    """有限元模型数据类，存储所有模型参数和计算结果"""
    def __init__(self):
        # 模型基本信息
        self.title = ""
        self.nsd = 0          # 空间维度
        self.ndof = 0         # 每节点自由度
        self.nnp = 0          # 总节点数
        self.nel = 0          # 总单元数
        self.nen = 2          # 杆单元固定为2节点
        self.neq = 0          # 总方程数
        self.nd = 0           # 约束自由度数量

        # 单元属性
        self.CArea = np.array([])  # 横截面积
        self.E = np.array([])      # 弹性模量
        self.leng = np.array([])   # 单元长度
        self.stress = np.array([]) # 单元应力

        # 几何信息
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.IEN = np.array([[]])  # 单元连接数组

        # 有限元核心矩阵
        self.LM = np.array([[]])   # 对号矩阵
        self.K = np.array([[]])    # 总体刚度矩阵
        self.f = np.array([])      # 节点力向量
        self.d = np.array([])      # 节点位移向量
        self.reaction = np.array([]) # 约束反力

        # 绘图设置
        self.plot_truss = False
        self.plot_node = False
        self.plot_tex = False