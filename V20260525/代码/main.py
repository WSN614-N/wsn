#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
from model import FEModel
from assembly import generate_LM, assemble_global_stiffness, check_stiffness_properties
from solver import solve_reduction
from postprocess import calculate_element_stress, print_results, plot_truss

def main():
    if len(sys.argv) != 2:
        print("用法:python main.py 输入文件.json")
        sys.exit(1)
    
    data_file = sys.argv[1]
    np.set_printoptions(precision=4, suppress=True)
    
    # ====================== 1. 前处理：读取JSON文件 ======================
    print(f"正在读取模型文件：{data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化模型
    model = FEModel()
    model.title = data['Title']
    model.nsd = data['nsd']
    model.ndof = data['ndof']
    model.nnp = data['nnp']
    model.nel = data['nel']
    model.nen = data['nen']
    model.CArea = np.array(data['CArea'])
    model.E = np.array(data['E'])
    model.x = np.array(data['x'])
    model.y = np.array(data['y'])
    model.z = np.array(data.get('z', []))  # 3D模型才有z坐标
    model.IEN = np.array(data['IEN'], dtype=int)
    model.plot_truss = data.get('plot_truss', 'no')
    model.plot_node = data.get('plot_node', 'no')
    model.plot_tex = data.get('plot_tex', 'no')
    
    # 边界条件和载荷
    fixed_dof = data['fixed_dof']
    fixed_value = data['fixed_value']
    force_dof = data['force_dof']
    force_value = data['force_value']
    
    print(f"模型标题：{model.title}")
    print(f"空间维度：{model.nsd}")
    print(f"节点数：{model.nnp}")
    print(f"单元数：{model.nel}")
    print(f"总自由度：{model.nnp * model.ndof}")
    
    # ====================== 2. 生成对号矩阵 ======================
    generate_LM(model)
    print("\n对号矩阵 LM:")
    print(model.LM)
    
    # ====================== 3. 组装总体刚度矩阵 ======================
    assemble_global_stiffness(model)
    print("\n总体刚度矩阵 K:")
    print(model.K)
    
    # ====================== 4. 刚度矩阵性质检查 ======================
    check_stiffness_properties(model)
    
    # ====================== 5. 求解方程 ======================
    print("求解总体刚度方程")
    solve_reduction(model, fixed_dof, fixed_value, force_dof, force_value)
    
    # ====================== 6. 后处理 ======================
    stress_results = calculate_element_stress(model)
    print_results(model, stress_results)
    plot_truss(model)

    print("计算完成！")

if __name__ == "__main__":
    main()