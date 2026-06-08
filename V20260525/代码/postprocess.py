#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def calculate_element_stress(model):
    """计算所有单元的应力和轴力"""
    model.stress = np.zeros(model.nel)
    stress_results = []
    
    for e in range(model.nel):
        de = model.d[model.LM[:, e]]
        const = model.E[e] / model.leng[e]
        L = model.leng[e]
        
        # 获取单元方向余弦
        node1 = model.IEN[e, 0] - 1
        node2 = model.IEN[e, 1] - 1
        dx = model.x[node2] - model.x[node1]
        dy = model.y[node2] - model.y[node1] if model.ndof >= 2 else 0.0
        dz = model.z[node2] - model.z[node1] if model.ndof == 3 else 0.0
        
        cx = dx / L
        cy = dy / L
        cz = dz / L
        
        # 计算应力
        if model.ndof == 1:
            B = np.array([-1, 1])
            dir_cos = [cx]
        elif model.ndof == 2:
            B = np.array([-cx, -cy, cx, cy])
            dir_cos = [cx, cy]
        else:
            B = np.array([-cx, -cy, -cz, cx, cy, cz])
            dir_cos = [cx, cy, cz]
        
        model.stress[e] = const * (B @ de)
        N = model.stress[e] * model.CArea[e]
        
        stress_results.append({
            'element': e+1,
            'length': L,
            'dir_cos': dir_cos,
            'sigma': model.stress[e],
            'N': N
        })
    
    return stress_results

def print_results(model, stress_results):
    """输出所有计算结果"""
    print("计算结果")
    
    print("\n节点位移:")
    for i in range(model.nnp):
        if model.ndof == 1:
            print(f"   节点 {i+1}: d = {model.d[i]:.6f}")
        elif model.ndof == 2:
            u = model.d[i*2]
            v = model.d[i*2+1]
            print(f"   节点 {i+1}: u = {u:.6f}, v = {v:.6f}")
        else:
            u = model.d[i*3]
            v = model.d[i*3+1]
            w = model.d[i*3+2]
            print(f"   节点 {i+1}: u = {u:.6f}, v = {v:.6f}, w = {w:.6f}")
    
    print("\n约束反力:")
    for i, dof in enumerate(range(model.nd)):
        node = dof // model.ndof + 1
        dirs = ['x', 'y', 'z']
        dir = dirs[dof % model.ndof]
        print(f"   节点 {node} {dir}方向反力：{model.reaction[i]:.2f} N")
    
    print("\n单元应力和轴力:")
    print(f"{'单元号':<6}{'长度(m)':<12}{'应力(Pa)':<12}{'轴力(N)':<12}")
    for res in stress_results:
        dir_cos_str = f"{[round(c,4) for c in res['dir_cos']]}"
        print(f"{res['element']:<6}{res['length']:<12.4f}{res['sigma']:<12.6f}{res['N']:<12.2f}")

def plot_truss(model):
    """绘制桁架结构图"""
    if model.plot_truss != "yes":
        return
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    plt.figure(figsize=(8, 6))
    
    if model.ndof == 1:
        for i in range(model.nel):
            XX = np.array([model.x[model.IEN[i,0]-1], model.x[model.IEN[i,1]-1]])
            YY = np.array([0.0, 0.0])
            plt.plot(XX, YY, "b-", linewidth=2)
            if model.plot_node == "yes":
                plt.text(XX[0], YY[0], str(model.IEN[i,0]), fontsize=12, ha='right')
                plt.text(XX[1], YY[1], str(model.IEN[i,1]), fontsize=12, ha='left')
    elif model.ndof == 2:
        for i in range(model.nel):
            XX = np.array([model.x[model.IEN[i,0]-1], model.x[model.IEN[i,1]-1]])
            YY = np.array([model.y[model.IEN[i,0]-1], model.y[model.IEN[i,1]-1]])
            plt.plot(XX, YY, "b-", linewidth=2)
            if model.plot_node == "yes":
                plt.text(XX[0], YY[0], str(model.IEN[i,0]), fontsize=12, ha='right')
                plt.text(XX[1], YY[1], str(model.IEN[i,1]), fontsize=12, ha='left')
    
    plt.title(model.title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig("truss.pdf", dpi=300)
    print("\n桁架图已保存为 truss.pdf")
    
    if model.plot_tex == "yes":
        import tikzplotlib
        tikzplotlib.clean_figure()
        tikzplotlib.save("fe_plot.tex")
        print("LaTeX绘图文件已保存为 fe_plot.tex")