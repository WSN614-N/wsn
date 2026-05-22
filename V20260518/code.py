import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    # 转换为numpy数组
    x1 = np.array(x1, dtype=np.float64)
    x2 = np.array(x2, dtype=np.float64)
    
    # 计算坐标差
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    
    # 计算单元长度
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 检查退化单元（节点重合）
    if L < 1e-12:
        raise ValueError("错误：两个节点重合，为退化单元！")
    
    # 计算方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L
    direction_cosines = np.array([cx, cy, cz])
    
    # 构建刚度矩阵
    k_factor = E * A / L
    Ke = k_factor * np.array([
        [cx**2, cx*cy, cx*cz, -cx**2, -cx*cy, -cx*cz],
        [cx*cy, cy**2, cy*cz, -cx*cy, -cy**2, -cy*cz],
        [cx*cz, cy*cz, cz**2, -cx*cz, -cy*cz, -cz**2],
        [-cx**2, -cx*cy, -cx*cz, cx**2, cx*cy, cx*cz],
        [-cx*cy, -cy**2, -cy*cz, cx*cy, cy**2, cy*cz],
        [-cx*cz, -cy*cz, -cz**2, cx*cz, cy*cz, cz**2]
    ])
    
    return L, direction_cosines, Ke

def truss3d_element_stress(x1, x2, E, A, de):
    # 获取单元长度和方向余弦
    L, (cx, cy, cz), _ = truss3d_element_stiffness(x1, x2, E, A)
    
    # 构建应变-位移矩阵B
    B = np.array([-cx, -cy, -cz, cx, cy, cz])
    
    # 计算应变、应力和轴力
    epsilon = B.dot(de) / L
    sigma = E * epsilon
    N = sigma * A
    
    return epsilon, sigma, N

# ====================== 验证算例 ======================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)  # 设置输出格式
    
    print("="*60)
    print("算例1：沿x轴的一维杆单元")
    print("="*60)
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9  # Pa
    A = 1.0e-4  # m²
    de = [0, 0, 0, 1.0e-3, 0, 0]  # m
    
    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    
    print(f"单元长度 L = {L:.4f} m")
    print(f"方向余弦 (cx, cy, cz) = {dir_cos}")
    print("\n单元刚度矩阵 Ke (N/m):")
    print(Ke)
    print(f"\n轴向应变 ε = {epsilon:.6e}")
    print(f"轴向应力 σ = {sigma/1e6:.2f} MPa")
    print(f"轴向轴力 N = {N:.2f} N")
    
    print("\n" + "="*60)
    print("算例2：空间任意方向杆单元")
    print("="*60)
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9  # Pa
    A = 2.0e-4  # m²
    de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]  # m
    
    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    
    print(f"单元长度 L = {L:.4f} m")
    print(f"方向余弦 (cx, cy, cz) = {dir_cos}")
    print("\n单元刚度矩阵 Ke (N/m):")
    print(Ke)
    print(f"\n轴向应变 ε = {epsilon:.6e}")
    print(f"轴向应力 σ = {sigma/1e6:.2f} MPa")
    print(f"轴向轴力 N = {N:.2f} N")
    
    # ====================== 单元刚度矩阵性质验证 ======================
    print("\n" + "="*60)
    print("单元刚度矩阵性质验证")
    print("="*60)
    
    # 1. 对称性验证
    is_symmetric = np.allclose(Ke, Ke.T)
    print(f"1. 对称性验证：{is_symmetric}")
    
    # 2. 奇异性验证（行列式接近0）
    det_Ke = np.linalg.det(Ke)
    print(f"2. 行列式值：{det_Ke:.4e}（接近0，说明矩阵奇异）")
    
    # 3. 特征值非负性验证
    eigenvalues, _ = np.linalg.eig(Ke)
    eigenvalues = np.sort(eigenvalues.real)  # 取实部并排序
    print(f"3. 特征值（从小到大）：{eigenvalues}")
    print(f"   最小特征值：{eigenvalues[0]:.4e}（接近0，对应刚体位移）")
    print(f"   所有特征值非负：{np.all(eigenvalues >= -1e-9)}")
    
    # 4. 刚体位移验证（整体平移不产生内力）
    de_rigid = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 整体平移1m
    Fe_rigid = Ke.dot(de_rigid)
    j=2
    de_test = np.zeros(6)
    de_test[j] = 1.0
    Fe_test = Ke.dot(de_test)
    print(f"\n4. 刚体位移验证（整体平移）：")
    print(f"   节点内力列阵 Fe = {Fe_rigid}（接近零向量）")
    print(f"   内力最大值：{np.max(np.abs(Fe_rigid)):.4e} N") 
    print(f"令第{j+1}个自由度（节点1的z方向）位移为1，其他固定：")
    print(f"计算得到的节点内力 Fe = {Fe_test}")
    print(f"刚度矩阵第{j+1}列 Ke[:,{j}] = {Ke[:, j]}")
    print(f"两者相等：{np.allclose(Fe_test, Ke[:, j])}")