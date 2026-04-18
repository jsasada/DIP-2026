import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 定义 Bundle Adjustment 模型
# ==========================================
class BundleAdjustmentModel(nn.Module):
    def __init__(self, num_points, num_views, img_size=(1024, 1024), d_init=2.5):
        super().__init__()
        self.num_points = num_points
        self.num_views = num_views
        self.cx = img_size[0] / 2.0
        self.cy = img_size[1] / 2.0
        
        # 初始化 3D 点: 在原点附近加入小范围的随机噪声 (N, 3)
        self.points3d = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        
        # 初始化相机参数 (50个相机)
        # 旋转: 使用 Euler 角初始化为 0 (单位矩阵)
        self.euler_angles = nn.Parameter(torch.zeros(num_views, 3))
        
        # 平移: 初始化在正前方 [0, 0, -d_init]
        t_init = torch.zeros(num_views, 3)
        t_init[:, 2] = -d_init
        self.translations = nn.Parameter(t_init)
        
        # 初始化焦距: 假设 FoV ~ 60度 -> f = W / (2 * tan(30°)) ≈ 886
        self.focal_length = nn.Parameter(torch.tensor([900.0]))

    def forward(self):
        """
        将 3D 点投影到 50 个 2D 视图中。
        """
        # 将 Euler 角转为旋转矩阵 (V, 3, 3)
        R = euler_angles_to_matrix(self.euler_angles)
        T = self.translations.unsqueeze(1) # (V, 1, 3)
        
        # 将 3D 点扩展到各视图 (V, N, 3)
        # P_c = P @ R^T + T
        points_expanded = self.points3d.unsqueeze(0).expand(self.num_views, -1, -1)
        points_cam = torch.bmm(points_expanded, R.transpose(1, 2)) + T
        
        # 提取 Xc, Yc, Zc
        Xc = points_cam[:, :, 0]
        Yc = points_cam[:, :, 1]
        Zc = points_cam[:, :, 2]
        
        # 防止 Zc 过小导致除零错误
        Zc = torch.clamp(Zc, max=-1e-4) # 因为相机在物体前方(Zc < 0)
        
        # 透视投影计算
        u = -self.focal_length * (Xc / Zc) + self.cx
        v =  self.focal_length * (Yc / Zc) + self.cy
        
        # 拼接预测的 (u, v) 坐标 -> (V, N, 2)
        predicted_uv = torch.stack([u, v], dim=-1)
        
        return predicted_uv

# ==========================================
# 核心优化与训练循环
# ==========================================
def run_bundle_adjustment(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    points2d_path = os.path.join(data_dir, 'points2d.npz')
    colors_path = os.path.join(data_dir, 'points3d_colors.npy')
    
    points2d_data = np.load(points2d_path)
    view_keys = sorted(points2d_data.files)
    num_views = len(view_keys)
    num_points = points2d_data[view_keys[0]].shape[0]
    
    # 组合真值 GT 数据 (V, N, 3) -> (50, 20000, 3)
    gt_2d_all = np.stack([points2d_data[k] for k in view_keys], axis=0)
    gt_2d_tensor = torch.tensor(gt_2d_all, dtype=torch.float32, device=device)
    
    gt_uv = gt_2d_tensor[:, :, :2]     # (V, N, 2)
    visibility = gt_2d_tensor[:, :, 2] # (V, N) - 1.0 为可见，0.0 为遮挡
    
    # --- 模型与优化器初始化 ---
    model = BundleAdjustmentModel(num_points, num_views).to(device)
    optimizer = optim.Adam([
        {'params': [model.points3d], 'lr': 1e-2},
        {'params': [model.euler_angles, model.translations], 'lr': 1e-3},
        {'params': [model.focal_length], 'lr': 1e-1}
    ])
    
    # --- 训练循环 ---
    num_epochs = 5000
    loss_history = []
    
    print("Starting optimization...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播得到预测的 2D 点
        pred_uv = model()
        
        # 计算 L2 重投影误差 (仅计算可见点的误差)
        diff = pred_uv - gt_uv
        sq_dist = torch.sum(diff ** 2, dim=-1) # (V, N)
        
        # 应用可见性掩码 (Masking)
        masked_sq_dist = sq_dist * visibility
        
        # 求均方误差 (MSE)
        loss = torch.sum(masked_sq_dist) / torch.sum(visibility)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Focal Length: {model.focal_length.item():.2f}")
            
    print("Optimization finished!")
    
    # --- 可视化 Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Reprojection Loss')
    plt.title("Bundle Adjustment Optimization")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("Saved loss curve to 'loss_curve.png'.")
    
    # --- 导出重建结果为带颜色的 OBJ ---
    optimized_points = model.points3d.detach().cpu().numpy()
    colors = np.load(colors_path) if os.path.exists(colors_path) else np.ones_like(optimized_points) * 0.8
    
    export_obj("reconstructed_head.obj", optimized_points, colors)

# ==========================================
# 辅助函数
# ==========================================
def export_obj(filename, points, colors):
    """
    将 3D 点和颜色写入 .obj 文件
    格式: v x y z r g b
    """
    assert points.shape == colors.shape, "Points and colors must have the same shape"
    with open(filename, 'w') as f:
        f.write("# Reconstructed 3D Points via Bundle Adjustment\n")
        for p, c in zip(points, colors):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    print(f"Saved reconstructed 3D points to '{filename}'.")

def euler_angles_to_matrix(euler_angles):
    """
    欧拉角到旋转矩阵
    """
    x, y, z = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]

    c1, s1 = torch.cos(x), torch.sin(x)
    c2, s2 = torch.cos(y), torch.sin(y)
    c3, s3 = torch.cos(z), torch.sin(z)

    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    # Rx
    Rx = torch.stack([
        one, zero, zero,
        zero, c1, -s1,
        zero, s1, c1
    ], dim=-1).reshape(euler_angles.shape[:-1] + (3, 3))

    # Ry
    Ry = torch.stack([
        c2, zero, s2,
        zero, one, zero,
        -s2, zero, c2
    ], dim=-1).reshape(euler_angles.shape[:-1] + (3, 3))

    # Rz
    Rz = torch.stack([
        c3, -s3, zero,
        s3, c3, zero,
        zero, zero, one
    ], dim=-1).reshape(euler_angles.shape[:-1] + (3, 3))

    # XYZ convention means R = Rx @ Ry @ Rz
    return Rx @ Ry @ Rz

if __name__ == "__main__":
    run_bundle_adjustment(data_dir="./data")