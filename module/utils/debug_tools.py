import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# ============ 1. 检查特征坍缩 ============ #
def check_collapse(z1, z2, epoch, batch_idx, save_dir=None, t=2):
    """
    检查特征 collapse 和数值异常
    Args:
        z1, z2: 两个模态或两视图的 embedding (tensor)
        epoch, batch_idx: 当前轮次、batch
        save_dir: 可选，日志保存目录
        t: uniformity 指标中的超参，通常取 2
    """
    # detach 并计算指标
    z1 = z1.detach()
    z2 = z2.detach()
    alignment = torch.norm(z1 - z2, dim=1).mean().item()

    pdist = torch.cdist(z1, z1, p=2)
    uniformity = torch.log(torch.exp(-t * (pdist ** 2)).mean() + 1e-9).item()

    msg = f"[collapse] epoch={epoch} batch={batch_idx} | align={alignment:.4f} uniform={uniformity:.4f}"
    print(msg)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "collapse_log.txt"), "a") as f:
            f.write(msg + "\n")

    return alignment, uniformity


# ============ 2. 保存 Sinkhorn 的 P 矩阵热力图 ============ #
# def save_P_heatmap(P, epoch, batch_idx=0, save_dir="./logs/debug", use_log=True, prefix=""):
#     """
#     保存 P 或 -log(P) 的热力图
#     Args:
#         P: Sinkhorn 输出矩阵 (tensor)
#         epoch: 当前轮次
#         batch_idx: batch编号
#         save_dir: 保存目录
#         use_log: 是否绘制 -log(P)
#         prefix: 文件名前缀，用于区分不同的P矩阵
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     P_np = P.detach().cpu().numpy()
    
#     # 添加前缀到文件名
#     prefix_str = f"{prefix}_" if prefix else ""
#     np.save(os.path.join(save_dir, f"{prefix_str}P_epoch{epoch}_b{batch_idx}.npy"), P_np)

#     if use_log:
#         img_data = -np.log(P_np + 1e-9)
#         title = f"-log(P) heatmap epoch {epoch}" + (f" ({prefix})" if prefix else "")
#     else:
#         img_data = P_np
#         title = f"P heatmap epoch {epoch}" + (f" ({prefix})" if prefix else "")

#     plt.figure(figsize=(5, 5))
#     plt.imshow(img_data, cmap="viridis")
#     plt.colorbar()
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"{prefix_str}P_epoch{epoch}_b{batch_idx}.png"))
#     plt.close()


def save_P_heatmap(
    P,
    epoch,
    batch_idx=0,
    save_dir="./logs/debug",
    use_log=True,
    prefix="",
    labels_row=None,      # 新增：行标签（按它重排行）
    labels_col=None,      # 新增：列标签（不传则用行标签）
    bucket_bins=None,     # 新增：回归标签时的分桶边界（如 [-1, -0.33, 0.33, 1]）
    sort_mode="asc",      # 新增："asc"升序 或 "desc"降序
    draw_boundaries=False # 新增：是否在热力图绘制类别分隔线与刻度
):
    """
    保存 P 或 -log(P) 的热力图
    Args:
        P: Sinkhorn 输出矩阵 (tensor, 形状 [B, B])
        epoch: 当前轮次
        batch_idx: batch编号
        save_dir: 保存目录
        use_log: 是否绘制 -log(P)
        prefix: 文件名前缀，用于区分不同的P矩阵
        labels_row: 按此标签对 P 的行排序（tensor 或 numpy，可为回归或分类）
        labels_col: 按此标签对 P 的列排序（不传则复用 labels_row）
        bucket_bins: 回归标签分桶边界（列表）；传了则先对标签分桶再排序
        sort_mode: "asc" 或 "desc"
        draw_boundaries: 是否在热力图上绘制类别分隔线与刻度
    """
    import torch
    os.makedirs(save_dir, exist_ok=True)

    # 1) 若提供标签，则先对 P 的行/列做重排（在转 numpy 之前）
    labels_sorted = None
    if labels_row is not None:
        # 统一到 CPU 1D tensor
        lr = torch.as_tensor(labels_row).detach().cpu().flatten()
        lc = torch.as_tensor(labels_col).detach().cpu().flatten() if labels_col is not None else lr.clone()

        # 若需要对回归标签分桶（把浮点标签离散化）
        if bucket_bins is not None:
            bins = torch.as_tensor(bucket_bins, dtype=torch.float32)
            if lr.dtype.is_floating_point:
                lr = torch.bucketize(lr, bins)
            if lc.dtype.is_floating_point:
                lc = torch.bucketize(lc, bins)

        # 计算排序索引
        row_idx = torch.argsort(lr)
        col_idx = torch.argsort(lc)
        if sort_mode == "desc":
            row_idx = torch.flip(row_idx, dims=[0])
            col_idx = torch.flip(col_idx, dims=[0])

        # 行列同步重排
        P = P[row_idx][:, col_idx]
        labels_sorted = lr[row_idx]  # 用于可选地绘制分隔线与刻度

    # 2) 之后与原逻辑一致：保存 .npy 与绘图
    P_np = P.detach().cpu().numpy()

    # 添加前缀到文件名
    prefix_str = f"{prefix}_" if prefix else ""
    np.save(os.path.join(save_dir, f"{prefix_str}P_epoch{epoch}_b{batch_idx}.npy"), P_np)

    if use_log:
        img_data = -np.log(P_np + 1e-9)
        title = f"-log(P) heatmap epoch {epoch}" + (f" ({prefix})" if prefix else "")
    else:
        img_data = P_np
        title = f"P heatmap epoch {epoch}" + (f" ({prefix})" if prefix else "")

    plt.figure(figsize=(5, 5))
    plt.imshow(img_data, cmap="viridis")
    plt.colorbar()
    plt.title(title)

    # 3) 可选：绘制类别分隔线与刻度（当提供 labels_row 且启用 draw_boundaries）
    if draw_boundaries and labels_sorted is not None:
        # 统计连续段边界（按排序后的标签）
        unique_vals, counts = torch.unique(labels_sorted, return_counts=True)
        boundaries = torch.cumsum(counts, dim=0).tolist()[:-1]  # 每段末尾的 index

        # 画分隔线（注意 -0.5 对齐网格）
        for b in boundaries:
            plt.axhline(b - 0.5, color="white", linewidth=0.5)
            plt.axvline(b - 0.5, color="white", linewidth=0.5)

        # 刻度放在每段中点
        ticks = []
        tick_labels = []
        start = 0
        for val, cnt in zip(unique_vals.tolist(), counts.tolist()):
            center = start + cnt / 2.0
            ticks.append(center)
            tick_labels.append(str(val))
            start += cnt
        plt.xticks(ticks, tick_labels, rotation=45, fontsize=8)
        plt.yticks(ticks, tick_labels, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix_str}P_epoch{epoch}_b{batch_idx}.png"))
    plt.close()

# ============ 3. 保存嵌入分布散点图（PCA 可视化） ============ #
def save_embeddings(z, epoch, batch_idx=0, save_dir="./logs/debug", n_components=2):
    """
    用 PCA 将 embedding 映射到二维并保存散点图
    Args:
        z: tensor [B, D]
        epoch, batch_idx: 当前轮次
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    z_np = z.detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    z_2d = pca.fit_transform(z_np)

    plt.figure(figsize=(5, 5))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=6, alpha=0.7)
    plt.title(f"Embeddings (PCA) epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"embed_epoch{epoch}_b{batch_idx}.png"))
    plt.close()
