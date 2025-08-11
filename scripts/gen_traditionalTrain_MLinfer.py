import argparse, os, time

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# ---- MP + 物理生成 ----

from mp_api.client import MPRester

import pymatgen.io.ase

from janus_core.calculations.phonons import Phonons

import euphonic

from euphonic.cli.utils import _get_debye_waller

from euphonic.powder import sample_sphere_structure_factor

from euphonic import ureg

# ---- 模型 ----

from model2 import SRCNN, PowderUNet, FNO2d

# ============== 配置 ==============

MATPROJ_APIKEY = '***REMOVED***'


# =================================

BLACKLIST = {"Ac"}  # 黑名单元素集合


# ---------- utils ----------
def sanitize(arr, clip_pct=99.9, nonneg=True):
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.all():
        a = np.where(finite, a, 0.0)
    if nonneg:
        a = np.maximum(a, 0.0)
    hi = np.percentile(a, clip_pct)
    if hi > 0:
        a = np.clip(a, None, hi)
    return a


def fetch_mp_ids(api_key, limit=80, num_elements=(1, 3), nsites_max=20, blacklist={"Ac"}):
    """自动从MP获取ID，并过滤黑名单元素"""
    ids = []
    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.summary.search(
            fields=["material_id", "elements", "nsites"],
            num_elements=num_elements,
            is_stable=True
        )
        for d in docs:
            if d.nsites is not None and d.nsites > nsites_max:
                continue
            if set(d.elements) & blacklist:
                continue
            ids.append(d.material_id)
            if len(ids) >= limit:
                break
    return ids


def gen_single(mp_id, do_plot=False):
    with MPRester(api_key=MATPROJ_APIKEY) as mp:
        struct = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(
            mp.get_structure_by_material_id(mp_id)
        )
    formula = struct.get_chemical_formula()
    print(f"Processing {mp_id} ({formula})")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    ph = Phonons(
        struct=struct.copy(), arch="mace_mp", device=device_str, model="small",
        calc_kwargs={"default_dtype": "float64"},
        supercell=[2, 2, 2], displacement=0.01, temp_step=2.0, temp_min=0.0, temp_max=2000.0,
        minimize=True,
        minimize_kwargs={"filter_kwargs": {"hydrostatic_strain": False}, "fmax": 0.1, "optimizer": "MDMin"},
        force_consts_to_hdf5=True, plot_to_file=False, symmetrize=False, write_full=True, write_results=True,
    )
    ph.calc_force_constants()
    ph.calc_bands(write_bands=True)

    fc = euphonic.ForceConstants.from_phonopy(
        summary_name=f'janus_results/{formula}-phonopy.yml',
        fc_name=f'janus_results/{formula}-force_constants.hdf5'
    )

    tt = 5 * ureg('K')
    dw = _get_debye_waller(tt, fc)

    # 高分辨率
    qbins_h = np.linspace(0, 6, 101) * ureg('1 / angstrom')
    qc_h = (qbins_h[:-1] + qbins_h[1:]) / 2
    ebins_h = np.linspace(0, 60, 201) * ureg('meV')
    z_high = np.empty((len(qc_h), len(ebins_h) - 1))
    for i, q in enumerate(qc_h):
        spec = sample_sphere_structure_factor(fc, q, dw=dw, temperature=tt,
                                              sampling='golden', jitter=True,
                                              energy_bins=ebins_h / 1.2)
        z_high[i, :] = spec.y_data.magnitude

    # 粗分辨率
    qbins_c = np.linspace(0, 6, 21) * ureg('1 / angstrom')
    qc_c = (qbins_c[:-1] + qbins_c[1:]) / 2
    ebins_c = np.linspace(0, 60, 21) * ureg('meV')
    z_coarse = np.empty((len(qc_c), len(ebins_c) - 1))
    for i, q in enumerate(qc_c):
        spec_c = sample_sphere_structure_factor(fc, q, dw=dw, temperature=tt,
                                                sampling='golden', npts=200,
                                                jitter=True, energy_bins=ebins_c / 1.2)
        z_coarse[i, :] = spec_c.y_data.magnitude

    if not (np.all(np.isfinite(z_coarse)) and np.all(np.isfinite(z_high))):
        raise ValueError("non-finite values in z_coarse/z_high")

    return z_coarse.flatten(), z_high.flatten()
# ---------- 训练 + 推断 ----------

def collect_samples(mp_ids):
    inputs, targets, ok_ids, errors = [], [], [], []

    for mpid in mp_ids:

        try:

            inp, tgt = gen_single(mpid, do_plot=False)

            if not (np.all(np.isfinite(inp)) and np.all(np.isfinite(tgt))):
                raise ValueError("non-finite values in sample")

            inputs.append(inp);
            targets.append(tgt);
            ok_ids.append(mpid)

            print(f"✓ {mpid}")

        except Exception as e:

            print(f"✗ {mpid}: {type(e).__name__}: {e}")

            errors.append((mpid, f"{type(e).__name__}: {e}"))

    return inputs, targets, ok_ids, errors


def compute_accuracy(pred, target, tol=0.05):
    """相对误差小于 tol 计为正确"""

    pred_np = pred.detach().cpu().numpy()

    target_np = target.detach().cpu().numpy()

    denom = np.abs(target_np) + 1e-8

    rel_err = np.abs(pred_np - target_np) / denom

    return float((rel_err < tol).mean())


from sklearn.metrics import r2_score


# ===== 自定义 RMSELoss =====
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(torch.mean((yhat - y) ** 2) + self.eps)

# ---- 反归一化工具 ----
def inverse_scale(arr, scaler):
    """反归一化：从 [0,1] 缩放值恢复到原物理量"""

    return arr * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['srcnn', 'unet', 'fno'], required=True)

    parser.add_argument('--epochs', type=int, default=80)

    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--limit', type=int, default=60)

    parser.add_argument('--speed-mpid', type=str, default=None)

    args = parser.parse_args()

    # ========== 采集数据 ==========

    mp_ids = fetch_mp_ids(MATPROJ_APIKEY, limit=args.limit, num_elements=(1, 3),

                          nsites_max=20, blacklist=BLACKLIST)

    print(f"Fetched {len(mp_ids)} IDs (first 10):", mp_ids[:10])

    if not mp_ids:
        raise RuntimeError("No mp_ids fetched. Check API key/filters.")

    inputs, targets = [], []

    for mpid in mp_ids:

        try:

            inp, tgt = gen_single(mpid, do_plot=False)

            if not (np.all(np.isfinite(inp)) and np.all(np.isfinite(tgt))):
                raise ValueError("non-finite values in sample")

            inputs.append(inp)

            targets.append(tgt)

            print(f"✓ {mpid}")

        except Exception as e:

            print(f"✗ {mpid}: {type(e).__name__}: {e}")

    if not inputs:
        raise RuntimeError("No valid samples collected!")

    # ========== 数据预处理 ==========

    X = sanitize(np.vstack(inputs), clip_pct=99.9, nonneg=True)

    Y = sanitize(np.vstack(targets), clip_pct=99.9, nonneg=True)

    X = np.log1p(X)

    Y = np.log1p(Y)

    sx, sy = MinMaxScaler(), MinMaxScaler()

    Xs = sx.fit_transform(X).reshape(-1, 1, 20, 20)

    Ys = sy.fit_transform(Y).reshape(-1, 1, 100, 200)

    X_tr, X_val, y_tr, y_val = train_test_split(Xs, Ys, test_size=0.2, random_state=0)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),

                                            torch.from_numpy(y_tr).float()),

                              batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(),

                                          torch.from_numpy(y_val).float()),

                            batch_size=args.batch_size, shuffle=False)

    # ========== 模型 ==========

    if args.model == 'srcnn':

        net = SRCNN()

    elif args.model == 'unet':

        net = PowderUNet()

    else:

        net = FNO2d(modes1=20, modes2=10, width=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device).float()

    # ========== 损失与优化器 ==========

    criterion = nn.SmoothL1Loss(beta=0.01)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=True)

    # ========== 训练 ==========

    best_val_rmse = float('inf')

    patience, bad = 99999, 0

    train_rmses, val_rmses = [], []

    train_r2s, val_r2s = [], []

    for ep in range(1, args.epochs + 1):

        net.train()

        train_preds, train_gts = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            base = nn.functional.interpolate(xb, size=(100, 200), mode='bilinear', align_corners=False)

            res = net(xb)

            pred = (base + res).clamp(0.0, 1.0)

            loss = criterion(pred, yb)

            opt.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            opt.step()

            # 反归一化

            pred_np = inverse_scale(pred.detach().cpu().numpy(), sy)

            gt_np = inverse_scale(yb.cpu().numpy(), sy)

            pred_np = np.expm1(pred_np)

            gt_np = np.expm1(gt_np)

            train_preds.append(pred_np)

            train_gts.append(gt_np)

        train_preds = np.vstack([p.reshape(len(p), -1) for p in train_preds])

        train_gts = np.vstack([g.reshape(len(g), -1) for g in train_gts])

        train_rmse = np.sqrt(np.mean((train_preds - train_gts) ** 2))

        train_r2 = r2_score(train_gts, train_preds)

        # ===== 验证 =====

        net.eval()

        val_preds, val_gts = [], []

        with torch.no_grad():

            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                base = nn.functional.interpolate(xb, size=(100, 200), mode='bilinear', align_corners=False)

                pred = (base + net(xb)).clamp(0.0, 1.0)

                pred_np = inverse_scale(pred.cpu().numpy(), sy)

                gt_np = inverse_scale(yb.cpu().numpy(), sy)

                pred_np = np.expm1(pred_np)

                gt_np = np.expm1(gt_np)

                val_preds.append(pred_np)

                val_gts.append(gt_np)

        val_preds = np.vstack([p.reshape(len(p), -1) for p in val_preds])

        val_gts = np.vstack([g.reshape(len(g), -1) for g in val_gts])

        val_rmse = np.sqrt(np.mean((val_preds - val_gts) ** 2))

        val_r2 = r2_score(val_gts, val_preds)

        train_rmses.append(train_rmse)

        val_rmses.append(val_rmse)

        train_r2s.append(train_r2)

        val_r2s.append(val_r2)

        sched.step(val_rmse)

        print(f"Epoch {ep:03d} | Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f} | "

              f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")

        # 保存最佳模型

        if val_rmse < best_val_rmse - 1e-6:

            best_val_rmse = val_rmse

            bad = 0

            os.makedirs('checkpoints', exist_ok=True)

            payload = {

                'model_name': args.model,

                'state_dict': net.state_dict(),

                'sx_min': torch.from_numpy(sx.data_min_.copy()),

                'sx_max': torch.from_numpy(sx.data_max_.copy()),

                'sy_min': torch.from_numpy(sy.data_min_.copy()),

                'sy_max': torch.from_numpy(sy.data_max_.copy()),

            }

            torch.save(payload, f'checkpoints/{args.model}.pt')

        else:

            bad += 1

            if bad >= patience:
                print("Early stopping.")

                break

    # ===== 保存训练曲线 =====

    fig = plt.figure(figsize=(6, 4))

    plt.plot(train_rmses, label='Train RMSE')

    plt.plot(val_rmses, label='Val RMSE')

    plt.xlabel('Epoch')

    plt.ylabel('RMSE')

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    fig.savefig('checkpoints/rmse_curve.png', dpi=180)

    plt.close(fig)

    # ====== 对比耗时 ======

    speed_mpid = args.speed_mpid or mp_ids[0]

    print(f"\n=== Speed test on {speed_mpid} ===")

    t0 = time.time()

    zc_vec, zh_vec = gen_single(speed_mpid, do_plot=False)

    t_brute = time.time() - t0

    zc = sanitize(zc_vec, clip_pct=99.9, nonneg=True)

    zc = np.log1p(zc).reshape(1, -1)

    zc_scaled = (zc - sx.data_min_) / (sx.data_max_ - sx.data_min_ + 1e-12)

    x = torch.from_numpy(zc_scaled.reshape(1, 1, 20, 20)).float().to(device)

    with torch.no_grad():

        t1 = time.time()

        base = nn.functional.interpolate(x, size=(100, 200), mode='bilinear', align_corners=False)

        res = net(x)

        y_scaled = (base + res).clamp(0.0, 1.0)

        t_ml = time.time() - t1

    print(f"Brute force: {t_brute:.2f} s")

    print(f"ML infer  : {t_ml * 1000:.2f} ms")

    if t_ml > 0:
        print(f"Speedup   : {t_brute / t_ml:.1f}×")

    # ===== 画对比图 =====

    y_scaled_np = y_scaled.cpu().numpy().reshape(1, -1)

    y_log = inverse_scale(y_scaled_np, sy)

    y_pred = np.expm1(y_log).reshape(100, 200)

    y_true = zh_vec.reshape(100, 200)

    vmin = 0.0

    vmax = np.nanpercentile(np.concatenate([y_true.ravel(), y_pred.ravel()]), 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(y_true, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    axes[0].set_title('Brute-force')

    axes[0].set_xlabel('Energy bin');
    axes[0].set_ylabel('|Q| bin')

    fig.colorbar(im0, ax=axes[0]).ax.set_ylabel('Intensity')

    im1 = axes[1].imshow(y_pred, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    axes[1].set_title(f'ML ({args.model})')

    axes[1].set_xlabel('Energy bin');
    axes[1].set_ylabel('|Q| bin')

    fig.colorbar(im1, ax=axes[1]).ax.set_ylabel('Intensity')

    fig.savefig('checkpoints/pred_vs_gt.png', dpi=180)

    plt.close(fig)

    print("Saved: checkpoints/pred_vs_gt.png")


if __name__ == '__main__':
    main()
