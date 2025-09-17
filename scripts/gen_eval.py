# eval.py
import argparse, os, time, random, csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---- MP + physics generation ----
from mp_api.client import MPRester
import pymatgen.io.ase
import ase
from janus_core.calculations.phonons import Phonons
import euphonic
from euphonic.cli.utils import _get_debye_waller
from euphonic.powder import sample_sphere_structure_factor
from euphonic import ureg

# ---- model ----
import sys
sys.path.append(os.path.dirname(__file__))
from model3 import SRCNN, PowderUNet, FNO2d, Hybrid_WFDN_FNO

# ============== Config ==============
MATPROJ_APIKEY = 'LvxElbvFT1ttFZLGiLgvtWPxN442GVdr'
BLACKLIST = {"Ac", "Th", "Pa", "U", "Np", "Pu"}
SEED = 1001
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

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

class myMPDoc():
    def __init__(self, m_id, ele, ns):
        self.material_id, self.elements, self.nsites = (m_id, ele, ns)

def fetch_mp_ids(api_key, limit=40, num_elements=(1, 3), nsites_max=20, blacklist=BLACKLIST, oversample=6):
    ids = []
    bl = {str(x) for x in blacklist}
    hard_cap = limit * max(1, int(oversample))
    with MPRester(api_key=api_key) as mpr:
        docs_fname = f'mpr_docs_nelem{num_elements}.npy'
        if os.path.exists(docs_fname):
            docs = np.load(docs_fname, allow_pickle=True)
        else:
            docs = mpr.materials.summary.search(
                fields=["material_id", "elements", "nsites"],
                num_elements=num_elements,
                is_stable=True
            )
            np.save(docs_fname, [myMPDoc(d.material_id, d.elements, d.nsites) for d in docs])
        random.seed(2025)
        for idd in random.choices(range(len(docs)), k=hard_cap*2):
            d = docs[idd]
            if d.nsites is not None and d.nsites > nsites_max: continue
            elems = {str(e) for e in (d.elements or [])}
            if elems & bl: continue
            ids.append(d.material_id)
            if len(ids) >= hard_cap: break
    return ids

def gen_single(mp_id, do_plot=False, no_load=False, input_size=(20,20), output_size=(100,200), npts=(200,1000)):
    innpy = f'janus_results/{mp_id}-powder-{input_size[0]}x{input_size[1]}-np{npts[0]}.npy'
    outnpy = f'janus_results/{mp_id}-powder-{output_size[0]}x{output_size[1]}-np{npts[1]}.npy'
    has_in, has_out = (False, False)
    if not no_load:
        if os.path.exists(innpy):
            z_coarse = np.load(innpy)
            if z_coarse.shape == input_size: has_in = True
        if os.path.exists(outnpy):
            z_high = np.load(outnpy)
            if z_high.shape == output_size: has_out = True
        if has_in and has_out:
            return z_coarse.flatten(), z_high.flatten()

    structnpy = f'janus_results/{mp_id}-struct.npy'
    if os.path.exists(structnpy):
        struct = ase.Atoms(np.load(structnpy, allow_pickle=True).tolist())
    else:
        with MPRester(api_key=MATPROJ_APIKEY) as mp:
            struct = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(mp.get_structure_by_material_id(mp_id))
        np.save(structnpy, struct)

    elems = {a.symbol for a in struct}
    if elems & BLACKLIST:
        raise ValueError(f"contains blacklisted elements: {elems & BLACKLIST}")

    formula = struct.get_chemical_formula()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(f'janus_results/{formula}-phonopy.yml') or not os.path.exists(f'janus_results/{formula}-force_constants.hdf5'):
        ph = Phonons(
            struct=struct.copy(), arch="mace_mp", device=device_str, model="small",
            calc_kwargs={"default_dtype": "float64"},
            supercell=[2, 2, 2], displacement=0.01, temp_step=2.0, temp_min=0.0, temp_max=2000.0,
            minimize=True, minimize_kwargs={"filter_kwargs": {"hydrostatic_strain": False}, "fmax": 0.1, "optimizer": "MDMin"},
            force_consts_to_hdf5=True, plot_to_file=False, symmetrize=False, write_full=True, write_results=True,
        )
        ph.calc_force_constants()

    fc = euphonic.ForceConstants.from_phonopy(
        summary_name=f'janus_results/{formula}-phonopy.yml',
        fc_name=f'janus_results/{formula}-force_constants.hdf5'
    )

    tt = 5 * ureg('K')
    dw = _get_debye_waller(tt, fc)

    qbins_h = np.linspace(0, 6, output_size[0]+1) * ureg('1 / angstrom')
    qc_h = (qbins_h[:-1] + qbins_h[1:]) / 2
    ebins_h = np.linspace(0, 60, output_size[1]+1) * ureg('meV')
    if not has_out:
        z_high = np.empty((len(qc_h), len(ebins_h) - 1))
        for i, q in enumerate(qc_h):
            spec = sample_sphere_structure_factor(fc, q, dw=dw, temperature=tt, sampling='golden', npts=npts[1], jitter=True, energy_bins=ebins_h / 1.2)
            z_high[i, :] = spec.y_data.magnitude

    qbins_c = np.linspace(0, 6, input_size[0]+1) * ureg('1 / angstrom')
    qc_c = (qbins_c[:-1] + qbins_c[1:]) / 2
    ebins_c = np.linspace(0, 60, input_size[1]+1) * ureg('meV')
    z_coarse = np.empty((len(qc_c), len(ebins_c) - 1))
    if not has_in:
        for i, q in enumerate(qc_c):
            spec_c = sample_sphere_structure_factor(fc, q, dw=dw, temperature=tt, sampling='golden', npts=npts[0], jitter=True, energy_bins=ebins_c / 1.2)
            z_coarse[i, :] = spec_c.y_data.magnitude

    if not (np.all(np.isfinite(z_coarse)) and np.all(np.isfinite(z_high))):
        raise ValueError("non-finite values in z_coarse/z_high")
    if not no_load:
        if not has_out: np.save(outnpy, z_high)
        if not has_in: np.save(innpy, z_coarse)

    return z_coarse.flatten(), z_high.flatten()

# ---- Physics-aware metrics (for reporting only) ----
def grad_loss(pred, gt):
    px = pred[..., :, 1:] - pred[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    gx = gt  [..., :, 1:] - gt  [..., :, :-1]
    gy = gt  [..., 1:, :] - gt  [..., :-1, :]
    return (px - gx).abs().mean() + (py - gy).abs().mean()

def fourier_q_loss(pred, gt, dim_q=-1):
    pf = torch.fft.rfft(pred, dim=dim_q).abs()
    gf = torch.fft.rfft(gt,   dim=dim_q).abs()
    return (pf - gf).abs().mean()

def psnr(pred, gt, data_range=None):
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt, dtype=np.float64)
    if data_range is None:
        data_range = np.nanmax(gt) - np.nanmin(gt) + 1e-12
    mse = np.nanmean((pred - gt)**2)
    if mse <= 1e-20: return 99.0
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

def peak_metrics_1d(y_pred, y_true, x_axis, topk=5):
    def peaks(y):
        y = np.asarray(y)
        m = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        idx = np.where(m)[0] + 1
        if len(idx)==0: return np.array([], int)
        idx = idx[np.argsort(y[idx])[::-1][:topk]]
        return np.sort(idx)
    ip, it = peaks(y_pred), peaks(y_true)
    if len(ip)==0 or len(it)==0:
        return np.nan, np.nan
    used = set()
    pos_err, amp_err = [], []
    for ii in ip:
        j = np.argmin(np.abs(it - ii))
        jj = int(it[j])
        if jj in used: continue
        used.add(jj)
        pos_err.append(abs(x_axis[ii] - x_axis[jj]))
        amp_err.append(abs(y_pred[ii] - y_true[jj]))
    if len(pos_err)==0:
        return np.nan, np.nan
    return float(np.nanmean(pos_err)), float(np.nanmean(amp_err))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['srcnn', 'unet', 'fno','wfdn_fno'], required=True)
    parser.add_argument('--ckpt', type=str, default='checkpoints/unet.pt')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--input-size', type=str, default='50x100')
    parser.add_argument('--output-size', type=str, default='100x200')
    parser.add_argument('--npts', type=str, default='200-1000')
    args = parser.parse_args()

    in_sz = tuple(int(v) for v in args.input_size.split('x'))
    out_sz = tuple(int(v) for v in args.output_size.split('x'))
    npts = tuple(int(v) for v in args.npts.split('-'))

    # ---- build model skeleton to load weights ----
    if args.model == 'srcnn':
        sf = (out_sz[0]/in_sz[0], out_sz[1]/in_sz[1])
        net = SRCNN(scale_factor=sf)
    elif args.model == 'unet':
        sf = (out_sz[0]/in_sz[0], out_sz[1]/in_sz[1])
        net = PowderUNet(scale_factor=sf, nf=64, n_blocks=16, kq=5, ke=5)
    elif args.model == 'wfdn_fno':
        net = Hybrid_WFDN_FNO(in_channels=1, base_channels=64, num_wfdn=4, num_fno=2, output_size=out_sz)
    else:
        net = FNO2d(modes1=24, modes2=24, width=64, output_size=out_sz)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device).float()

    # ---- load checkpoint ----
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt['state_dict'], strict=True)
    # saved as numpy arrays in train.py
    sx_min, sx_max = ckpt['sx_min'].cpu().numpy(), ckpt['sx_max'].cpu().numpy()
    sy_min, sy_max = ckpt['sy_min'].cpu().numpy(), ckpt['sy_max'].cpu().numpy()

    # ---- data for eval ----
    mp_ids = fetch_mp_ids(MATPROJ_APIKEY, limit=args.limit, num_elements=(1, 3), nsites_max=20, blacklist=BLACKLIST, oversample=10)

    inputs, targets, ok_ids, errors = [], [], [], []
    for mpid in mp_ids[:args.limit]:
        try:
            inp, tgt = gen_single(mpid, do_plot=False, input_size=in_sz, output_size=out_sz, npts=npts)
            inputs.append(inp); targets.append(tgt); ok_ids.append(mpid)
            print(f"✓ {mpid}")
        except Exception as e:
            print(f"✗ {mpid}: {e}")

    if len(inputs) == 0:
        raise RuntimeError("No evaluation samples.")

    # ---- helpers for scaling ----
    def scale_with_minmax(arr, mn, mx):
        arr2d = arr.reshape(arr.shape[0], -1)
        return (arr2d - mn) / (mx - mn + 1e-12)

    def inverse_minmax(arr2d, mn, mx):
        return arr2d * (mx - mn + 1e-12) + mn

    # ---- run inference & metrics over N samples, also time brute vs ML per sample ----
    X = np.log1p(sanitize(np.vstack(inputs), clip_pct=99.9, nonneg=True)).reshape(-1, 1, in_sz[0], in_sz[1])
    Y = np.log1p(sanitize(np.vstack(targets), clip_pct=99.9, nonneg=True)).reshape(-1, 1, out_sz[0], out_sz[1])

    Xs = scale_with_minmax(X, sx_min, sx_max).reshape(-1,1,in_sz[0],in_sz[1])
    Ys = scale_with_minmax(Y, sy_min, sy_max).reshape(-1,1,out_sz[0],out_sz[1])

    os.makedirs('eval_outputs', exist_ok=True)
    csv_path = 'eval_outputs/metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mp_id','brute_time_s','ml_time_ms','speedup','rmse','mae','psnr','grad_loss','fourier_q','peak_pos_mae_meV','peak_amp_mae'])

    speedups, rmses, maes, psnrs = [], [], [], []
    g_losses, f_losses, ppos_mae, pamp_mae = [], [], [], []

    net.eval()
    for i, mpid in enumerate(ok_ids):
        # Time brute force (force recompute)
        t0 = time.time()
        zc_vec, zh_vec = gen_single(mpid, do_plot=False, no_load=True, input_size=in_sz, output_size=out_sz, npts=npts)
        brute_t = time.time() - t0

        # ML prediction
        zc = np.log1p(sanitize(zc_vec, clip_pct=99.9, nonneg=True)).reshape(1, -1)
        zc_scaled = (zc - sx_min) / (sx_min*0 + (sx_max - sx_min) + 1e-12)  # safe denom
        xb = torch.from_numpy(zc_scaled.reshape(1,1,in_sz[0],in_sz[1])).float().to(device)

        with torch.no_grad():
            t1 = time.time()
            base = nn.functional.interpolate(xb, size=out_sz, mode='bicubic', align_corners=False)
            pred_scaled = base + net(xb)
            ml_t = time.time() - t1

        # inverse scale to real domain
        pred_log = inverse_minmax(pred_scaled.cpu().numpy().reshape(1,-1), sy_min, sy_max).reshape(1,1,*out_sz)
        gt_log   = inverse_minmax(Ys[i].reshape(1,-1),                      sy_min, sy_max).reshape(1,1,*out_sz)
        pred_lin = np.expm1(pred_log); gt_lin = np.expm1(gt_log)

        rmse = float(np.sqrt(np.mean((pred_lin - gt_lin)**2)))
        mae  = float(np.mean(np.abs(pred_lin - gt_lin)))
        psnr_v = psnr(pred_lin, gt_lin, data_range=np.nanpercentile(gt_lin, 99.5))

        pt_pred = torch.from_numpy(pred_lin).float()
        pt_gt   = torch.from_numpy(gt_lin).float()
        g_loss  = float(grad_loss(pt_pred, pt_gt))
        f_loss  = float(fourier_q_loss(pt_pred, pt_gt, dim_q=-1))

        W = out_sz[1]
        e_edges = np.linspace(0, 60.0, W+1)
        e_cent  = 0.5*(e_edges[:-1] + e_edges[1:])
        pos_mae, amp_mae = peak_metrics_1d(pred_lin[0,0,0,:], gt_lin[0,0,0,:], e_cent, topk=5)

        sp = brute_t / max(ml_t, 1e-6)

        speedups.append(sp); rmses.append(rmse); maes.append(mae); psnrs.append(psnr_v)
        g_losses.append(g_loss); f_losses.append(f_loss); ppos_mae.append(pos_mae); pamp_mae.append(amp_mae)

        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([mpid, f"{brute_t:.3f}", f"{ml_t*1000:.2f}", f"{sp:.1f}", f"{rmse:.6f}", f"{mae:.6f}", f"{psnr_v:.2f}", f"{g_loss:.6f}", f"{f_loss:.6f}", f"{pos_mae:.4f}", f"{amp_mae:.6f}"])

    # Summary
    print(f"\n=== EVAL SUMMARY on {len(ok_ids)} samples ===")
    print(f"Avg speedup: {np.nanmean(speedups):.1f}×  (median {np.nanmedian(speedups):.1f}×)")
    print(f"RMSE:  mean {np.nanmean(rmses):.4f} | median {np.nanmedian(rmses):.4f}")
    print(f"MAE :  mean {np.nanmean(maes):.4f} | median {np.nanmedian(maes):.4f}")
    print(f"PSNR: mean {np.nanmean(psnrs):.2f} dB | median {np.nanmedian(psnrs):.2f} dB")
    print(f"grad: mean {np.nanmean(g_losses):.4f}")
    print(f"fourier(Q): mean {np.nanmean(f_losses):.4f}")
    print(f"peak_pos_mae: mean {np.nanmean(ppos_mae):.4f} meV")
    print(f"peak_amp_mae: mean {np.nanmean(pamp_mae):.4f}")
    print(f"Saved per-sample metrics to {csv_path}")

    # Visualize one example
    os.makedirs('eval_outputs', exist_ok=True)
    idx = 0
    xb = torch.from_numpy(Xs[idx:idx+1]).float().to(device)
    with torch.no_grad():
        base = nn.functional.interpolate(xb, size=out_sz, mode='bicubic', align_corners=False)
        pred_scaled = base + net(xb)
    pred_log = inverse_minmax(pred_scaled.cpu().numpy().reshape(1,-1), sy_min, sy_max).reshape(1,1,*out_sz)
    gt_log   = inverse_minmax(Ys[idx:idx+1].reshape(1,-1), sy_min, sy_max).reshape(1,1,*out_sz)
    y_pred = np.expm1(pred_log)[0,0]; y_true = np.expm1(gt_log)[0,0]

    vmin = 0.0; vmax = np.nanpercentile(np.concatenate([y_true.ravel(), y_pred.ravel()]), 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis'); axes[0].set_title('Brute-force (GT)')
    im1 = axes[1].imshow(y_pred, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis'); axes[1].set_title(f'Prediction ({args.model})')
    for ax in axes:
        ax.set_xlabel('Energy bin'); ax.set_ylabel('|Q| bin')
    fig.colorbar(im0, ax=axes[0]).ax.set_ylabel('Intensity')
    fig.colorbar(im1, ax=axes[1]).ax.set_ylabel('Intensity')
    fig.savefig('eval_outputs/example_pred_vs_gt.png', dpi=180); plt.close(fig)
    print("Saved: eval_outputs/example_pred_vs_gt.png")

if __name__ == '__main__':
    main()
