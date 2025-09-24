# train.py
import argparse, os, time, random, csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
from model import SRCNN, PowderUNet, FNO2d, Hybrid_WFDN_FNO, Hybrid_GhostWFDN_FNO

# ============== Config ==============
MATPROJ_APIKEY = 'LvxElbvFT1ttFZLGiLgvtWPxN442GVdr'
BLACKLIST = {"Ac", "Th", "Pa", "U", "Np", "Pu"}
SEED = 1001
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---------- utils ----------
def sanitize(arr, clip_pct=99.9, nonneg=True):
    """Clip outliers and ensure non-negative values"""
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

def fetch_mp_ids(api_key, limit=80, num_elements=(1, 3), nsites_max=20,
                 blacklist=BLACKLIST, oversample=10, seed=SEED):
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
        random.seed(seed)
        for idd in random.choices(range(len(docs)), k=hard_cap*2):
            d = docs[idd]
            if d.nsites is not None and d.nsites > nsites_max:
                continue
            elems = {str(e) for e in (d.elements or [])}
            if elems & bl:
                continue
            ids.append(d.material_id)
            if len(ids) >= hard_cap:
                break
    return ids

def gen_spec(output_size, npts, fc, dw, tt):
    qbins_h = np.linspace(0, 6, output_size[0]+1) * ureg('1 / angstrom')
    qc_h = (qbins_h[:-1] + qbins_h[1:]) / 2
    ebins_h = np.linspace(0, 60, output_size[1]+1) * ureg('meV')
    z_high = np.empty((len(qc_h), len(ebins_h) - 1))
    for i, q in enumerate(qc_h):
        spec = sample_sphere_structure_factor(fc, q, dw=dw, temperature=tt,
                                              sampling='golden', npts=npts, jitter=True,
                                              energy_bins=ebins_h / 1.2)
        z_high[i, :] = spec.y_data.magnitude
    return z_high

def get_struct(mp_id):
    structnpy = f'janus_results/{mp_id}-struct.npy'
    if os.path.exists(structnpy):
        struct = ase.Atoms(np.load(structnpy, allow_pickle=True).tolist())
    else:
        with MPRester(api_key=MATPROJ_APIKEY) as mp:
            struct = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(
                mp.get_structure_by_material_id(mp_id)
            )
        np.save(structnpy, struct)
    return struct

def gen_single(mp_id, do_plot=False, no_load=False, input_size=(20,20), output_size=(100,200), npts=(200,1000)):
    innpy = f'janus_results/{mp_id}-powder-{input_size[0]}x{input_size[1]}-np{npts[0]}.npy'
    outnpy = f'janus_results/{mp_id}-powder-{output_size[0]}x{output_size[1]}-np{npts[1]}.npy'
    has_in, has_out = (False, False)
    if not no_load:
        if os.path.exists(innpy):
            z_coarse = np.load(innpy)
            if z_coarse.shape == input_size:
                has_in = True
        if os.path.exists(outnpy):
            z_high = np.load(outnpy)
            if z_high.shape == output_size:
                has_out = True
        if has_in and has_out:
            return z_coarse.flatten(), z_high.flatten()

    struct = get_struct(mp_id)

    elems = {a.symbol for a in struct}
    if elems & BLACKLIST:
        raise ValueError(f"contains blacklisted elements: {elems & BLACKLIST}")

    formula = struct.get_chemical_formula()
    print(f"Processing {mp_id} ({formula})")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(f'janus_results/{formula}-phonopy.yml') or not os.path.exists(f'janus_results/{formula}-force_constants.hdf5'):
        ph = Phonons(
            struct=struct.copy(), arch="mace_mp", device=device_str, model="small",
            calc_kwargs={"default_dtype": "float64"},
            supercell=[2, 2, 2], displacement=0.01, temp_step=2.0, temp_min=0.0, temp_max=2000.0,
            minimize=True,
            minimize_kwargs={"filter_kwargs": {"hydrostatic_strain": False}, "fmax": 0.1, "optimizer": "MDMin"},
            force_consts_to_hdf5=True, plot_to_file=False, symmetrize=False, write_full=True, write_results=True,
        )
        ph.calc_force_constants()

    fc = euphonic.ForceConstants.from_phonopy(
        summary_name=f'janus_results/{formula}-phonopy.yml',
        fc_name=f'janus_results/{formula}-force_constants.hdf5'
    )

    tt = 5 * ureg('K')
    dw = _get_debye_waller(tt, fc)

    if not has_out:
        z_high = gen_spec(output_size, npts[1], fc, dw, tt)
    if not has_in:
        z_coarse = gen_spec(input_size, npts[0], fc, dw, tt)

    if not (np.all(np.isfinite(z_coarse)) and np.all(np.isfinite(z_high))):
        raise ValueError("non-finite values in z_coarse/z_high")
    if not no_load:
        if not has_out: np.save(outnpy, z_high)
        if not has_in: np.save(innpy, z_coarse)

    return z_coarse.flatten(), z_high.flatten()

def collect_samples(mp_ids, need_n=None, input_size=(20,20), output_size=(100,200), npts=(200,1000)):
    need_n = need_n or len(mp_ids)
    inputs, targets, ok_ids, errors = [], [], [], []
    for mpid in mp_ids:
        if len(ok_ids) >= need_n:
            break
        try:
            inp, tgt = gen_single(mpid, do_plot=False, input_size=input_size, output_size=output_size, npts=npts)
            if not (np.all(np.isfinite(inp)) and np.all(np.isfinite(tgt))):
                raise ValueError("non-finite values in sample")
            inputs.append(inp); targets.append(tgt); ok_ids.append(mpid)
            print(f"✓ {mpid}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"✗ {mpid}: {msg}")
            errors.append((mpid, msg))
    print(f"[summary] success: {len(ok_ids)}, failed: {len(errors)}")
    return inputs, targets, ok_ids, errors

# ---- Physics-aware losses ----
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

def combined_loss(pred_hr, gt_hr, w_pix=1.0, w_grad=0.2, w_fft=0.1, q_axis_last=True):
    lpix = nn.functional.l1_loss(pred_hr, gt_hr)
    lgrad = grad_loss(pred_hr, gt_hr) if w_grad>0 else 0.0
    if w_fft>0:
        if q_axis_last:
            lfft = fourier_q_loss(pred_hr, gt_hr, dim_q=-1)
        else:
            lfft = fourier_q_loss(pred_hr.transpose(-1, -2), gt_hr.transpose(-1, -2), dim_q=-1)
    else:
        lfft = 0.0
    return w_pix*lpix + w_grad*lgrad + w_fft*lfft

# ---- Train main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['srcnn', 'unet', 'fno', 'wfdn_fno', 'ghost'], required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--speed-mpid', type=str, default='mp-8566')
    parser.add_argument('--input-size', type=str, default='20x20')
    parser.add_argument('--output-size', type=str, default='100x200')
    parser.add_argument('--npts', type=str, default='200-1000')
    args = parser.parse_args()

    in_sz = tuple(int(v) for v in args.input_size.split('x'))
    out_sz = tuple(int(v) for v in args.output_size.split('x'))
    npts = tuple(int(v) for v in args.npts.split('-'))

    mp_ids = fetch_mp_ids(MATPROJ_APIKEY, limit=args.limit, num_elements=(1, 3),
                          nsites_max=20, blacklist=BLACKLIST, oversample=10)
    inputs, targets, ok_ids, errors = collect_samples(mp_ids, need_n=args.limit,
                                                     input_size=in_sz, output_size=out_sz, npts=npts)

    X = sanitize(np.vstack(inputs), clip_pct=99.9, nonneg=True)
    Y = sanitize(np.vstack(targets), clip_pct=99.9, nonneg=True)
    X = np.log1p(X); Y = np.log1p(Y)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xs = sx.fit_transform(X).reshape(-1, 1, in_sz[0], in_sz[1])
    Ys = sy.fit_transform(Y).reshape(-1, 1, out_sz[0], out_sz[1])
    X_tr, X_val, y_tr, y_val = train_test_split(Xs, Ys, test_size=0.2, random_state=0)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                            torch.from_numpy(y_tr).float()),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(),
                                          torch.from_numpy(y_val).float()),
                            batch_size=args.batch_size, shuffle=False)

    if args.model == 'srcnn':
        net = SRCNN(scale_factor=(out_sz[0]/in_sz[0], out_sz[1]/in_sz[1]))
    elif args.model == 'unet':
        net = PowderUNet(scale_factor=(out_sz[0]/in_sz[0], out_sz[1]/in_sz[1]))
    elif args.model == 'wfdn_fno':
        net = Hybrid_WFDN_FNO(in_channels=1, base_channels=64, num_wfdn=4, num_fno=2, output_size=out_sz)
    elif args.model == 'ghost':
        net = Hybrid_GhostWFDN_FNO(in_channels=1, base_channels=64, num_wfdn=4, num_fno=2, output_size=out_sz)
    else:
        net = FNO2d(modes1=20, modes2=10, width=64, output_size=out_sz)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device).float()

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    best_val = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/val_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch','val_loss','psnr','grad_loss','fourier_q'])

    for ep in range(1, args.epochs+1):
        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            base = nn.functional.interpolate(xb, size=out_sz, mode='bicubic', align_corners=False)
            pred = base + net(xb)
            loss = combined_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        # ===== val =====
        net.eval()
        with torch.no_grad():
            val_losses, psnrs, grads, ffts = [], [], [], []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                base = nn.functional.interpolate(xb, size=out_sz, mode='bicubic', align_corners=False)
                pred = base + net(xb)
                val_losses.append(float(nn.functional.l1_loss(pred, yb)))
                psnrs.append(float(20*np.log10(1.0) - 10*np.log10(((pred-yb)**2).mean().item())))
                grads.append(float(grad_loss(pred, yb)))
                ffts.append(float(fourier_q_loss(pred, yb, dim_q=-1)))
            vloss = np.mean(val_losses)
            vpsnr = np.mean(psnrs)
            vgrad = np.mean(grads)
            vfft  = np.mean(ffts)
            print(f"Epoch {ep} | Val L1 {vloss:.4f} | PSNR {vpsnr:.2f} dB | grad {vgrad:.4f} | fourier {vfft:.4f}")
            with open('checkpoints/val_metrics.csv', 'a', newline='') as f:
                w = csv.writer(f); w.writerow([ep, vloss, vpsnr, vgrad, vfft])
            sched.step(vloss)
            if vloss < best_val:
                best_val = vloss
                torch.save({
                    'model': args.model,
                    'state_dict': net.state_dict(),
                    'sx_min': torch.tensor(sx.data_min_.copy(), dtype=torch.float64, device='cpu'),
                    'sx_max': torch.tensor(sx.data_max_.copy(), dtype=torch.float64, device='cpu'),
                    'sy_min': torch.tensor(sy.data_min_.copy(), dtype=torch.float64, device='cpu'),
                    'sy_max': torch.tensor(sy.data_max_.copy(), dtype=torch.float64, device='cpu'),
                    'sy_scale': torch.tensor(sy.scale_.copy(), dtype=torch.float64, device='cpu'),
                    'sy_offset': torch.tensor(sy.min_.copy(), dtype=torch.float64, device='cpu'),
                    'in_sz': in_sz,
                    'out_sz': out_sz,
                    'npts': npts 
                }, f'checkpoints/{args.model}.pt')

if __name__ == '__main__':
    main()
