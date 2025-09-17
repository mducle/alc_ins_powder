import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
 
# ---- Materials Project + physics generation ----
from mp_api.client import MPRester
import pymatgen.io.ase
import ase
from janus_core.calculations.phonons import Phonons
import euphonic
from euphonic.cli.utils import _get_debye_waller
from euphonic.powder import sample_sphere_structure_factor
from euphonic import ureg
 
# ---- Models ----
import sys
sys.path.append(os.path.dirname(__file__))
from model import SRCNN, PowderUNet, FNO2d, Hybrid_WFDN_FNO
 
# ============== Config ==============
MATPROJ_APIKEY = "LvxElbvFT1ttFZLGiLgvtWPxN442GVdr"
BLACKLIST = {"Ac", "Th", "Pa", "U", "Np", "Pu"}
SEED = 1001
import random
 
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
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input-size", type=str, default="20x20")
    parser.add_argument("--output-size", type=str, default="100x200")
    parser.add_argument("--npts", type=str, default="200-1000")
    parser.add_argument("--speed-mpid", type=str, default="mp-8566")
    args = parser.parse_args()
 
    in_sz = tuple(int(v) for v in args.input_size.split("x"))
    out_sz = tuple(int(v) for v in args.output_size.split("x"))
    npts = tuple(int(v) for v in args.npts.split("-"))
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # ========= Load checkpoint =========
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = ckpt["model"]
    state_dict = ckpt["state_dict"]
 
    # Load scaler params for inverse transform
    sy_scale = ckpt["sy_scale"].cpu().numpy()
    sy_offset = ckpt["sy_offset"].cpu().numpy()
 
    # ========= Build model =========
    if model_name == "srcnn":
        net = SRCNN(scale_factor=(out_sz[0] / in_sz[0], out_sz[1] / in_sz[1]))
    elif model_name == "unet":
        net = PowderUNet(scale_factor=(out_sz[0] / in_sz[0], out_sz[1] / in_sz[1]))
    elif model_name == "wfdn_fno":
        net = Hybrid_WFDN_FNO(
            in_channels=1, base_channels=64, num_wfdn=4, num_fno=2, output_size=out_sz
        )
    else:
        net = FNO2d(modes1=20, modes2=10, width=64, output_size=out_sz)
 
    net.load_state_dict(state_dict)
    net = net.to(device).eval()
 
    # ========= Test one sample for speedup =========
    speed_mpid = args.speed_mpid
 
    def gen_single(mp_id, input_size=(20, 20), output_size=(100, 200), npts=(200, 1000)):
        """Generate one pair of coarse/high-res spectra"""
        structnpy = f"janus_results/{mp_id}-struct.npy"
        if os.path.exists(structnpy):
            struct = ase.Atoms(np.load(structnpy, allow_pickle=True).tolist())
        else:
            with MPRester(api_key=MATPROJ_APIKEY) as mp:
                struct = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(
                    mp.get_structure_by_material_id(mp_id)
                )
            np.save(structnpy, struct)
 
        elems = {a.symbol for a in struct}
        if elems & BLACKLIST:
            raise ValueError(f"contains blacklisted elements: {elems & BLACKLIST}")
 
        formula = struct.get_chemical_formula()
        print(f"Processing {mp_id} ({formula})")
 
        # Run phonon calculation if force constants are missing
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(
            f"janus_results/{formula}-phonopy.yml"
        ) or not os.path.exists(f"janus_results/{formula}-force_constants.hdf5"):
            ph = Phonons(
                struct=struct.copy(),
                arch="mace_mp",
                device=device_str,
                model="small",
                calc_kwargs={"default_dtype": "float64"},
                supercell=[2, 2, 2],
                displacement=0.01,
                temp_step=2.0,
                temp_min=0.0,
                temp_max=2000.0,
                minimize=True,
                minimize_kwargs={
                    "filter_kwargs": {"hydrostatic_strain": False},
                    "fmax": 0.1,
                    "optimizer": "MDMin",
                },
                force_consts_to_hdf5=True,
                plot_to_file=False,
                symmetrize=False,
                write_full=True,
                write_results=True,
            )
            ph.calc_force_constants()
 
        fc = euphonic.ForceConstants.from_phonopy(
            summary_name=f"janus_results/{formula}-phonopy.yml",
            fc_name=f"janus_results/{formula}-force_constants.hdf5",
        )
 
        tt = 5 * ureg("K")
        dw = _get_debye_waller(tt, fc)
 
        # High-resolution spectrum
        qbins_h = np.linspace(0, 6, output_size[0] + 1) * ureg("1 / angstrom")
        qc_h = (qbins_h[:-1] + qbins_h[1:]) / 2
        ebins_h = np.linspace(0, 60, output_size[1] + 1) * ureg("meV")
 
        z_high = np.empty((len(qc_h), len(ebins_h) - 1))
        for i, q in enumerate(qc_h):
            spec = sample_sphere_structure_factor(
                fc,
                q,
                dw=dw,
                temperature=tt,
                sampling="golden",
                npts=npts[1],
                jitter=True,
                energy_bins=ebins_h / 1.2,
            )
            z_high[i, :] = spec.y_data.magnitude
 
        # Coarse-resolution spectrum
        qbins_c = np.linspace(0, 6, input_size[0] + 1) * ureg("1 / angstrom")
        qc_c = (qbins_c[:-1] + qbins_c[1:]) / 2
        ebins_c = np.linspace(0, 60, input_size[1] + 1) * ureg("meV")
 
        z_coarse = np.empty((len(qc_c), len(ebins_c) - 1))
        for i, q in enumerate(qc_c):
            spec_c = sample_sphere_structure_factor(
                fc,
                q,
                dw=dw,
                temperature=tt,
                sampling="golden",
                npts=npts[0],
                jitter=True,
                energy_bins=ebins_c / 1.2,
            )
            z_coarse[i, :] = spec_c.y_data.magnitude
 
        return z_coarse.flatten(), z_high.flatten()
 
    # ====== Measure runtime ======
    print(f"\n=== Speed test on {speed_mpid} ===")
    t0 = time.time()
    zc_vec, zh_vec = gen_single(speed_mpid, input_size=in_sz, output_size=out_sz, npts=npts)
    t_brute = time.time() - t0
 
    zc = sanitize(zc_vec, clip_pct=99.9, nonneg=True)
    zc = np.log1p(zc).reshape(1, -1)
 
    # Normalize to [0,1] using training stats
    zc_scaled = (zc - ckpt["sx_min"].cpu().numpy()) / (
        ckpt["sx_max"].cpu().numpy() - ckpt["sx_min"].cpu().numpy() + 1e-12
    )
 
    x = torch.from_numpy(zc_scaled.reshape(1, 1, in_sz[0], in_sz[1])).float().to(device)
 
    with torch.no_grad():
        t1 = time.time()
        base = nn.functional.interpolate(x, size=out_sz, mode="bilinear", align_corners=False)
        res = net(x)
        y_scaled = (base + res).clamp(0.0, 1.0)
        t_ml = time.time() - t1
 
    print(f"Brute force: {t_brute:.2f} s")
    print(f"ML infer  : {t_ml*1000:.2f} ms")
    if t_ml > 0:
        print(f"Speedup   : {t_brute/t_ml:.1f}×")
 
    # ====== Inverse scaling ======
    y_scaled_np = y_scaled.cpu().numpy().reshape(1, -1)
    y_unscaled = (y_scaled_np - sy_offset) / sy_scale
    y_pred = np.expm1(y_unscaled).reshape(*out_sz)
    y_true = zh_vec.reshape(*out_sz)
 
    # ====== Plotting ======
    vmin = 0.0
    vmax = np.nanpercentile(np.concatenate([y_true.ravel(), y_pred.ravel()]), 99.5)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Brute-force")
    axes[0].set_xlabel("Energy bin")
    axes[0].set_ylabel("|Q| bin")
    fig.colorbar(im0, ax=axes[0]).ax.set_ylabel("Intensity")
 
    im1 = axes[1].imshow(y_pred, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"ML ({model_name})")
    axes[1].set_xlabel("Energy bin")
    axes[1].set_ylabel("|Q| bin")
    fig.colorbar(im1, ax=axes[1]).ax.set_ylabel("Intensity")
 
    os.makedirs("checkpoints", exist_ok=True)
    fig.savefig("checkpoints/pred_vs_gt.png", dpi=180)
    plt.close(fig)
 
    print("Saved: checkpoints/pred_vs_gt.png")
 
 
if __name__ == "__main__":
    main()
