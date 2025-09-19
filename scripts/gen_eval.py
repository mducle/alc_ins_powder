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
from gen_training import sanitize, gen_spec, get_struct, fetch_mp_ids, myMPDoc
 
# ============== Config ==============
MATPROJ_APIKEY = "***REMOVED***"
BLACKLIST = {"Ac", "Th", "Pa", "U", "Np", "Pu"}

def plot_fig(y_true, y_pred, outdir, idstr, model_name, speedup):
    vmin = 0.0
    vmax = np.nanpercentile(np.concatenate([y_true.ravel(), y_pred.ravel()]), 99.5)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"Brute-force {idstr}")
    axes[0].set_xlabel("Energy bin")
    axes[0].set_ylabel("|Q| bin")
    fig.colorbar(im0, ax=axes[0]).ax.set_ylabel("Intensity")
 
    im1 = axes[1].imshow(y_pred, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"ML ({model_name}) x{speedup:.1f}")
    axes[1].set_xlabel("Energy bin")
    axes[1].set_ylabel("|Q| bin")
    fig.colorbar(im1, ax=axes[1]).ax.set_ylabel("Intensity")
 
    os.makedirs("checkpoints", exist_ok=True)
    fig.savefig(f"{outdir}/pred_vs_gt{idstr}.png", dpi=180)
    plt.close(fig)
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--speed-mpid", type=str, default="mp-8566")
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--dir", type=str, default="")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # ========= Load checkpoint =========
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = ckpt["model"]
    state_dict = ckpt["state_dict"]
    outdir = os.path.dirname(args.ckpt)
    outdir = args.dir if args.dir else (outdir if outdir else "checkpoints")
 
    # Load scaler params for inverse transform
    sy_scale = ckpt["sy_scale"].cpu().numpy()
    sy_offset = ckpt["sy_offset"].cpu().numpy()
    in_sz, out_sz, npts = (ckpt["in_sz"], ckpt["out_sz"], ckpt["npts"])
 
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
 
    # ========= Test one or more sample for speedup =========
    mp_ids = fetch_mp_ids(MATPROJ_APIKEY, limit=args.num, num_elements=(1, 3),
                          nsites_max=20, blacklist=BLACKLIST, oversample=10, seed=args.seed)
    mp_ids = [args.speed_mpid] + mp_ids
 
    def gen_single(mp_id, input_size=(20, 20), output_size=(100, 200), npts=(200, 1000)):
        """Generate one pair of coarse/high-res spectra"""
        struct = get_struct(mp_id)
 
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
        t0 = time.time()
        z_high = gen_spec(output_size, npts[1], fc, dw, tt)
        t1 = time.time()
        # Coarse-resolution spectrum
        z_coarse = gen_spec(input_size, npts[0], fc, dw, tt)
        t2 = time.time() 
        return z_coarse.flatten(), z_high.flatten(), t1-t0, t2-t1, formula
 
    # ====== Measure runtime ======
    ok_ids = []
    for mp_id in mp_ids:
        if len(ok_ids) > (args.num+1):
            break
        try:
            print(f"\n=== Speed test on {mp_id} ===")
            zc_vec, zh_vec, t_brute, t_coarse, formula = gen_single(mp_id, input_size=in_sz, output_size=out_sz, npts=npts)
         
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
            speedup = t_brute / (t_coarse + t_ml)
            print(f"Brute force: {t_brute:.2f} s")
            print(f"Coarse     : {t_coarse:.2f} s")
            print(f"ML infer   : {t_ml*1000:.2f} ms")
            print(f"Speedup    : {speedup:.1f}×")
         
            # ====== Inverse scaling ======
            y_scaled_np = y_scaled.cpu().numpy().reshape(1, -1)
            y_unscaled = (y_scaled_np - sy_offset) / sy_scale
            y_pred = np.expm1(y_unscaled).reshape(*out_sz)
            y_true = zh_vec.reshape(*out_sz)
         
            # ====== Plotting ======
            plot_fig(y_true, y_pred, outdir, "" if mp_id == args.speed_mpid else f"_{mp_id}-{formula}", model_name, speedup)
            ok_ids.append(mp_id)
        except Exception as e:
            print(f"✗ {mp_id}: {type(e).__name__}: {e}")
 
 
if __name__ == "__main__":
    main()
