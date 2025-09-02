
#!/usr/bin/env python3
"""
Analyse actin fibres: single image or batch.
Saves intermediate figures (original, cropped, binary, skeleton overlay) and a summary CSV.

Dependencies (install once):
    python -m pip install scikit-image skan pandas matplotlib tqdm

Example (single image):
    python analyse_actin_fibres.py --image /path/img.tif --pixel-um 0.065 --outdir results --preview

Example (batch):
    python analyse_actin_fibres.py --dir /path/images --pattern '*.tif' --pixel-um 0.065 --outdir results --preview

Key tunables:
    --crop-mode none|center|centroid|bbox|manual
    --crop-frac-w/--crop-frac-h, --crop-offset-x/--crop-offset-y  (center)
    --crop-frac (centroid), --pad-px (bbox), --crop-box X1 Y1 X2 Y2 (manual)
    --rolling-ball, --clahe-clip, --gaussian-sigma
    --thresh otsu|yen|local|<float>  plus --block-size, --offset
    --min-obj, --min-hole, --open-radius, --close-radius
    --min-leaf-px, --min-isolated-px, --remove-cycles-below
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage import io, color, exposure, filters, morphology, restoration, measure
from skimage.color import label2rgb
from skan import Skeleton, csr


# ----------------------------- Image IO & cropping -----------------------------

def load_grayscale_2d(path: Path,
                      prefer_channel: Optional[int] = None,
                      z_project: str = 'max') -> np.ndarray:
    """Return a 2D grayscale image from many shapes.
    Supports (H,W), (H,W,3|4), (C,H,W), (Z,H,W), (H,W,Z).
    If both channel & z exist, set prefer_channel to select a channel first.
    """
    img = io.imread(str(path))
    if img.ndim == 2:
        return img

    if img.ndim == 3:
        # Heuristics
        if img.shape[-1] in (3, 4):
            return color.rgb2gray(img[..., :3])
        if img.shape[0] in (2, 3, 4):  # channel-first
            c = 0 if prefer_channel is None else int(prefer_channel)
            return img[c]
        # Likely z-stack (Z,H,W)
        if z_project == 'max':
            return img.max(axis=0)
        elif z_project == 'mean':
            return img.mean(axis=0)
        elif z_project == 'median':
            return np.median(img, axis=0)
        else:
            raise ValueError("z_project must be 'max'|'mean'|'median'")
    raise ValueError(f"Unexpected image shape {img.shape} (path={path})")


def center_crop_box(shape, frac_w=0.7, frac_h=0.7, offset_px=(0, 0)):
    h, w = shape
    cw, ch = int(w*frac_w), int(h*frac_h)
    cx, cy = w//2 + int(offset_px[0]), h//2 + int(offset_px[1])
    x1, x2 = cx - cw//2, cx + cw//2
    y1, y2 = cy - ch//2, cy + ch//2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2)


def centroid_crop_box(img: np.ndarray, frac=0.7):
    thr = filters.threshold_otsu(img)
    mask = morphology.remove_small_objects(img > thr, 256)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, img.shape[1], img.shape[0])
    y, x = ys.mean(), xs.mean()
    h, w = img.shape
    cw, ch = int(w*frac), int(h*frac)
    x1, x2 = int(x - cw/2), int(x + cw/2)
    y1, y2 = int(y - ch/2), int(y + ch/2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2)


def largest_component_bbox(img: np.ndarray, pad_px=50):
    thr = filters.threshold_otsu(img)
    mask = morphology.remove_small_objects(img > thr, 256)
    lab = measure.label(mask)
    props = measure.regionprops(lab)
    if not props:
        return (0, 0, img.shape[1], img.shape[0])
    y1, x1, y2, x2 = max(props, key=lambda r: r.area).bbox
    y1, x1 = max(0, y1 - pad_px), max(0, x1 - pad_px)
    y2, x2 = min(img.shape[0], y2 + pad_px), min(img.shape[1], x2 + pad_px)
    return (x1, y1, x2, y2)


def apply_crop(img: np.ndarray, box: Optional[Tuple[int,int,int,int]]) -> np.ndarray:
    if box is None:
        return img
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


# ------------------------------ Preprocess & mask ------------------------------

def preprocess_image(img: np.ndarray,
                     rolling_ball_radius: int = 50,
                     clahe_clip: float = 0.01,
                     gaussian_sigma: float = 1.0) -> np.ndarray:
    # to float32
    work = img.astype(np.float32, copy=False)

    # background subtraction
    if rolling_ball_radius and rolling_ball_radius > 0:
        background = restoration.rolling_ball(image = work, radius = rolling_ball_radius)
        work = work - background

    # clip negatives, then rescale to [0,1] **before** CLAHE
    work = np.clip(work, 0, None)
    work = exposure.rescale_intensity(work, in_range='image', out_range=(0, 1))

    # CLAHE (expects [0,1] float)
    if clahe_clip and clahe_clip > 0:
        work = exposure.equalize_adapthist(work, clip_limit=clahe_clip)

    # light denoise
    if gaussian_sigma and gaussian_sigma > 0:
        # preserve_range True keeps values in [0,1]
        work = filters.gaussian(work, sigma=gaussian_sigma, preserve_range=True)

    return work


def binarize_image(img: np.ndarray,
                   thresh: str | float = "otsu",
                   block_size: int = 51,
                   offset: float = 0.0,
                   min_obj: int = 64,
                   min_hole: int = 64,
                   open_radius: int = 0,
                   close_radius: int = 0) -> np.ndarray:
    # Threshold
    if isinstance(thresh, (float, int)):
        binary = img > float(thresh)
    else:
        m = str(thresh).lower()
        if m == "otsu":
            t = filters.threshold_otsu(img)
            binary = img > t
        elif m == "yen":
            t = filters.threshold_yen(img)
            binary = img > t
        elif m == "local":
            timg = filters.threshold_local(img, block_size=block_size, offset=offset)
            binary = img > timg
        else:
            raise ValueError(f"Unknown thresh: {thresh}")

    # Morphological clean-up
    if open_radius and open_radius > 0:
        se = morphology.disk(open_radius)
        binary = morphology.binary_opening(binary, se)
    if close_radius and close_radius > 0:
        se = morphology.disk(close_radius)
        binary = morphology.binary_closing(binary, se)

    if min_obj and min_obj > 0:
        binary = morphology.remove_small_objects(binary, min_obj)
    if min_hole and min_hole > 0:
        binary = morphology.remove_small_holes(binary, min_hole)

    return binary


# ------------------------------- Skeleton & prune ------------------------------

def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    return morphology.skeletonize(binary)


def prune_short_branches_by_type(skeleton_bool: np.ndarray,
                                 min_leaf_px: int = 20,
                                 min_isolated_px: int = 0,
                                 remove_cycles_below_px: Optional[int] = None):
    """Prune short branches without breaking the backbone.
    Uses your schema:
        columns: 'branch-distance', 'branch-type'; row index = branch id.
        types: 0=isolated, 1=leaf, 2=backbone, 3=cycle
    Returns pruned skeleton and a small DataFrame of removed branches.
    """
    sk = Skeleton(skeleton_bool)
    df = csr.summarize(sk)

    len_col, type_col = 'branch-distance', 'branch-type'
    drop_mask = ((df[type_col] == 1) & (df[len_col] < min_leaf_px)) |                 ((df[type_col] == 0) & (df[len_col] < min_isolated_px))
    if remove_cycles_below_px is not None:
        drop_mask |= (df[type_col] == 3) & (df[len_col] < remove_cycles_below_px)

    pruned = skeleton_bool.copy()
    removed = df.loc[drop_mask, [len_col, type_col]].copy()
    for idx in df.index[drop_mask]:
        coords = sk.path_coordinates(int(idx))
        pruned[coords[:, 0], coords[:, 1]] = False
    return pruned, removed


# --------------------------------- Metrics ------------------------------------

def per_branch_table(skel: np.ndarray, intensity_image: Optional[np.ndarray], pixel_um: float) -> pd.DataFrame:
    sk = Skeleton(skel)
    df = csr.summarize(sk).copy()
    if 'mean-pixel-value' in df.columns:
        df.rename(columns={'mean-pixel-value': 'mean_intensity'}, inplace=True)
    df['length_px'] = df['branch-distance']
    df['length_um'] = df['length_px'] * pixel_um
    # straightness
    df['straightness'] = df['euclidean-distance'] / df['length_px'].clip(lower=1)
    return df


def summarize_image(df_branch: pd.DataFrame, roi_area_um2: float) -> Dict[str, Any]:
    stats = {
        'n_branches': len(df_branch),
        'min_len_um': df_branch['length_um'].min(),
        'max_len_um': df_branch['length_um'].max(),
        'mean_len_um': df_branch['length_um'].mean(),
        'median_len_um': df_branch['length_um'].median(),
        'std_len_um': df_branch['length_um'].std(),
        'total_len_um': df_branch['length_um'].sum(),
        'density_count_per_100um2': 100 * len(df_branch) / roi_area_um2,
        'density_length_um_per_100um2': 100 * df_branch['length_um'].sum() / roi_area_um2,
    }
    if 'mean_intensity' in df_branch:
        stats.update({
            'mean_intensity': df_branch['mean_intensity'].mean(),
            'min_intensity': df_branch['mean_intensity'].min(),
            'max_intensity': df_branch['mean_intensity'].max(),
        })
    return stats


# --------------------------------- Preview ------------------------------------

def save_preview(original: np.ndarray,
                 cropped: np.ndarray,
                 binary: np.ndarray,
                 skeleton: np.ndarray,
                 out_png: Path,
                 dilate_radius: int = 1):
    # Thicken skeleton for visibility
    sk_vis = morphology.dilation(skeleton, morphology.disk(max(0, int(dilate_radius)))) if dilate_radius > 0 else skeleton
    overlay = label2rgb(sk_vis.astype(int), image=cropped, alpha=0.8, bg_label=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
    axes[0].imshow(original, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(cropped, cmap='gray');  axes[1].set_title('Cropped');  axes[1].axis('off')
    axes[2].imshow(binary, cmap='gray');   axes[2].set_title('Binary');   axes[2].axis('off')
    axes[3].imshow(overlay);               axes[3].set_title('Skeleton overlay'); axes[3].axis('off')
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# --------------------------------- Workflow -----------------------------------

def compute_crop_box(img: np.ndarray, args) -> Optional[Tuple[int,int,int,int]]:
    if args.crop_mode == 'none':
        return None
    if args.crop_mode == 'manual':
        if args.crop_box is None:
            raise ValueError('crop_mode manual requires --crop-box X1 Y1 X2 Y2')
        return tuple(args.crop_box)
    if args.crop_mode == 'center':
        return center_crop_box(img.shape, args.crop_frac_w, args.crop_frac_h,
                               (args.crop_offset_x, args.crop_offset_y))
    if args.crop_mode == 'centroid':
        return centroid_crop_box(img, frac=args.crop_frac)
    if args.crop_mode == 'bbox':
        return largest_component_bbox(img, pad_px=args.pad_px)
    raise ValueError(f"Unknown crop_mode {args.crop_mode}")


def process_single_image(path: Path, args, outdir: Path) -> Dict[str, Any]:
    # Load & crop
    orig = load_grayscale_2d(path, prefer_channel=args.prefer_channel, z_project=args.z_project)
    crop_box = compute_crop_box(orig, args)
    cropped = apply_crop(orig, crop_box)

    # Preprocess & mask
    print("running preprocessing")
    enhanced = preprocess_image(cropped, args.rolling_ball, args.clahe_clip, args.gaussian_sigma)
    print("preprocessing completed")
    print("running binarization")
    binary = binarize_image(enhanced, args.thresh, args.block_size, args.offset,
                            args.min_obj, args.min_hole, args.open_radius, args.close_radius)
    print("binerization completed")
    # Skeletonize & prune
    skel = skeletonize_binary(binary)
    skel_prune, removed = prune_short_branches_by_type(
        skel, args.min_leaf_px, args.min_isolated_px,
        None if args.remove_cycles_below < 0 else args.remove_cycles_below
    )

    # Branch table & summary
    df_branch = per_branch_table(skel_prune, intensity_image=cropped, pixel_um=args.pixel_um)
    h, w = cropped.shape
    roi_area_um2 = h * w * (args.pixel_um ** 2)
    stats = summarize_image(df_branch, roi_area_um2)
    stats.update({
        'filename': path.name,
        'height_px': h, 'width_px': w,
        'crop_box': crop_box,
        'n_removed_branches': len(removed),
    })

    # Save outputs
    base = outdir / path.stem
    if args.preview:
        save_preview(orig, cropped, binary, skel_prune, base.with_name(base.stem + '_preview.png'), args.dilate_radius)
    if args.save_branch_csv:
        df_branch.assign(filename=path.name).to_csv(base.with_suffix('_per_branch.csv'), index=False)
    return stats


def run_batch(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    paths: List[Path] = []
    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(
    p for p in Path(args.dir).rglob('*.ome.tif')
    if not p.name.endswith('_overview.ome.tif')
)

    for p in tqdm(paths, desc='Processing'):
        try:
            stats = process_single_image(p, args, outdir)
            results.append(stats)
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(outdir / 'summary.csv', index=False)
    print(f"Done. Processed {len(results)} / {len(paths)} images. Results in {outdir}")


# ----------------------------------- CLI --------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Analyse actin fibres with skeletonisation and pruning')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', help='Path to a single image file')
    src.add_argument('--dir', help='Directory of images to process (use with --pattern)')

    p.add_argument('--pattern', default='*.tif', help="Glob pattern for --dir (default: '*.tif')")
    p.add_argument('--outdir', default='actin_results', help='Output directory for results')
    p.add_argument('--pixel-um', dest='pixel_um', type=float, default=0.1, help='Pixel size in µm')
    p.add_argument('--prefer-channel', type=int, default=None, help='If input is (C,H,W), pick this channel index')
    p.add_argument('--z-project', choices=['max','mean','median'], default='max', help='Projection for z-stacks')

    # Cropping
    p.add_argument('--crop-mode', choices=['none','center','centroid','bbox','manual'], default='none')
    p.add_argument('--crop-frac-w', type=float, default=0.7, help='Center crop fraction width')
    p.add_argument('--crop-frac-h', type=float, default=0.7, help='Center crop fraction height')
    p.add_argument('--crop-offset-x', type=int, default=0, help='Center crop offset (x)')
    p.add_argument('--crop-offset-y', type=int, default=0, help='Center crop offset (y)')
    p.add_argument('--crop-frac', type=float, default=0.7, help='Centroid crop fraction (both dims)')
    p.add_argument('--pad-px', type=int, default=50, help='Padding for bbox crop')
    p.add_argument('--crop-box', nargs=4, type=int, metavar=('X1','Y1','X2','Y2'), help='Manual crop rectangle')

    # Preprocess & threshold
    p.add_argument('--rolling-ball', type=int, default=50, help='Rolling-ball radius (0 to disable)')
    p.add_argument('--clahe-clip', type=float, default=0.01, help='CLAHE clip limit (0 to disable)')
    p.add_argument('--gaussian-sigma', type=float, default=1.0, help='Gaussian sigma (0 to disable)')
    p.add_argument('--thresh', default='otsu', help='otsu|yen|local|<float> threshold')
    p.add_argument('--block-size', type=int, default=51, help='Local threshold block size')
    p.add_argument('--offset', type=float, default=0.0, help='Local threshold offset')
    p.add_argument('--min-obj', type=int, default=64, help='Remove small objects (pixels)')
    p.add_argument('--min-hole', type=int, default=64, help='Remove small holes (pixels)')
    p.add_argument('--open-radius', type=int, default=0, help='Binary opening radius (0 to skip)')
    p.add_argument('--close-radius', type=int, default=0, help='Binary closing radius (0 to skip)')

    # Skeleton & prune
    p.add_argument('--min-leaf-px', type=int, default=20, help='Prune leaf branches shorter than this (px)')
    p.add_argument('--min-isolated-px', type=int, default=10, help='Prune isolated branches shorter than this (px)')
    p.add_argument('--remove-cycles-below', type=int, default=-1, help='Prune cycles shorter than this (px); -1 to keep')
    p.add_argument('--dilate-radius', type=int, default=1, help='Overlay skeleton dilation radius for visibility')

    # Output
    p.add_argument('--preview', action='store_true', help='Save preview PNGs')
    p.add_argument('--save-branch-csv', action='store_true', help='Save per-branch CSV per image')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == '__main__':
    main()
