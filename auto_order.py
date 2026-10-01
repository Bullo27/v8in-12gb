#!/usr/bin/env python
"""Pick the depth order for 9 um ink models (v8-in, ink_9um) from the segment's geometry, instead of running both orders.

The ink models read the layers going from behind the sheet toward the scroll centre. vc_render_tifxyz with --flip-normals
(and the team's published surface volumes, checked on PHerc0841) stores layers along -N, where N = dP/dcol x dP/drow is the
tifxyz grid normal. So the stored order is the model's order ("forward") when N points away from the scroll axis, and must
be reversed when N points toward it. The axis is the centroid of the masked scan per z-slab (coarse pyramid level).

usage: auto_order.py SEGMENT_TIFXYZ_DIR --volume s3://.../<scan>-masked.zarr [--no-flip]
       auto_order.py SEGMENT_TIFXYZ_DIR --axis axis.json            (cached axis: {"z": [...], "cx": [...], "cy": [...]})
prints: forward | reverse | both (mixed patch), and the fraction of vertices whose N points outward."""
import argparse, json, os, sys
import numpy as np, tifffile

def axis_from_volume(url, level=5):
    import zarr
    a = zarr.open(url.rstrip('/') + f'/{level}', mode='r', storage_options={'anon': True} if url.startswith('s3://') else None)
    a = np.asarray(a[:])                                 # one read: the coarse level is small (PHerc0841: 607 x 241 x 241)
    s = 2 ** level; zs, cx, cy = [], [], []
    for z in range(a.shape[0]):
        yy, xx = np.nonzero(a[z])
        if len(xx) >= 50: zs.append(z * s); cx.append(float(xx.mean() * s)); cy.append(float(yy.mean() * s))
    return {'level': level, 'z': zs, 'cx': cx, 'cy': cy}

def outward_fraction(seg, ax):
    P = np.stack([tifffile.imread(os.path.join(seg, f'{a}.tif')).astype(np.float64) for a in 'xyz'], -1)[::2, ::2]
    ok = P[..., 0] > 0
    dc = np.zeros_like(P); dr = np.zeros_like(P); dc[:, 1:-1] = P[:, 2:] - P[:, :-2]; dr[1:-1] = P[2:] - P[:-2]
    v = ok.copy(); v[:, 1:-1] &= ok[:, 2:] & ok[:, :-2]; v[1:-1] &= ok[2:] & ok[:-2]; v[[0, -1], :] = False; v[:, [0, -1]] = False
    z = P[..., 2]
    R = np.stack([P[..., 0] - np.interp(z, ax['z'], ax['cx']), P[..., 1] - np.interp(z, ax['z'], ax['cy']), 0 * z], -1)
    return float(((np.cross(dc, dr) * R).sum(-1)[v] > 0).mean())

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('segment'); ap.add_argument('--volume'); ap.add_argument('--axis')
    ap.add_argument('--no-flip', action='store_true', help='the surface volume was rendered WITHOUT --flip-normals (layers along +N)')
    ap.add_argument('--save-axis', help='write the computed axis to this JSON file')
    a = ap.parse_args()
    if a.axis: ax = json.load(open(a.axis))
    elif a.volume: ax = axis_from_volume(a.volume)
    else: sys.exit('give --volume (masked scan) or --axis')
    if a.save_axis: json.dump(ax, open(a.save_axis, 'w'))
    f = outward_fraction(a.segment, ax)
    if a.no_flip: f = 1 - f
    order = 'forward' if f > 0.7 else 'reverse' if f < 0.3 else 'both'
    print(f'{order}\t{f:.3f}')

if __name__ == '__main__':
    main()
