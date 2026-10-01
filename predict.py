#!/usr/bin/env python
"""Ink prediction with v8-in or a finetune of it (the released recipe: 64 px tiles at stride 21, tiles fully inside the
coverage mask, Gaussian-weighted stitching), on a layers/ directory or an OME-zarr surface volume, in a given depth order.

usage: predict.py SURFACE (layers dir or .zarr) OUT.png --order forward|reverse [--model YoussefMoNader/ink-8um-v8in]
                  [--stride 21] [--batch 4] [--layer-start N] [--crop R0 R1 C0 C1]
Decide --order with auto_order.py (forward = layers as stored). Writes OUT.png (8 bit) and OUT.tif (float32)."""
import argparse, os, sys, time
import numpy as np, cv2, tifffile

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('surface'); ap.add_argument('out'); ap.add_argument('--order', choices=['forward', 'reverse'], required=True)
    ap.add_argument('--model', default='YoussefMoNader/ink-8um-v8in'); ap.add_argument('--code', default=None)
    ap.add_argument('--stride', type=int, default=21); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--layer-start', type=int, default=None); ap.add_argument('--crop', type=int, nargs=4, default=None)
    a = ap.parse_args()
    code = a.code
    if code is None:                                    # the ink8um package: next to the weights, or the code files of v8-in
        from huggingface_hub import snapshot_download
        code = a.model if os.path.isdir(os.path.join(a.model, 'ink8um')) else \
            snapshot_download('YoussefMoNader/ink-8um-v8in', allow_patterns=['ink8um/*', 'config.json'])
    sys.path.insert(0, code)
    from ink8um import InkDetector
    from ink8um.inference import read_stack, predict_stack
    model = InkDetector.from_pretrained(a.model)
    if os.path.isdir(os.path.join(a.surface, '0')) or a.surface.rstrip('/').endswith('.zarr'):
        import zarr
        z = zarr.open(a.surface, mode='r'); arr = z['0'] if hasattr(z, 'keys') and '0' in z else z
        s0 = (arr.shape[0] - model.in_depth) // 2 if a.layer_start is None else a.layer_start
        sl = (slice(s0, s0 + model.in_depth),) + ((slice(a.crop[0], a.crop[1]), slice(a.crop[2], a.crop[3])) if a.crop else ())
        stack = np.ascontiguousarray(np.clip(np.asarray(arr[sl]), 0, model.clip_max).astype(np.uint8).transpose(1, 2, 0))
    else:
        stack = read_stack(a.surface, layer_start=a.layer_start, depth=model.in_depth, clip_max=model.clip_max)
        if a.crop: stack = np.ascontiguousarray(stack[a.crop[0]:a.crop[1], a.crop[2]:a.crop[3]])
    t = time.time()
    p = predict_stack(model, stack, reverse=(a.order == 'reverse'), stride=a.stride, batch_size=a.batch, num_workers=2)
    base = os.path.splitext(a.out)[0]
    cv2.imwrite(base + '.png', (np.clip(p, 0, 1) * 255).astype(np.uint8)); tifffile.imwrite(base + '.tif', p.astype(np.float32))
    print(f'{a.surface}: {stack.shape} {a.order} stride {a.stride}: {time.time() - t:.0f}s -> {base}.png', flush=True)

if __name__ == '__main__':
    main()
