#!/usr/bin/env python
"""Finetune v8-in (Youssef Nader's 9 um ink model, YoussefMoNader/ink-8um-v8in) on your own labelled segments on one 12 GB GPU.

The recipe follows the released loo-w062 finetune (YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062, training/configs/loo_w062.json):
init from v8-in (all weights), fresh AdamW (weight decay 1e-6, not on biases/norms), lr 1e-5 -> 1e-6 per-step cosine, 10 epochs,
effective batch 32, fp16 autocast, grad-norm clip 1.0, 64 px tiles at stride 48 admitted only where the mask covers the whole
tile (unlabelled pixels inside the mask are negatives), loss 0.5 Dice + 0.5 soft-BCE (label smoothing 0.25), augmentations:
flips, rotation +-180 / scale +-10 % / shift, Gaussian blur, coarse dropout, depth jitter.
What differs, so that it fits a 12 GB card: micro-batch 2 x accumulation 16 with BatchNorm statistics frozen; OpenCV/numpy
augmentations instead of albumentations (no motion blur); no validation during training. Gradient checkpointing switches on
by itself after an out-of-memory error.
Measured: retraining loo-w062 this way (w058 + w060, 2,712 tiles; the original used 2,718) gives a model whose map of the
held-out w062 correlates r = 0.972 with the released loo-w062 on the same render (RTX 3060 12 GB at 120 W: 3.3 h, 9.0 GB peak).

A segment is a directory with
  - the surface volume: layers/00.tif, 01.tif, ... (the layout of the PHerc1447 dataset), or one *.zarr (OME-zarr, level 0
    shaped depth x height x width, e.g. a vc_render_tifxyz --zarr-output render); the central 24 layers are used unless
    --layer-start is given;
  - ink labels: inklabels.png, labels/inklabels.png or <name>_inklabels.png (> 127 = ink);
  - a mask of where the labels are trusted: mask.png, labels/mask.png or <name>_mask.png (> 0 = supervised).
--order: the model reads the layers going toward the scroll centre. Use auto_order.py on the segment's tifxyz to decide
(forward = as stored, reverse = flipped); for the PHerc1447 dataset layers it is reverse.

usage: finetune.py --segment DIR [--segment DIR ...] --order forward|reverse --out MODEL_DIR [--epochs 10] [--smoke N]
The output directory loads with InkDetector.from_pretrained(MODEL_DIR) (config.json + model.safetensors)."""
import argparse, glob, json, math, os, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

def find_one(d, names):
    for n in names:
        hits = sorted(glob.glob(os.path.join(d, n)))
        if hits: return hits[0]
    sys.exit(f'{d}: none of {names} found')

def load_segment(d, order, layer_start=None, depth=24, clip_max=200):
    """(24, H, W) uint8 stack in the model's order, float32 labels (0/1), bool mask."""
    zs = sorted(glob.glob(os.path.join(d, '*.zarr')))
    if os.path.isdir(os.path.join(d, 'layers')):
        from ink8um.inference import read_stack
        st = read_stack(os.path.join(d, 'layers'), layer_start=layer_start, depth=depth, clip_max=clip_max).transpose(2, 0, 1)
    elif zs:
        import zarr
        z = zarr.open(zs[0], mode='r'); a = z['0'] if hasattr(z, 'keys') and '0' in z else z
        s0 = (a.shape[0] - depth) // 2 if layer_start is None else layer_start
        st = np.clip(np.asarray(a[s0:s0 + depth]), 0, clip_max).astype(np.uint8)
    else:
        sys.exit(f'{d}: no layers/ directory and no *.zarr surface volume')
    if order == 'reverse': st = np.ascontiguousarray(st[::-1])
    lab = cv2.imread(find_one(d, ['inklabels.png', 'labels/inklabels.png', '*_inklabels.png']), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(find_one(d, ['mask.png', 'labels/mask.png', '*_mask.png']), cv2.IMREAD_GRAYSCALE)
    H, W = st.shape[1:]
    if lab.shape != (H, W) or msk.shape != (H, W):
        sys.exit(f'{d}: labels {lab.shape} / mask {msk.shape} do not match the surface volume {(H, W)}')
    sup = (msk > 0) & (st[depth // 2] > 0)
    return st, (lab > 127).astype(np.float32), sup

def tiles(sup, ts=64, stride=48, margin=16):
    H, W = sup.shape; ii = cv2.integral(sup.astype(np.uint8)); out = []
    for y in range(margin, H - ts - margin + 1, stride):
        for x in range(margin, W - ts - margin + 1, stride):
            if ii[y + ts, x + ts] - ii[y, x + ts] - ii[y + ts, x] + ii[y, x] == ts * ts: out.append((y, x))
    return out

class Tiles(torch.utils.data.Dataset):
    def __init__(self, segs):
        self.data = [(st, lab) for st, lab, _ in segs]
        self.items = [(k, y, x) for k, (_, _, sup) in enumerate(segs) for y, x in tiles(sup)]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        k, y, x = self.items[i]; st, lab = self.data[k]; ts, m = 64, 16
        win = st[:, y - m:y + ts + m, x - m:x + ts + m].transpose(1, 2, 0).astype(np.float32)
        lw = lab[y - m:y + ts + m, x - m:x + ts + m].copy()
        if random.random() < 0.75:                                       # shift / scale / rotate about the window centre
            c = (ts / 2 + m + random.uniform(-0.15, 0.15) * ts, ts / 2 + m + random.uniform(-0.15, 0.15) * ts)
            M = cv2.getRotationMatrix2D(c, random.uniform(-180, 180), 1 + random.uniform(-0.1, 0.1))
            win = np.stack([cv2.warpAffine(win[..., j], M, (96, 96), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for j in range(win.shape[2])], -1)
            lw = cv2.warpAffine(lw, M, (96, 96), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.5: win, lw = win[:, ::-1], lw[:, ::-1]
        if random.random() < 0.5: win, lw = win[::-1], lw[::-1]
        win = np.ascontiguousarray(win[m:m + ts, m:m + ts]); lw = np.ascontiguousarray(lw[m:m + ts, m:m + ts]); D = win.shape[2]
        if random.random() < 0.4:                                        # Gaussian blur in the tile plane
            s = random.uniform(0.3, 1.2); win = np.stack([cv2.GaussianBlur(win[..., j], (0, 0), s) for j in range(D)], -1)
        if random.random() < 0.5:                                        # coarse dropout: up to 2 holes <= 20 % of the side
            for _ in range(random.randint(1, 2)):
                h, w = random.randint(1, int(0.2 * ts)), random.randint(1, int(0.2 * ts))
                yy, xx = random.randint(0, ts - h), random.randint(0, ts - w); win[yy:yy + h, xx:xx + w] = 0
        if random.random() < 0.6:                                        # depth jitter: 90-100 % of the layers re-pasted at a random offset
            n = random.randint(int(0.9 * D), D); s0 = random.randint(0, D - n); d0 = random.randint(0, D - n)
            out = np.zeros_like(win); out[..., d0:d0 + n] = win[..., s0:s0 + n]; win = out
            if random.random() < 0.6:
                for _ in range(random.randint(1, 2)): win[..., random.randint(0, D - 1)] = 0
        return torch.from_numpy(win.transpose(2, 0, 1).copy()) / 255.0, torch.from_numpy(lw[None].copy())

def dice_loss(logits, y, eps=1e-7):                                     # segmentation_models_pytorch DiceLoss(mode="binary")
    p = torch.sigmoid(logits).reshape(logits.shape[0], 1, -1); t = y.reshape(y.shape[0], 1, -1)
    d = 2 * (p * t).sum((0, 2)) / (p + t).sum((0, 2)).clamp_min(eps)
    return ((1 - d) * (t.sum((0, 2)) > 0).float()).mean()
def soft_bce(logits, y, s=0.25):                                         # SoftBCEWithLogitsLoss(smooth_factor=0.25)
    return F.binary_cross_entropy_with_logits(logits, (1 - y) * s + y * (1 - s))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--segment', action='append', required=True); ap.add_argument('--order', choices=['forward', 'reverse'], required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--init', default='YoussefMoNader/ink-8um-v8in')
    ap.add_argument('--code', default=None, help='directory containing the ink8um package (default: download it with the model)')
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--micro', type=int, default=2); ap.add_argument('--accum', type=int, default=16)
    ap.add_argument('--layer-start', type=int, default=None); ap.add_argument('--seed', type=int, default=130697)
    ap.add_argument('--smoke', type=int, default=0, help='run N micro-batches only, to check the setup')
    ap.add_argument('--ckpt', action='store_true', help='gradient checkpointing from the start')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    code = a.code
    if code is None:                                    # the ink8um package: next to the weights, or the code files of v8-in
        from huggingface_hub import snapshot_download
        code = a.init if os.path.isdir(os.path.join(a.init, 'ink8um')) else \
            snapshot_download('YoussefMoNader/ink-8um-v8in', allow_patterns=['ink8um/*', 'config.json'])
    sys.path.insert(0, code)
    from ink8um import InkDetector
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    segs = [load_segment(d, a.order, a.layer_start) for d in a.segment]
    ds = Tiles(segs)
    if not len(ds): sys.exit('no 64 px tile lies entirely inside the masks')
    print(f'{len(a.segment)} segments, {len(ds)} tiles, ink fraction {np.mean([lab[sup].mean() for _, lab, sup in segs]):.3f}', flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.micro, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    model = InkDetector.from_pretrained(a.init).cuda(); model.train()
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)): mod.eval()       # frozen BN statistics (micro-batch 2)
    decay = [p for p in model.parameters() if p.ndim > 1]; no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 1e-6}, {'params': no_decay, 'weight_decay': 0.0}], lr=1e-5)
    total = (max(1, a.smoke // a.accum) if a.smoke else (len(dl) // a.accum) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1e-6 + 0.5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * min(s, total) / total))) / 1e-5)
    scaler = torch.amp.GradScaler('cuda')
    def use_checkpointing():
        from torch.utils.checkpoint import checkpoint
        bb = model.backbone
        def fwd(x):
            x = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
            x1 = checkpoint(bb.layer1, x, use_reentrant=False); x2 = checkpoint(bb.layer2, x1, use_reentrant=False)
            x3 = checkpoint(bb.layer3, x2, use_reentrant=False); return [x1, x2, x3, checkpoint(bb.layer4, x3, use_reentrant=False)]
        bb.forward = fwd; print('gradient checkpointing on', flush=True)
    if a.ckpt: use_checkpointing()
    def step_loss(x, y):
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)
        out = out.float(); loss = 0.5 * dice_loss(out, y) + 0.5 * soft_bce(out, y)
        scaler.scale(loss / a.accum).backward(); return loss.item()
    log = open(os.path.join(a.out, 'train_log.jsonl'), 'a'); step, t0 = 0, time.time()
    for ep in range(1 if a.smoke else a.epochs):
        run, nb = 0.0, 0; opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl):
            if a.smoke and i >= a.smoke: break
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            oom = False
            try: lv = step_loss(x, y)
            except torch.cuda.OutOfMemoryError:
                if a.ckpt: raise
                oom = True
            if oom:                                     # retry outside the except block, once its traceback has released the activations
                a.ckpt = True; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); use_checkpointing(); lv = step_loss(x, y)
            run += lv; nb += 1
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 10 == 0:
                    print(f'  epoch {ep} step {step}/{total} loss {run / nb:.4f} lr {sched.get_last_lr()[0]:.2e} '
                          f'{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)
        rec = dict(epoch=ep, step=step, loss=run / max(nb, 1), secs=round(time.time() - t0),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1)); log.write(json.dumps(rec) + '\n'); log.flush()
        print('epoch', rec, flush=True)
    cfg = dict(getattr(model, '_hub_mixin_config', None) or json.load(open(os.path.join(code, 'config.json'))))
    cfg['reverse_layers'] = a.order == 'reverse'        # so that predict.py / predict_surface read the layers the way we trained
    model.eval(); model.save_pretrained(a.out, config=cfg)
    json.dump(dict(segments=a.segment, order=a.order, epochs=a.epochs, tiles=len(ds), micro=a.micro, accum=a.accum, seed=a.seed,
                   grad_checkpointing=a.ckpt, init=a.init), open(os.path.join(a.out, 'finetune_meta.json'), 'w'), indent=1)
    print('done', a.out, f'{time.time() - t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
