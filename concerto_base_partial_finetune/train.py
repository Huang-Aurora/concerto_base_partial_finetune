"""Partial fine-tune: Concerto base last encoder stage unfrozen + binary head.

Usage (Kaggle, from this folder):
  pip install -r requirements.txt
  python train.py                # full experiment: 30 epochs, auto-resume
  python train.py --mode smoke   # 2-epoch sanity check
  python train.py --no-resume    # ignore an existing resume.pth (start over)

Outputs (checkpoints / logs / PLY / history) go to
$OUTPUT_ROOT/concerto_base_partial_finetune_<stamp> (default
/kaggle/working/outputs/...). Each epoch saves resume.pth (model + optimizer
+ history, written atomically); re-running train.py continues from it.

Differences vs the frozen probe, and only these:
  - two AdamW param groups (unfrozen encoder stage @ ENC_LR, head @ HEAD_LR),
    per-group cosine decay, frozen params absent from the optimizer;
  - forward without torch.no_grad (gradients reach the unfrozen stage);
  - early stopping on val FULL-SIZE rock IoU (patience, max 30 epochs);
  - freeze verification checks the FROZEN subset stayed bit-identical and
    reports how many unfrozen params moved;
  - checkpoints store head + unfrozen-encoder state (not the 414MB encoder).
Evaluation protocol, loss, data pipeline, seed, and checkpoint-selection
metric are IDENTICAL to the frozen probe (see config_finetune.py).
Test scenes 15/19/20 are NOT used.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

WORK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK_ROOT)

import config as C  # noqa: E402
import bootstrap  # noqa: E402
bootstrap.ensure(C.CONCERTO_CKPT)

import config_finetune as CF  # noqa: E402
from dataset import RockSceneDataset  # noqa: E402
from model_finetune import ConcertoPartialFinetuneSeg  # noqa: E402
from utils import encoder_fingerprint, evaluate, export_ply, log, \
    move_batch, set_seed  # noqa: E402

RESUME_NAME = "resume.pth"


def cosine_lr(peak, floor, ep, epochs):
    return floor + 0.5 * (peak - floor) * (1 + np.cos(np.pi * (ep - 1) / epochs))


def save_resume(path, model, opt, epoch, best, history, run_cfg,
                epochs_no_improve):
    tmp = path + ".tmp"
    torch.save(dict(
        head=model.head.state_dict(),
        unfrozen_encoder=model.unfrozen_state_dict(),
        optimizer=opt.state_dict(),
        epoch=epoch, best=best, history=history,
        epochs_no_improve=epochs_no_improve,
        unfreeze_stages=model.unfreeze_stages,
        trainable_enc_params=model.trainable_enc_param_names,
        run_config=run_cfg),
        tmp)
    os.replace(tmp, path)  # atomic: never a half-written resume file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--unfreeze-stages", default=None,
                    help="comma list overriding config_finetune.UNFREEZE_STAGES")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh even if resume.pth exists")
    args = ap.parse_args()

    unfreeze = (tuple(int(x) for x in args.unfreeze_stages.split(","))
                if args.unfreeze_stages else None)
    run_cfg = dict(CF.SMOKE if args.mode == "smoke" else CF.FINETUNE)
    if args.epochs:
        run_cfg["epochs"] = args.epochs
    set_seed(CF.SEED)

    device = "cuda"
    free_b, _ = torch.cuda.mem_get_info()
    print(f"[env] GPU: {torch.cuda.get_device_name(0)}, free "
          f"{free_b / 2**30:.1f} GiB", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(C.OUTPUT_ROOT,
                           args.tag or f"{CF.RUN_NAME}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "w")
    log(f"[run] mode={args.mode} cfg={run_cfg} out={out_dir}", log_f)
    log(f"[run] unfreeze_stages={unfreeze or CF.UNFREEZE_STAGES} "
        f"enc_lr={CF.ENC_LR}->{CF.ENC_LR_MIN} head_lr={CF.HEAD_LR}->"
        f"{CF.HEAD_LR_MIN} wd={CF.WEIGHT_DECAY} patience="
        f"{CF.EARLY_STOP_PATIENCE} seed={CF.SEED}", log_f)

    # ---------------------------------------------------------------- model
    model = ConcertoPartialFinetuneSeg(unfreeze_stages=unfreeze).to(device)
    n_head = sum(p.numel() for p in model.head.parameters())
    n_enc_train = sum(p.numel() for p in model.trainable_encoder_parameters())
    assert all(not p.requires_grad for n, p in model.encoder.named_parameters()
               if n in model.frozen_param_names)
    log(f"[model] encoder total={model.n_encoder / 1e6:.2f}M | frozen="
        f"{model.n_frozen / 1e6:.2f}M | trainable encoder="
        f"{n_enc_train / 1e6:.3f}M ({len(model.trainable_enc_param_names)} "
        f"params, stages {model.unfreeze_stages}) | head={n_head / 1e3:.1f}K",
        log_f)
    trainable_mods = sorted(set(n.rsplit(".", 1)[0]
                                for n in model.trainable_enc_param_names))
    log(f"[model] trainable encoder modules: {trainable_mods}", log_f)
    log(f"[model] ckpt: {C.CONCERTO_CKPT}", log_f)
    fp_before = encoder_fingerprint(model)
    json.dump(fp_before, open(os.path.join(out_dir, "encoder_fp_before.json"),
                             "w"))

    # two param groups; frozen params are NOT in the optimizer by construction
    enc_params = model.trainable_encoder_parameters()
    head_params = [p for p in model.head.parameters()]
    assert all(p.requires_grad for p in enc_params + head_params)
    opt = torch.optim.AdamW(
        [dict(params=enc_params, lr=CF.ENC_LR),
         dict(params=head_params, lr=CF.HEAD_LR)],
        weight_decay=CF.WEIGHT_DECAY)
    log(f"[opt] AdamW groups: encoder(lr={CF.ENC_LR}) {len(enc_params)} "
        f"params, head(lr={CF.HEAD_LR}) {len(head_params)} params, "
        f"wd={CF.WEIGHT_DECAY}", log_f)

    weight = (torch.tensor(C.CLASS_WEIGHTS, dtype=torch.float32).to(device)
              if C.CLASS_WEIGHTS else None)
    log(f"[loss] CE weight={C.CLASS_WEIGHTS} ignore_index={C.IGNORE_LABEL} "
        f"(same as the frozen probe)", log_f)

    # ------------------------------------------------------------ resume?
    resume_path = os.path.join(out_dir, RESUME_NAME)
    start_ep, best, history, epochs_no_improve = 1, dict(rock_iou_full=-1.0,
                                                         epoch=-1), [], 0
    if os.path.isfile(resume_path) and not args.no_resume:
        ck = torch.load(resume_path, map_location="cpu")
        assert ck["run_config"] == run_cfg, \
            f"resume run_config mismatch: {ck['run_config']} vs {run_cfg}"
        assert tuple(ck["unfreeze_stages"]) == model.unfreeze_stages
        model.head.load_state_dict(ck["head"])
        model.load_unfrozen_state_dict(ck["unfrozen_encoder"])
        opt.load_state_dict(ck["optimizer"])
        start_ep = ck["epoch"] + 1
        best, history = ck["best"], ck["history"]
        epochs_no_improve = ck["epochs_no_improve"]
        log(f"[resume] {resume_path}: continuing from epoch {start_ep} "
            f"(best so far ep{best['epoch']} rockIoU="
            f"{best['rock_iou_full']:.4f})", log_f)
        if start_ep > run_cfg["epochs"]:
            log("[resume] training already complete; nothing to do", log_f)
            print(out_dir)
            return

    # ------------------------------------------------------------ training
    train_ids, val_ids = run_cfg["train_scenes"], run_cfg["val_scenes"]
    train_ds = RockSceneDataset(train_ids, "train")
    epochs, patience = run_cfg["epochs"], CF.EARLY_STOP_PATIENCE
    stop_reason = "max_epochs"

    for ep in range(start_ep, epochs + 1):
        t0 = time.time()
        lr_e = cosine_lr(CF.ENC_LR, CF.ENC_LR_MIN, ep, epochs)
        lr_h = cosine_lr(CF.HEAD_LR, CF.HEAD_LR_MIN, ep, epochs)
        opt.param_groups[0]["lr"] = lr_e
        opt.param_groups[1]["lr"] = lr_h
        model.train()
        losses, n_seen = [], 0
        grad_norm_first = None
        for i in range(len(train_ds)):
            sample = train_ds[i]
            batch = move_batch(sample, device)
            logits, point = model(batch)
            seg = point.segment.long()
            loss = F.cross_entropy(logits.float(), seg, weight=weight,
                                   ignore_index=C.IGNORE_LABEL)
            assert torch.isfinite(loss), f"non-finite loss at ep{ep} scene {i}"
            opt.zero_grad()
            loss.backward()
            if ep == 1 and i == 0:
                grad_norm_first = [
                    float(sum(p.grad.norm() ** 2 for p in g["params"]
                              if p.grad is not None) ** 0.5)
                    for g in opt.param_groups]
            opt.step()
            losses.append(float(loss) * seg.numel())
            n_seen += seg.numel()
        train_loss = sum(losses) / max(n_seen, 1)

        records, m_g, m_f = evaluate(model, val_ids, device)
        mg, mf = m_g.compute(), m_f.compute()
        rec = dict(epoch=ep, lr_encoder=lr_e, lr_head=lr_h,
                   train_loss=train_loss, time=round(time.time() - t0, 1),
                   val_grid={k: v for k, v in mg.items() if k != "confusion"},
                   val_full={k: v for k, v in mf.items() if k != "confusion"},
                   grad_norm_first=grad_norm_first)
        history.append(rec)
        np.save(os.path.join(out_dir, f"confusion_ep{ep}_grid.npy"),
                mg["confusion"])
        np.save(os.path.join(out_dir, f"confusion_ep{ep}_full.npy"),
                mf["confusion"])
        log(f"[ep {ep:02d}] loss={train_loss:.4f} lr_e={lr_e:.2e} "
            f"lr_h={lr_h:.2e} val_full OA={mf['oa']:.4f} "
            f"mIoU={mf['miou']:.4f} rockIoU={mf['iou_rock']:.4f} "
            f"rockF1={mf['f1_rock']:.4f} rockP={mf['precision_rock']:.4f} "
            f"rockR={mf['recall_rock']:.4f} ({rec['time']}s)", log_f)
        if grad_norm_first is not None:
            log(f"[check] grad norm (1st batch) encoder={grad_norm_first[0]:.4e}"
                f" head={grad_norm_first[1]:.4e}", log_f)

        def save_ckpt(fname):
            torch.save(dict(
                head=model.head.state_dict(),
                unfrozen_encoder=model.unfrozen_state_dict(),
                epoch=ep, metrics=rec["val_full"],
                unfreeze_stages=model.unfreeze_stages,
                trainable_enc_params=model.trainable_enc_param_names,
                run_config=run_cfg),
                os.path.join(out_dir, fname))

        save_ckpt("last_head.pth")
        if mf["iou_rock"] > best["rock_iou_full"]:
            best = dict(rock_iou_full=mf["iou_rock"], epoch=ep)
            save_ckpt("best_head.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_resume(resume_path, model, opt, ep, best, history, run_cfg,
                    epochs_no_improve)

        if epochs_no_improve >= patience:
            stop_reason = f"early_stop(patience={patience})"
            log(f"[stop] no val rock-IoU improvement for {patience} "
                f"epochs (best ep{best['epoch']})", log_f)
            break

    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"),
              indent=1)

    # ------------------------------------------------- post-training checks
    fp_after = encoder_fingerprint(model)
    frozen_changed = [k for k in model.frozen_param_names
                      if fp_before[k] != fp_after.get(k)]
    assert not frozen_changed, f"frozen params changed: {frozen_changed[:5]}"
    unfrozen_changed = [k for k in model.trainable_enc_param_names
                        if fp_before[k] != fp_after.get(k)]
    json.dump(dict(frozen_unchanged=True, n_frozen_changed=len(frozen_changed),
                   n_unfrozen_changed=len(unfrozen_changed),
                   n_unfrozen_total=len(model.trainable_enc_param_names)),
              open(os.path.join(out_dir, "freeze_check.json"), "w"), indent=1)
    log(f"[check] frozen params unchanged: True | unfrozen moved: "
        f"{len(unfrozen_changed)}/{len(model.trainable_enc_param_names)}",
        log_f)

    # export PLYs with the BEST checkpoint
    ckpt = torch.load(os.path.join(out_dir, "best_head.pth"),
                      map_location="cpu")
    model.head.load_state_dict(ckpt["head"])
    model.load_unfrozen_state_dict(ckpt["unfrozen_encoder"])
    log(f"[export] using best epoch {ckpt['epoch']} (val full rock IoU "
        f"{ckpt['metrics']['iou_rock']:.4f})", log_f)
    export_ply(model, run_cfg["export_scenes"], os.path.join(out_dir, "ply"),
               device, log_f)

    log(f"[done] stop={stop_reason} best epoch={best['epoch']} val full "
        f"rock IoU={best['rock_iou_full']:.4f}", log_f)
    print(out_dir)  # machine-readable final line


if __name__ == "__main__":
    main()
