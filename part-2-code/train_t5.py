import os, torch, argparse
from tqdm import tqdm
import torch.nn as nn

from t5_utils import prepare_model, setup_optimizer_and_sched, store_model, reload_model, init_wandb
from load_data import load_t5_dataloaders
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0


def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--optimizer_type", default="AdamW")
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--scheduler_type", default="linear")
    p.add_argument("--num_warmup_epochs", type=int, default=1)
    p.add_argument("--max_n_epochs", type=int, default=10)
    p.add_argument("--patience_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--experiment_name", default="final_t5")
    return p.parse_args()


def train_loop(cfg, mdl, train, dev, opt, sch):
    best_f1, patience = -1, 0
    kind = "ft" if cfg.finetune else "scr"
    ckpt_dir = os.path.join("checkpoints", f"{kind}_runs", cfg.experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg.checkpoint_dir = ckpt_dir

    gt_sql, gt_rec = "data/dev.sql", "records/ground_truth_dev.pkl"
    mdl_sql, mdl_rec = f"results/{kind}_dev.sql", f"records/{kind}_dev.pkl"
    scaler = torch.cuda.amp.GradScaler() if cfg.fp16 and torch.cuda.is_available() else None

    for ep in range(cfg.max_n_epochs):
        loss = run_train_epoch(cfg, mdl, train, opt, sch, scaler)
        print(f"[Epoch {ep}] train_loss={loss:.4f}")
        ev_loss, f1, em, sql_em, err = run_eval_epoch(cfg, mdl, dev, gt_sql, mdl_sql, gt_rec, mdl_rec)
        print(f"[Epoch {ep}] dev_loss={ev_loss:.4f} F1={f1:.4f} EM={em:.4f} SQL_EM={sql_em:.4f} err={err*100:.1f}%")

        if f1 > best_f1:
            best_f1, patience = f1, 0
            store_model(ckpt_dir, mdl, best_flag=True)
        else:
            patience += 1
        store_model(ckpt_dir, mdl, best_flag=False)
        if patience >= cfg.patience_epochs:
            print("Early stop")
            break


def run_train_epoch(cfg, mdl, loader, opt, sch, scaler):
    mdl.train()
    ce = nn.CrossEntropyLoss()
    tot_loss, tot_tok = 0, 0
    for enc, mask, dec_in, dec_tgt, _ in tqdm(loader):
        enc, mask, dec_in, dec_tgt = enc.to(DEVICE), mask.to(DEVICE), dec_in.to(DEVICE), dec_tgt.to(DEVICE)
        valid = dec_tgt != PAD_IDX
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = mdl(input_ids=enc, attention_mask=mask, decoder_input_ids=dec_in)
            loss = ce(out["logits"][valid], dec_tgt[valid])
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), cfg.grad_clip)
            opt.step()
        if sch: sch.step()
        n = valid.sum().item()
        tot_loss += loss.item() * n
        tot_tok += n
    return tot_loss / tot_tok if tot_tok else 0.0


def run_eval_epoch(cfg, mdl, loader, gt_sql, mdl_sql, gt_rec, mdl_rec):
    mdl.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss, tot_tok, gens = 0, 0, []
    from transformers import T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    with torch.no_grad():
        for enc, mask, dec_in, dec_tgt, _ in tqdm(loader):
            enc, mask = enc.to(DEVICE), mask.to(DEVICE)
            if dec_tgt.numel() > 0:
                dec_in, dec_tgt = dec_in.to(DEVICE), dec_tgt.to(DEVICE)
                out = mdl(input_ids=enc, attention_mask=mask, decoder_input_ids=dec_in)
                v = dec_tgt != PAD_IDX
                loss = ce(out["logits"][v], dec_tgt[v])
                n = v.sum().item()
                tot_loss += loss.item() * n
                tot_tok += n
            preds = mdl.generate(input_ids=enc, attention_mask=mask, max_new_tokens=512)
            gens.extend(tok.batch_decode(preds, skip_special_tokens=True))
    save_queries_and_records(gens, mdl_sql, mdl_rec)
    sql_em, rec_em, rec_f1, errs = compute_metrics(gt_sql, mdl_sql, gt_rec, mdl_rec)
    err_rate = sum(1 for e in errs if e) / len(errs) if errs else 0
    return tot_loss / tot_tok if tot_tok else 0, rec_f1, rec_em, sql_em, err_rate


def main():
    cfg = parse_cfg()
    if cfg.use_wandb: init_wandb(cfg)
    train, dev, test = load_t5_dataloaders(cfg.batch_size, cfg.test_batch_size)
    mdl = prepare_model(cfg)
    opt, sch = setup_optimizer_and_sched(cfg, mdl, len(train))
    train_loop(cfg, mdl, train, dev, opt, sch)
    mdl = reload_model(cfg, use_best=True)
    gt_sql, gt_rec = "data/dev.sql", "records/ground_truth_dev.pkl"
    mdl_sql, mdl_rec = "results/final_dev.sql", "records/final_dev.pkl"
    run_eval_epoch(cfg, mdl, dev, gt_sql, mdl_sql, gt_rec, mdl_rec)


if __name__ == "__main__":
    main()
