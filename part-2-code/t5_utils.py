import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb(cfg):
    """Optional setup for wandb."""
    try:
        import importlib
        wandb = importlib.import_module("wandb")
        wandb.init(
            project=getattr(cfg, "wandb_project", "text2sql"),
            name=getattr(cfg, "experiment_name", None),
        )
        if hasattr(cfg, "__dict__"):
            wandb.config.update(cfg.__dict__)
    except Exception:
        print("wandb unavailable or failed — skipping logging.")


def prepare_model(cfg):
    """Initialize T5 model (from pretrained or config)."""
    base = "google-t5/t5-small"
    if getattr(cfg, "finetune", False):
        print(f"Loading pretrained model: {base}")
        mdl = T5ForConditionalGeneration.from_pretrained(base)
    else:
        print(f"Initializing new T5 from config: {base}")
        conf = T5Config.from_pretrained(base)
        mdl = T5ForConditionalGeneration(conf)
    mdl.to(DEVICE)
    return mdl


def ensure_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def store_model(save_dir, mdl, best_flag):
    ensure_dir(save_dir)
    fname = "best.pt" if best_flag else "last.pt"
    path = os.path.join(save_dir, fname)
    try:
        torch.save(mdl.state_dict(), path)
        print(f"Model checkpoint saved: {path}")
    except Exception as e:
        print(f"Failed saving model: {e}")


def reload_model(cfg, use_best):
    if hasattr(cfg, "checkpoint_dir") and cfg.checkpoint_dir:
        ckpt_dir = cfg.checkpoint_dir
    else:
        kind = "ft" if getattr(cfg, "finetune", False) else "scr"
        ckpt_dir = os.path.join("checkpoints", f"{kind}_runs", getattr(cfg, "experiment_name", "exp"))

    fname = "best.pt" if use_best else "last.pt"
    ckpt_path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    mdl = prepare_model(cfg)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        mdl.load_state_dict(state)
    except RuntimeError as e:
        print(f"Warning: {e}, loading with strict=False")
        mdl.load_state_dict(state, strict=False)

    mdl.to(DEVICE)
    print(f"Loaded weights from {ckpt_path}")
    return mdl


def setup_optimizer_and_sched(cfg, mdl, n_batches):
    opt = configure_optimizer(cfg, mdl)
    sch = configure_scheduler(cfg, opt, n_batches)
    return opt, sch


def configure_optimizer(cfg, mdl):
    decay_keys = extract_param_names(mdl, ALL_LAYERNORM_LAYERS)
    decay_keys = [k for k in decay_keys if "bias" not in k]

    groups = [
        {"params": [p for n, p in mdl.named_parameters() if (n in decay_keys and p.requires_grad)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in mdl.named_parameters() if (n not in decay_keys and p.requires_grad)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, eps=1e-8, betas=(0.9, 0.999))


def configure_scheduler(cfg, opt, steps_per_epoch):
    total_steps = steps_per_epoch * cfg.max_n_epochs
    warmup_steps = steps_per_epoch * cfg.num_warmup_epochs
    if cfg.scheduler_type == "none":
        return None
    if cfg.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    if cfg.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler_type}")


def extract_param_names(module, excluded_layers):
    names = []
    for name, submod in module.named_children():
        names += [f"{name}.{n}" for n in extract_param_names(submod, excluded_layers)
                  if not isinstance(submod, tuple(excluded_layers))]
    names += list(module._parameters.keys())
    return names
