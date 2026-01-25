import importlib.util
import sys
import json
import os
import torch
import os, json, time, argparse, random, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def tok_to_mid(tok: str) -> int:
    tok = tok.strip()
    if tok.startswith("<item_") and tok.endswith(">"):
        return int(tok[len("<item_"):-1])
    if tok.startswith("item_"):
        return int(tok[len("item_"):])
    return int(tok)

def load_item_ids_map(item_ids_path: str):
    with open(item_ids_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    vocab = obj["item_ids"] if isinstance(obj, dict) and "item_ids" in obj else obj
    vocab = [int(x) for x in vocab]
    id2idx = {mid: (i+1) for i, mid in enumerate(sorted(vocab))}
    n_items = len(vocab)
    return id2idx, n_items

@dataclass
class ExampleWin:
    seq_idx: List[int]
    tgt_idx: int        

class EvalWindowDataset(Dataset):
    """test.jsonl: {"prompt": "<item_1> <item_23> ...", "target": "<item_999>"}"""
    def __init__(self, rows: List[dict], id2idx: Dict[int,int], maxlen: int):
        self.samples: List[ExampleWin] = []
        self.maxlen = maxlen
        for r in rows:
            p = (r.get("prompt") or "").strip()
            t = (r.get("target") or "").strip()
            if not p or not t: continue
            mids = [tok_to_mid(x) for x in p.split()]
            seq = [id2idx[m] for m in mids if m in id2idx]
            tgt_mid = tok_to_mid(t)
            if tgt_mid not in id2idx: continue
            tgt_idx = id2idx[tgt_mid]
            if len(seq) == 0: continue
            self.samples.append(ExampleWin(seq[-maxlen:], tgt_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def eval_collate(batch: List[ExampleWin]):
    B = len(batch)
    maxlen = max(len(b.seq_idx) for b in batch) if B>0 else 1
    seq = torch.zeros((B, maxlen), dtype=torch.long)
    tgt = torch.zeros((B,), dtype=torch.long)
    ctx_list = []
    for i, ex in enumerate(batch):
        L = len(ex.seq_idx)
        seq[i, -L:] = torch.tensor(ex.seq_idx, dtype=torch.long)
        tgt[i] = int(ex.tgt_idx)
        ctx_list.append(np.array(ex.seq_idx, dtype=np.int64))
    return {"seq": seq, "tgt": tgt, "ctx": ctx_list}

def _import_sasrec_class(py_path: str):
    if (py_path is None) or (py_path == ""):
        raise RuntimeError("必须提供 --model_py 指向你的 SASRec 源码文件")
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"SASRec 源码不存在: {py_path}")
    mod_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "SASRec"):
        raise RuntimeError(f"{py_path} 中未找到类 `SASRec`")
    return mod.SASRec

def load_sasrec_from_pretrained(model_dir: str, device: torch.device, model_py: str):
    SASRec = _import_sasrec_class(model_py)
    if hasattr(SASRec, "from_pretrained"):
        model = SASRec.from_pretrained(model_dir, map_location=device).to(device)
        model.eval()
        return model

    cfg_path = os.path.join(model_dir, "config.json")
    sd_path  = os.path.join(model_dir, "pytorch_model.bin")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    n_items       = cfg.get("n_items")
    maxlen        = cfg.get("maxlen")
    hidden_units  = cfg.get("hidden_units", 128)
    num_blocks    = cfg.get("num_blocks", 2)
    num_heads     = cfg.get("num_heads", 2)
    dropout       = cfg.get("dropout", 0.2)

    if (n_items is None) or (maxlen is None):
        raise RuntimeError(f"config.json 缺少关键字段 n_items/maxlen: {cfg}")

    model = SASRec(
        n_items=n_items, maxlen=maxlen, hidden_units=hidden_units,
        num_blocks=num_blocks, num_heads=num_heads, dropout=dropout
    ).to(device)
    state = torch.load(sd_path, map_location=device)
    if any(k.startswith("base_model.") for k in state.keys()):
        state = {k.replace("base_model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def _final_scores(model, seq_ids: torch.Tensor) -> torch.Tensor:
    h = model(seq_ids)  # [B, L, D]
    last = (seq_ids != 0).sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(seq_ids.size(0), device=seq_ids.device)
    h_t = h[rows, last, :]  # [B, D]
    scores = model.score(h_t)  # [B, N]
    return scores

def _recall_at_k(rank: int, k: int) -> float: return 1.0 if rank <= k else 0.0
def _ndcg_at_k(rank: int, k: int) -> float:
    if rank > k: return 0.0
    return 1.0 / math.log2(rank + 1)

@torch.no_grad()
def evaluate_windows(model, loader: DataLoader, n_items: int, topk_list: List[int],
                     device, exclude_seen=True) -> Dict[str, float]:
    model.eval()
    K_list = sorted(topk_list)
    sums = {f"Recall@{k}":0.0 for k in K_list}
    sums.update({f"NDCG@{k}":0.0 for k in K_list})
    n = 0
    for batch in tqdm(loader, desc="Eval(win)", leave=False):
        seq = batch['seq'].to(device)          # [B,L]
        tgt_idx = batch['tgt'].to(device)      # [B] (1..N)
        ctx_list = batch['ctx']
        logits = _final_scores(model, seq)     # [B, N]
        scores = logits.clone()
        if exclude_seen:
            for i, ctx_np in enumerate(ctx_list):
                seen = set(int(x) for x in ctx_np.tolist() if x > 0)
                seen.discard(int(tgt_idx[i].item()))
                if seen:
                    seen0 = torch.tensor([s-1 for s in seen], device=scores.device, dtype=torch.long)
                    scores[i].index_fill_(0, seen0, float('-inf'))
        top_idx = torch.argsort(scores, dim=1, descending=True)
        tgt0 = (tgt_idx - 1).unsqueeze(1)   # 0-based 列号
        pos = (top_idx == tgt0).nonzero(as_tuple=False)
        B = seq.size(0)
        for i in range(B):
            where = pos[(pos[:,0] == i)]
            r = int(where[0,1].item()) + 1 if where.numel() > 0 else (n_items + 1)
            for k in K_list:
                sums[f"Recall@{k}"] += _recall_at_k(r, k)
                sums[f"NDCG@{k}"]   += _ndcg_at_k(r, k)
            n += 1
    return {k: (v / max(1, n)) for k, v in sums.items()}

def mine_edit_pairs_from_topk_sasrec(model, ds: Dataset, device, topk=10, per_user=2, max_pairs=10000):
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=eval_collate)
    pairs = []
    with torch.no_grad():
        for b in tqdm(loader, desc="Mine E from Top-K", leave=False):
            seq = b['seq'].to(device)
            tgt = b['tgt'].to(device)
            scores = _final_scores(model, seq)           # [B, N]
            idx = torch.argsort(scores, dim=1, descending=True)[:, :topk]  # 0..N-1
            B = seq.size(0)
            for i in range(B):
                top_cols = [int(x) for x in idx[i].tolist()]
                bads = [c+1 for c in top_cols if (c+1) != int(tgt[i].item())][:per_user]  # 1..N
                ctx = [int(x) for x in seq[i].tolist() if x > 0]
                for bad in bads:
                    pairs.append({"user_prompt_idx": ctx, "bad_item_idx": bad})
            if len(pairs) >= max_pairs: break
    return pairs[:max_pairs]

@torch.no_grad()
def _embed_prompts_sasrec(model, seq: torch.Tensor) -> torch.Tensor:
    h = model(seq)  # [B,L,D]
    mask = (seq != 0).float().unsqueeze(-1)
    v = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    v = F.normalize(v, dim=1)
    return v

def sr_cr_concentrated_sasrec(model_before, model_after, edit_pairs, device,
                              n_items: int, maxlen: int, topk=10, sim_topM=50):
    def build_batch(rows: List[List[int]]):
        B = len(rows)
        L = maxlen
        seq = torch.zeros((B, L), dtype=torch.long)
        for i, idxs in enumerate(rows):
            idxs = idxs[-L:]
            if idxs:
                seq[i, -len(idxs):] = torch.tensor(idxs, dtype=torch.long)
        return seq.to(device)

    prompts = [r["user_prompt_idx"] for r in edit_pairs]
    bad_idx = torch.tensor([r["bad_item_idx"] for r in edit_pairs], dtype=torch.long, device=device)  # 1..N
    seq = build_batch(prompts)

    with torch.no_grad():
        s_b = _final_scores(model_before, seq)  # [B, N]
        s_a = _final_scores(model_after, seq)
        top_idx_b = torch.argsort(s_b, dim=1, descending=True)
        top_idx_a = torch.argsort(s_a, dim=1, descending=True)

        def _rank_of(top_idx, tgt1):
            tgt0 = (tgt1 - 1).unsqueeze(1)
            pos = (top_idx == tgt0).nonzero(as_tuple=False)
            B = top_idx.size(0)
            ranks = torch.full((B,), n_items+1, device=top_idx.device, dtype=torch.long)
            if pos.numel() > 0:
                ranks[pos[:,0]] = pos[:,1] + 1
            return ranks

        rb = _rank_of(top_idx_b, bad_idx)
        ra = _rank_of(top_idx_a, bad_idx)
        sr_atK = (ra > topk).float().mean().item()
        delta_rank = (rb - ra).float().mean().item()

    with torch.no_grad():
        v = _embed_prompts_sasrec(model_after, seq)
        sims = v @ v.t()
        B = sims.size(0)
        topm = min(sim_topM, max(1, B-1))
        cr_list = []
        eye = torch.eye(B, device=sims.device)
        for i in range(B):
            vals, idx = torch.topk(sims[i] - eye[i]*1e9, k=topm, dim=0)  # 排除自己
            nn_seq = seq[idx]
            sb_nn = _final_scores(model_before, nn_seq)
            sa_nn = _final_scores(model_after, nn_seq)
            topb = torch.argsort(sb_nn, dim=1, descending=True)
            topa = torch.argsort(sa_nn, dim=1, descending=True)
            tgt0 = (bad_idx[i].repeat(topm, 1) - 1)
            pos_b = (topb == tgt0).nonzero(as_tuple=False)
            pos_a = (topa == tgt0).nonzero(as_tuple=False)
            ranks_b = torch.full((topm,), n_items+1, device=seq.device, dtype=torch.long)
            ranks_a = torch.full((topm,), n_items+1, device=seq.device, dtype=torch.long)
            if pos_b.numel() > 0: ranks_b[pos_b[:,0]] = pos_b[:,1] + 1
            if pos_a.numel() > 0: ranks_a[pos_a[:,0]] = pos_a[:,1] + 1
            cr_list.append((ranks_b - ranks_a).float().mean().item())
        cr = float(np.mean(cr_list)) if len(cr_list)>0 else 0.0
    concentrated = {"DeltaNDCG@10": 0.0}

    return {
        f"SR@{topk}": sr_atK,
        "AvgDeltaRank": delta_rank,
        "CR_AvgDeltaRank": cr,
        **concentrated
    }

def select_params_for_scope_sasrec(model, scope: str):
    params=[]
    if scope == "loraB":
        lora_found=False
        for n,p in model.named_parameters():
            if ("lora_B" in n):
                p.requires_grad_(True); params.append(p); lora_found=True
        if lora_found: return params
        scope = "embedding"  # fallback
    if scope == "embedding":
        emb = model.item_emb
        emb.weight.requires_grad_(True); params.append(emb.weight)
        return params
    for p in model.parameters(): p.requires_grad = False
    return params

def ebpr_edit_sasrec(model, edit_pairs, device, n_items: int, maxlen: int,
                     steps=20, lr=3e-4, scope="embedding", neg_per_step=20, grad_clip=1.0):
    model.train()
    # freeze all
    for p in model.parameters(): p.requires_grad = False
    params = select_params_for_scope_sasrec(model, scope)
    if len(params) == 0:
        raise RuntimeError(f"No trainable params for scope={scope}.")
    opt = torch.optim.Adam(params, lr=lr)

    def build_batch(batch_pairs):
        B = len(batch_pairs)
        seq = torch.zeros((B, maxlen), dtype=torch.long, device=device)
        bad_col = torch.zeros((B,), dtype=torch.long, device=device)  # 0..N-1
        for i, r in enumerate(batch_pairs):
            ctx = r["user_prompt_idx"][-maxlen:]
            if ctx:
                seq[i, -len(ctx):] = torch.tensor(ctx, device=device)
            bad_col[i] = int(r["bad_item_idx"]) - 1
        return seq, bad_col

    for _ in tqdm(range(steps), desc=f"E-BPR[SASRec:{scope}]"):
        bs = min(512, len(edit_pairs))
        batch_pairs = random.sample(edit_pairs, k=bs) if len(edit_pairs) > bs else edit_pairs
        seq, bad_col = build_batch(batch_pairs)

        neg_cols = torch.randint(0, n_items, (bs, neg_per_step), device=device)
        neg_cols = torch.where(neg_cols==bad_col.view(-1,1), (neg_cols+1)%n_items, neg_cols)

        scores = _final_scores(model, seq)              # [B, N]
        s_bad = scores.gather(1, bad_col.view(-1,1)).expand(-1, neg_per_step)  # [B,M]
        s_neg = scores.gather(1, neg_cols)                                      # [B,M]

        loss = -torch.log(torch.sigmoid(s_neg - s_bad) + 1e-8).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            for p in params:
                torch.nn.utils.clip_grad_norm_(p, grad_clip)
        opt.step()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zj/code/ml-10M100K",
                    help="contain item_ids.json, test.jsonl")
    ap.add_argument("--model_dir", type=str, default="/home/zj/code/llm_cluster_lora/runs/Bert4Rec_ml10M100K/base_model",
                    help="base model path")
    ap.add_argument("--output_dir", type=str, default="/home/zj/code/llm_cluster_lora/runs/Bert4Rec_ml10M100K/edit_baseline")
    ap.add_argument("--model_py", type=str, default="/home/zj/code/llm_cluster_lora/Bert4Rec_RAIE_ALL_NEW.py",
                    help="model_py path")

    ap.add_argument("--maxlen", type=int, default=30)
    ap.add_argument("--topk_eval", type=str, default="5,10,20")

    ap.add_argument("--edit_source", type=str, choices=["from_topk","from_file"], default="from_topk")
    ap.add_argument("--topk_mine", type=int, default=10)
    ap.add_argument("--per_user", type=int, default=2)
    ap.add_argument("--max_pairs", type=int, default=10000)
    ap.add_argument("--edit_pairs", type=str, default="")

    ap.add_argument("--budget_steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scope", type=str, choices=["embedding","loraB"], default="embedding")

    ap.add_argument("--load_lora_default", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Vocab / Data =====
    id2idx, n_items = load_item_ids_map(os.path.join(args.data_dir, "item_ids.json"))

    test_path = os.path.join(args.data_dir, "original.jsonl")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found")
    test_rows = read_jsonl(test_path)

    eval_ds = EvalWindowDataset(test_rows, id2idx, maxlen=args.maxlen)
    eval_loader = DataLoader(eval_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=eval_collate)

    # ===== Load SASRec base =====
    model = load_sasrec_from_pretrained(args.model_dir, device, args.model_py)

    model.eval()

    if args.load_lora_default and PEFT_AVAILABLE:
        default_dir = os.path.join(os.path.dirname(args.model_dir), "default")
        if os.path.isdir(default_dir):
            model = PeftModel.from_pretrained(model, default_dir, is_trainable=True).to(device)
            try: model.set_adapter("default")
            except: pass

    # ===== Before metrics =====
    topk_tuple = tuple(int(x) for x in args.topk_eval.split(","))
    metrics_before = evaluate_windows(model, eval_loader, n_items, list(topk_tuple), device)
    print("[Before]", " ".join([f"{k}:{v:.6f}" for k,v in sorted(metrics_before.items())]))

    # ===== Edit pairs =====
    if args.edit_source == "from_file":
        rows = read_jsonl(args.edit_pairs)
        edit_pairs = []
        for r in rows:
            if "user_prompt_idx" in r and "bad_item_idx" in r:
                edit_pairs.append({"user_prompt_idx": r["user_prompt_idx"], "bad_item_idx": r["bad_item_idx"]})
            else:
                p = (r.get("user_prompt") or "").strip()
                bi = (r.get("bad_item") or "").strip()
                if (not p) or (not bi): continue
                ctx_idx = [id2idx.get(tok_to_mid(t), None) for t in p.split()]
                ctx_idx = [x for x in ctx_idx if x is not None]
                bad_mid = tok_to_mid(bi)
                if bad_mid not in id2idx: continue
                edit_pairs.append({"user_prompt_idx": ctx_idx[-args.maxlen:], "bad_item_idx": id2idx[bad_mid]})
    else:
        edit_pairs = mine_edit_pairs_from_topk_sasrec(
            model, eval_ds, device, topk=args.topk_mine, per_user=args.per_user, max_pairs=args.max_pairs
        )
        write_jsonl(os.path.join(args.output_dir, "mined_edit_pairs.jsonl"), edit_pairs)
        print(f"[Mine] got {len(edit_pairs)} edit pairs")

    # ===== Clone model (after) =====
    model_after = load_sasrec_from_pretrained(args.model_dir, device, args.model_py)

    if args.load_lora_default and PEFT_AVAILABLE:
        default_dir = os.path.join(os.path.dirname(args.model_dir), "default")
        if os.path.isdir(default_dir):
            model_after = PeftModel.from_pretrained(model_after, default_dir, is_trainable=True).to(device)
            try: model_after.set_adapter("default")
            except: pass

    # ===== E-BPR Editing =====
    t0 = time.time()
    ebpr_edit_sasrec(
        model_after, edit_pairs, device,
        n_items=n_items, maxlen=args.maxlen,
        steps=args.budget_steps, lr=args.lr, scope=args.scope,
        neg_per_step=20, grad_clip=1.0
    )
    edit_time = time.time() - t0

    # ===== After metrics =====
    metrics_after = evaluate_windows(model_after, eval_loader, n_items, list(topk_tuple), device)
    print("[After ]", " ".join([f"{k}:{v:.6f}" for k,v in sorted(metrics_after.items())]))

    # ===== SR / CR / Concentrated =====
    # srcr = sr_cr_concentrated_sasrec(
    #     model_before=model, model_after=model_after,
    #     edit_pairs=edit_pairs, device=device,
    #     n_items=n_items, maxlen=args.maxlen,
    #     topk=max(topk_tuple), sim_topM=50
    # )

    out = {
        "before": metrics_before,
        "after": metrics_after,
        # "SR_CR_Concentrated": srcr,
        "edit_pairs": len(edit_pairs),
        "budget_steps": args.budget_steps,
        "lr": args.lr,
        "scope": args.scope,
        "edit_time_sec": float(edit_time)
    }
    with open(os.path.join(args.output_dir, "edit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[EDIT][Summary]", json.dumps(out, ensure_ascii=False))
    print(f"Saved to: {os.path.join(args.output_dir, 'edit_metrics.json')}")


if __name__ == "__main__":
    main()
