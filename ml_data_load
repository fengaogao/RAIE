import os, json
import argparse
from typing import List, Dict, Tuple
import numpy as np

def format_item_token(mid: int) -> str:
    return f"<item_{mid}>"

def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_mlm1m_ratings(ratings_path: str) -> Dict[int, List[Tuple[int, int]]]:
    user2seq = {}
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            uid = int(parts[0]); mid = int(parts[1]); rating = float(parts[2]); ts = int(parts[3])
            if rating >= 4.0:
                user2seq.setdefault(uid, []).append((mid, ts))
    for u in user2seq:
        user2seq[u].sort(key=lambda x: x[1])
    return user2seq

def sliding_windows(seq: List[str], win_len: int, stride: int = 1, align: str = "end"):
    n = len(seq)
    if n < win_len:
        return
    if align == "end":
        start = (n - win_len) % stride
    else: 
        start = 0

    for i in range(start, n - win_len + 1, stride):
        window = seq[i:i + win_len]
        ctx, tgt = window[:-1], window[-1]
        yield ctx, tgt, (i + win_len - 1)

def build_examples_global_time_split(
    user2seq: Dict[int, List[Tuple[int, int]]],
    win_len: int = 10,
    stride: int = 1,
    q_orig: float = 0.5,
    q_test_start: float = 0.8
):
    ts_all = np.array([ts for pairs in user2seq.values() for _, ts in pairs], dtype=np.int64)
    t0 = int(np.quantile(ts_all, q_orig))
    t1 = int(np.quantile(ts_all, q_test_start))
    if t1 <= t0: t1 = t0 + 1

    original, finetune, test = [], [], []
    users_o, items_o, inter_o = set(), set(), 0
    users_f, items_f, inter_f = set(), set(), 0
    users_t, items_t, inter_t = set(), set(), 0

    original_stride1 = []
    all_items = set()

    for uid, pairs in user2seq.items():
        tss = [ts for _, ts in pairs]

        mids_o = [m for m, ts in pairs if ts < t0]
        tss_o = [ts for ts in tss if ts < t0]

        mids_f = [m for m, ts in pairs if t0 <= ts < t1]
        tss_f = [ts for ts in tss if t0 <= ts < t1]

        mids_t = [m for m, ts in pairs if ts >= t1]
        tss_t = [ts for ts in tss if ts >= t1]

        # --- NEW: accumulate raw positive interactions per block ---
        if len(mids_o) > 0:
            users_o.add(uid)
            items_o.update(mids_o)
            inter_o += len(mids_o)

        if len(mids_f) > 0:
            users_f.add(uid)
            items_f.update(mids_f)
            inter_f += len(mids_f)

        if len(mids_t) > 0:
            users_t.add(uid)
            items_t.update(mids_t)
            inter_t += len(mids_t)


        if len(mids_o) >= win_len:
            for (ctx, tgt, tgt_idx) in sliding_windows(mids_o, win_len, stride, align="end"):
                ts = tss_o[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                original.append({
                    "user_id": uid, "timestamp": int(ts),
                    "prompt": prompt, "target": target,
                })
                all_items.update(ctx);
                all_items.add(tgt)


            for (ctx, tgt, tgt_idx) in sliding_windows(mids_o, win_len, 1, align="end"):
                ts = tss_o[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                original_stride1.append({
                    "user_id": uid, "timestamp": int(ts),
                    "prompt": prompt, "target": target,
                })
                all_items.update(ctx);
                all_items.add(tgt)

        if len(mids_f) >= win_len:
            for (ctx, tgt, tgt_idx) in sliding_windows(mids_f, win_len, stride, align="end"):
                ts = tss_f[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                finetune.append({
                    "user_id": uid, "timestamp": int(ts),
                    "prompt": prompt, "target": target,
                })
                all_items.update(ctx);
                all_items.add(tgt)

        if len(mids_t) >= win_len:
            for (ctx, tgt, tgt_idx) in sliding_windows(mids_t, win_len, stride, align="end"):
                ts = tss_t[tgt_idx]
                prompt = " ".join(format_item_token(m) for m in ctx)
                target = format_item_token(tgt)
                test.append({
                    "user_id": uid, "timestamp": int(ts),
                    "prompt": prompt, "target": target,
                })
                all_items.update(ctx);
                all_items.add(tgt)

    stats = {
        "O": {"users": len(users_o), "items": len(items_o), "interactions": inter_o},
        "F": {"users": len(users_f), "items": len(items_f), "interactions": inter_f},
        "T": {"users": len(users_t), "items": len(items_t), "interactions": inter_t},
    }

    return {
        "original": original, "finetune": finetune, "test": test, "original_stride1": original_stride1,
        "all_items": sorted(all_items),
        "t0": t0, "t1": t1,
        "stats": stats,  # <-- NEW
    }

def save_stream(path, user2seq: Dict[int, List[Tuple[int,int]]], t0: int, t1: int, which: str):
    """
    which: 'original' | 'finetune'
    original: ts < t0
    finetune: t0 <= ts < t1
    """
    rows = []
    for uid, pairs in user2seq.items():
        if which == 'original':
            seg = [(m,ts) for m,ts in pairs if ts < t0]
        elif which == 'finetune':
            seg = [(m,ts) for m,ts in pairs if t0 <= ts < t1]
        else:
            raise ValueError("which must be 'original' or 'finetune'")
        if len(seg) < 2:  
            continue
        mids = [m for m,_ in seg]
        tss  = [ts for _,ts in seg]
        rows.append({
            "user_id": uid,
            "items": " ".join(format_item_token(m) for m in mids),
            "timestamps": " ".join(str(ts) for ts in tss),
        })
    save_jsonl(path, rows)

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="")
    args.add_argument("--output_dir", type=str, default="")
    args.add_argument("--max_len", type=int, default="")
    args.add_argument("--save_stream", action="store_true", default=True)
    args = args.parse_args()
    ratings_path = os.path.join(args.data_dir, "ratings.dat")
    if not os.path.exists(ratings_path): raise FileNotFoundError(ratings_path)

    print("Loading MovieLens ...")
    user2seq = load_mlm1m_ratings(ratings_path)

    print("Building examples ...")
    pack = build_examples_global_time_split(user2seq, win_len=args.max_len+1, stride=5, q_orig=0.5, q_test_start=0.8)
    s = pack["stats"]
    print("[Block Stats]")
    print(f"O: users={s['O']['users']}, items={s['O']['items']}, interactions={s['O']['interactions']}")
    print(f"F: users={s['F']['users']}, items={s['F']['items']}, interactions={s['F']['interactions']}")
    print(f"T: users={s['T']['users']}, items={s['T']['items']}, interactions={s['T']['interactions']}")
    data_original_stride1 = pack["original_stride1"]
    data_original = pack["original"]
    data_finetune = pack["finetune"]
    data_test = pack["test"]
    item_ids = pack["all_items"]

    save_jsonl(os.path.join(args.output_dir, "original.jsonl"), data_original)
    save_jsonl(os.path.join(args.output_dir, "original_stride1.jsonl"), data_original_stride1)
    save_jsonl(os.path.join(args.output_dir, "finetune.jsonl"), data_finetune)
    save_jsonl(os.path.join(args.output_dir, "test.jsonl"), data_test)

    if args.save_stream:
        save_stream(os.path.join(args.output_dir, "original_stream.jsonl"), user2seq, pack["t0"], pack["t1"],
                    which="original")
        save_stream(os.path.join(args.output_dir, "finetune_stream.jsonl"), user2seq, pack["t0"], pack["t1"],
                    which="finetune")
    with open(os.path.join(args.output_dir, "item_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"item_ids": item_ids}, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": {
                    "original": len(data_original),
                    "finetune": len(data_finetune),
                    "test": len(data_test),
                    "num_items": len(item_ids),
                },
                "t0": pack["t0"],
                "t1": pack["t1"],
                "schema": {"prompt": "str", "target": "str", "user_id": "int", "timestamp": "int"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
if __name__ == "__main__":
    main()
