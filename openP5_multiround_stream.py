import argparse
import copy
import gc
import importlib.util
import os
import shutil
import types
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _load_openp5_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("openP5_RAIE", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def aggressive_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _free_cuda(*objs):
    seen = set()
    for o in objs:
        if o is None:
            continue
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        try:
            target = o.module if hasattr(o, "module") else o
            if isinstance(target, torch.nn.Module):
                try:
                    target.to("cpu")
                except Exception:
                    pass
            if hasattr(o, "module"):
                try:
                    delattr(o, "module")
                except Exception:
                    pass
        except Exception:
            pass
    aggressive_gc()


@dataclass
class RoundSpec:
    name: str
    finetune_jsonl_path: str
    test_jsonl_path: str
    original_jsonl_path: str


class MultiRoundPredictor:
    """Multi-round incremental runner built on openP5_RAIE primitives."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.module = _load_openp5_module(os.path.join(os.path.dirname(__file__), "openP5_RAIE.py"))
        self._init_distributed()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() and self.local_rank >= 0 else "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.is_main = (not self.is_distributed) or dist.get_rank() == 0
        self.load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if self.is_main:
            os.makedirs(args.output_dir, exist_ok=True)

        self.rounds = self._build_rounds()
        self.base_output = os.path.join(args.output_dir, "round_0_base")

    # -------------------------------------------------
    # Distributed helpers
    # -------------------------------------------------
    def _init_distributed(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank < 0:
            return
        if dist.is_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

    # -------------------------------------------------
    # Round building
    # -------------------------------------------------
    def _build_rounds(self) -> List[RoundSpec]:
        finetune_list = [p for p in self.args.finetune_rounds.split(",") if p]
        test_list = [p for p in self.args.test_rounds.split(",") if p]
        name_list = [p for p in self.args.round_names.split(",") if p]
        if len(test_list) and len(test_list) != len(finetune_list):
            raise ValueError("test_rounds length must match finetune_rounds length")
        if len(name_list) and len(name_list) != len(finetune_list):
            raise ValueError("round_names length must match finetune_rounds length")

        rounds: List[RoundSpec] = []
        for idx, fpath in enumerate(finetune_list):
            tpath = test_list[idx] if idx < len(test_list) else (self.args.test_jsonl_path or fpath)
            rname = name_list[idx] if idx < len(name_list) else f"r{idx+1}"
            rounds.append(
                RoundSpec(
                    name=rname,
                    finetune_jsonl_path=fpath,
                    test_jsonl_path=tpath,
                    original_jsonl_path=self.args.original_jsonl_path,
                )
            )
        return rounds

    # -------------------------------------------------
    # Base prep + cloning
    # -------------------------------------------------
    def _prepare_base_once(self):
        if os.path.isdir(os.path.join(self.base_output, "base_with_new_tokens")):
            return
        base_args = copy.deepcopy(self.args)
        base_args.output_dir = self.base_output
        base_args.finetune_jsonl_path = self.rounds[0].finetune_jsonl_path if self.rounds else base_args.finetune_jsonl_path
        base_args.test_jsonl_path = self.rounds[0].test_jsonl_path if self.rounds else base_args.test_jsonl_path
        self.module.stage_pre(
            base_args,
            device=self.device,
            is_distributed=self.is_distributed,
            local_rank=self.local_rank,
            is_main=self.is_main,
            load_dtype=self.load_dtype,
        )

    @staticmethod
    def _clone_args_for_round(args: argparse.Namespace, spec: RoundSpec, output_dir: str, resume_base: str) -> argparse.Namespace:
        new_args = copy.deepcopy(args)
        new_args.output_dir = output_dir
        new_args.finetune_jsonl_path = spec.finetune_jsonl_path
        new_args.test_jsonl_path = spec.test_jsonl_path
        new_args.original_jsonl_path = spec.original_jsonl_path
        new_args.resume_base_dir = resume_base
        return new_args

    @staticmethod
    def _materialize_base(prev_dir: str, round_dir: str):
        if not os.path.isdir(prev_dir):
            raise FileNotFoundError(f"Previous base dir not found: {prev_dir}")
        for folder in ["base_with_new_tokens", "default", "default_for_raie"]:
            src = os.path.join(prev_dir, folder)
            dst = os.path.join(round_dir, folder)
            if os.path.isdir(src) and not os.path.isdir(dst):
                shutil.copytree(src, dst, dirs_exist_ok=True)

    # -------------------------------------------------
    # Execution
    # -------------------------------------------------
    def run_method(self, method: str):
        method = method.lower()
        if method not in {"lora", "replay", "lwf", "lsat", "raie", "mole"}:
            raise ValueError(f"Unsupported method: {method}")

        self._prepare_base_once()
        resume_base = os.path.join(self.base_output, "base_with_new_tokens")
        prev_dir = self.base_output

        for idx, spec in enumerate(self.rounds, start=1):
            round_dir = os.path.join(self.args.output_dir, f"round_{idx}_{spec.name}")
            os.makedirs(round_dir, exist_ok=True)
            self._materialize_base(prev_dir, round_dir)
            round_args = self._clone_args_for_round(self.args, spec, round_dir, resume_base)

            stage_fn = {
                "lora": self.module.stage_lora,
                "replay": self.module.stage_replay,
                "lwf": self.module.stage_lwf,
                "lsat": self.module.stage_lsat,
                "raie": self.module.stage_raie,
                "mole": self.module.stage_mole,
            }[method]

            if method == "raie" and self.is_distributed and dist.get_rank() != 0:
                if self.is_main:
                    print(f"[INFO] Skip RAIE on non-zero rank {dist.get_rank()}")
                continue

            stage_fn(
                round_args,
                device=self.device,
                is_distributed=self.is_distributed if method != "raie" else False,
                local_rank=self.local_rank,
                is_main=self.is_main,
                load_dtype=self.load_dtype,
            )

            prev_dir = round_dir
            resume_base = os.path.join(round_dir, "base_with_new_tokens") if os.path.isdir(os.path.join(round_dir, "base_with_new_tokens")) else resume_base

    def close(self):
        aggressive_gc()
        if self.is_distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


# -------------------------------------------------
# Argument parser
# -------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', type=str, default='/home/zj/model/Llama-2-7b-hf')
    ap.add_argument('--data_dir', type=str, default='/home/zj/code/ml-10M100K/')
    ap.add_argument('--output_dir', type=str, default='./runs/openP5_multiround')

    ap.add_argument('--train_jsonl_path', type=str, default='')
    ap.add_argument('--original_jsonl_path', type=str, default='')
    ap.add_argument('--test_jsonl_path', type=str, default='')
    ap.add_argument('--item_ids_path', type=str, default='')
    ap.add_argument('--finetune_jsonl_path', type=str, default='')

    ap.add_argument('--item_indexing', type=str, choices=['sequential', 'random', 'collaborative'], default='sequential')
    ap.add_argument('--min_user_len', type=int, default=5)
    ap.add_argument('--max_history', type=int, default=20)

    # training
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup_ratio', type=float, default=0.05)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--fp16', type=bool, default=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--final_token_only_loss', action='store_true', default=True)

    # LoRA
    ap.add_argument('--use_lora', action='store_true', default=True)
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--grad_checkpointing', action='store_true')

    # eval
    ap.add_argument('--topk', type=int, nargs='+', default=[5, 10, 20])

    # finetune shared
    ap.add_argument('--finetune_batch_size', type=int, default=32)
    ap.add_argument('--finetune_epochs', type=int, default=3)

    # Replay
    ap.add_argument('--replay_ratio', type=float, default=0.3)

    # LwF
    ap.add_argument('--lwf_T', type=float, default=2.0)
    ap.add_argument('--lwf_alpha', type=float, default=0.5)

    # LSAT
    ap.add_argument('--lsat_alpha', type=float, default=0.6)
    ap.add_argument('--lsat_long_epochs', type=int, default=1)
    ap.add_argument('--lsat_short_epochs', type=int, default=3)

    # RAIE
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--gap_thr", type=float, default=0.02)
    ap.add_argument("--orig_mix_ratio", type=float, default=0.3)
    ap.add_argument("--soft_rep_factor", type=int, default=2)
    ap.add_argument("--raie_epochs_per_region", type=int, default=3)
    ap.add_argument("--lambda_anchor", type=float, default=1e-4)

    # MoLE
    ap.add_argument('--mole_num_experts', type=int, default=3)
    ap.add_argument('--mole_gating_hidden', type=int, default=128)
    ap.add_argument('--mole_temp', type=float, default=0.7)
    ap.add_argument('--mole_balance', type=float, default=0.01)

    # multi-round controls
    ap.add_argument('--finetune_rounds', type=str, default='', help='Comma-separated finetune jsonl paths per round')
    ap.add_argument('--test_rounds', type=str, default='', help='Comma-separated test jsonl paths per round')
    ap.add_argument('--round_names', type=str, default='', help='Optional comma-separated round names')
    ap.add_argument('--methods', type=str, nargs='+', default=['lora'], choices=['lora','replay','lwf','lsat','raie','mole'])
    ap.add_argument('--resume_base_dir', type=str, default='', help='Optional existing base path; otherwise built in round_0_base')
    return ap


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.finetune_rounds.strip() == '':
        if not args.finetune_jsonl_path:
            raise ValueError("finetune_rounds or finetune_jsonl_path must be provided")
        args.finetune_rounds = args.finetune_jsonl_path
        args.test_rounds = args.test_rounds or args.test_jsonl_path or args.finetune_jsonl_path

    runner = MultiRoundPredictor(args)
    try:
        for method in args.methods:
            runner.run_method(method)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
