import argparse
import importlib.util
import json
import os
import types
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, get_linear_schedule_with_warmup


# -----------------------------------------------
# Utilities for loading the original Bert4Rec_RAIE
# -----------------------------------------------

def _load_bert4rec_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("Bert4Rec_RAIE_ALL_NEW.py", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@dataclass
class StageResult:
    finetune_stage: str
    predict_stage: str
    predict_metrics: Dict[str, float]
    original_metrics: Dict[str, float]
    model_dir: str


class MultiStageRunner:
    """Orchestrates sequential fine-tune/eval cycles."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.module = _load_bert4rec_module(os.path.join(os.path.dirname(__file__), "Bert4Rec_RAIE_ALL_NEW.py"))

        # shared vocab + collators
        self.token2id, self.id2token, self.item_token_ids = self.module.load_item_vocab(args.data_dir)
        self.pad_id = self.token2id["[PAD]"]
        self.cls_id = self.token2id["[CLS]"]
        self.mask_id = self.token2id["[MASK]"]

        self.train_ds = self.module.StreamDataset(os.path.join(args.data_dir, "original_stream.jsonl"))
        self.train_collator = self.module.ClozeTrainCollator(
            self.token2id, args.max_len, self.pad_id, self.mask_id, self.cls_id, args.mask_prob
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.train_collator,
        )

        self.eval_collator = self.module.EvalCollator(
            self.token2id, args.max_len, self.mask_id, self.cls_id, self.pad_id
        )
        self.ft_collator = self.module.EvalCollator(
            self.token2id,
            args.max_len,
            self.mask_id,
            self.cls_id,
            self.pad_id,
            do_prompt_mask=args.do_prompt_mask,
            prompt_mask_prob=args.prompt_mask_prob,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Model helpers
    # -------------------------
    def _init_base_model(self) -> BertForMaskedLM:
        cfg = BertConfig(
            vocab_size=len(self.id2token),
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_hidden_layers,
            num_attention_heads=self.args.num_attention_heads,
            intermediate_size=self.args.intermediate_size,
            max_position_embeddings=self.args.max_len + 2,
            hidden_dropout_prob=self.args.dropout,
            attention_probs_dropout_prob=self.args.dropout,
            pad_token_id=self.pad_id,
        )
        return BertForMaskedLM(cfg).to(self.device)

    def _apply_lora(self, base_model: torch.nn.Module):
        target_modules = [s.strip() for s in self.args.lora_target.split(",") if s.strip()]
        lora_cfg = self.module.LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=self.module.TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
        )
        return self.module.get_peft_model(base_model, lora_cfg)

    # -------------------------
    # Training primitives
    # -------------------------
    def _train_on_stream(self, model: torch.nn.Module):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = torch.optim.AdamW(params, lr=self.args.lr)
        total_steps = len(self.train_loader) * max(self.args.epochs, 1)
        sched = get_linear_schedule_with_warmup(
            optim, int(total_steps * self.args.warmup_ratio), total_steps
        )
        for ep in range(1, self.args.epochs + 1):
            tr = self.module.train_one_epoch(
                model,
                self.train_loader,
                optim,
                sched,
                self.device,
                self.item_token_ids,
                objective=self.args.objective,
                grad_clip=self.args.grad_clip,
            )
            print(f"[O][Epoch {ep}] train_loss={tr['train_loss']:.4f}")

    def _finetune_general_lora(
        self, lora_model: torch.nn.Module, ft_loader: DataLoader, replay_buf=None, teacher=None
    ) -> float:
        ft_params = [p for p in lora_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        loss = 0.0
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_lora(
                lora_model,
                ft_loader,
                ft_optim,
                ft_sched,
                self.device,
                self.item_token_ids,
                self.pad_id,
                self.cls_id,
                self.mask_id,
                plugin=self.args.mode,
                replay_buf=replay_buf,
                teacher=teacher,
                lwf_T=self.args.lwf_T,
                lwf_alpha=self.args.lwf_alpha,
                grad_clip=self.args.grad_clip,
                replay_ratio=self.args.replay_ratio,
                log_every=100,
            )
            print(f"[F][{self.args.mode}][Epoch {ep}] loss={loss:.4f}")
        return loss

    # -------------------------
    # Stage execution
    # -------------------------
    def run(self) -> List[StageResult]:
        # base training
        base_model = self._init_base_model()
        self._train_on_stream(base_model)
        base_dir = os.path.join(self.args.output_dir, "base_model")
        os.makedirs(base_dir, exist_ok=True)
        base_model.save_pretrained(base_dir)
        with open(os.path.join(base_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"id2token": self.id2token}, f, ensure_ascii=False, indent=2)

        current_model_dir = base_dir
        results: List[StageResult] = []

        ft_stages = self.args.stages.split(",")
        pred_stages = self.args.pred_stages.split(",")
        if len(ft_stages) != len(pred_stages):
            raise ValueError("--stages and --pred_stages must align")

        original_eval_ds = self.module.NextItemDataset(os.path.join(self.args.data_dir, "original.jsonl"))
        original_loader = DataLoader(
            original_eval_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self.eval_collator,
        )

        for idx, (ft_name, pred_name) in enumerate(zip(ft_stages, pred_stages), start=1):
            print(f"===== Stage {idx}: finetune {ft_name} -> predict {pred_name} =====")
            ft_path = os.path.join(self.args.data_dir, f"{ft_name}.jsonl")
            pred_path = os.path.join(self.args.data_dir, f"{pred_name}.jsonl")
            if not os.path.exists(ft_path):
                raise FileNotFoundError(ft_path)
            if not os.path.exists(pred_path):
                raise FileNotFoundError(pred_path)

            ft_ds = self.module.NextItemDataset(ft_path)
            ft_loader = DataLoader(
                ft_ds,
                batch_size=self.args.finetune_batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=self.ft_collator,
            )
            pred_ds = self.module.NextItemDataset(pred_path)
            pred_loader = DataLoader(
                pred_ds,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=self.eval_collator,
            )

            model = BertForMaskedLM.from_pretrained(current_model_dir).to(self.device)

            if self.args.mode == "base":
                self._finetune_base(model, ft_loader)
                eval_model = model
            elif self.args.mode in {"lora", "lora_replay", "lora_lwf", "lsat", "raie"}:
                lora_model = self._apply_lora(model)
                eval_model = self._finetune_lora_based(lora_model, ft_loader)
            elif self.args.mode == "mole":
                eval_model = self._finetune_mole(model, ft_loader)
            else:  # pragma: no cover - parser guards values
                raise ValueError(f"Unsupported mode {self.args.mode}")

            predict_metrics = self.module.evaluate(
                eval_model,
                pred_loader,
                self.device,
                self.item_token_ids,
                topk=tuple(int(x) for x in self.args.topk.split(",")),
            )
            original_metrics = self.module.evaluate(
                eval_model,
                original_loader,
                self.device,
                self.item_token_ids,
                topk=tuple(int(x) for x in self.args.topk.split(",")),
            )

            stage_dir = os.path.join(self.args.output_dir, f"stage_{ft_name}")
            os.makedirs(stage_dir, exist_ok=True)
            eval_model.save_pretrained(stage_dir)
            current_model_dir = stage_dir

            results.append(
                StageResult(
                    finetune_stage=ft_name,
                    predict_stage=pred_name,
                    predict_metrics=predict_metrics,
                    original_metrics=original_metrics,
                    model_dir=stage_dir,
                )
            )
            print(f"[Eval][{ft_name}->{pred_name}] {predict_metrics}")
            print(f"[Eval][{ft_name}->original] {original_metrics}")

        self._save_summary(results)
        return results

    # -------------------------
    # Mode-specific helpers
    # -------------------------
    def _finetune_base(self, model: torch.nn.Module, ft_loader: DataLoader):
        ft_params = [p for p in model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_bert(
                model,
                ft_loader,
                ft_optim,
                ft_sched,
                self.device,
                self.item_token_ids,
                grad_clip=self.args.grad_clip,
                E0=None,
                ids_t=None,
                lambda_anchor=0.0,
            )
            print(f"[Base][F][Epoch {ep}] loss={loss:.4f}")

    def _finetune_lora_based(self, lora_model: torch.nn.Module, ft_loader: DataLoader):
        if self.args.mode == "lsat":
            long_adapter = "default"
            short_adapter = "short_term"
            if short_adapter not in getattr(lora_model, "peft_config", {}):
                lora_model.add_adapter(short_adapter, lora_model.peft_config[long_adapter])
            if hasattr(lora_model, "set_adapter") and hasattr(lora_model, "train_adapter"):
                lora_model.set_adapter(long_adapter)
                lora_model.train_adapter(long_adapter)
            self._train_on_stream(lora_model)
            if hasattr(lora_model, "set_adapter") and hasattr(lora_model, "train_adapter"):
                lora_model.set_adapter(short_adapter)
                lora_model.train_adapter(short_adapter)
            self._finetune_general_lora(lora_model, ft_loader)
            return lora_model

        replay_buf = None
        teacher = None
        if self.args.mode == "lora_replay":
            replay_buf = self.module.ReplayBuffer(self.train_ds, self.train_collator, self.args.finetune_batch_size)
        if self.args.mode == "lora_lwf":
            teacher = BertForMaskedLM.from_pretrained(os.path.join(self.args.output_dir, "base_model")).to(self.device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        self._finetune_general_lora(lora_model, ft_loader, replay_buf=replay_buf, teacher=teacher)
        return lora_model

    def _finetune_mole(self, base_model: torch.nn.Module, ft_loader: DataLoader):
        lora_model = self._apply_lora(base_model)
        adapters = ["default"] + [f"expert_{i}" for i in range(self.args.mole_num_experts - 1)]
        for name in adapters[1:]:
            lora_model.add_adapter(name, lora_model.peft_config["default"])
        mole_model = self.module.MoLEMixture(
            lora_model,
            adapter_names=tuple(adapters),
            hidden_size=self.args.hidden_size,
            num_items=len(self.item_token_ids),
            gate_hidden=self.args.mole_gating_hidden,
            temperature=self.args.mole_temp,
            balance_coef=self.args.mole_balance,
        ).to(self.device)

        ft_params = [p for p in mole_model.parameters() if p.requires_grad]
        ft_optim = torch.optim.AdamW(ft_params, lr=self.args.finetune_lr)
        ft_steps = len(ft_loader) * max(self.args.finetune_epochs, 1)
        ft_sched = get_linear_schedule_with_warmup(
            ft_optim, int(ft_steps * self.args.warmup_ratio), ft_steps
        )
        for ep in range(1, self.args.finetune_epochs + 1):
            loss = self.module.finetune_one_epoch_mole(
                mole_model, ft_loader, ft_optim, ft_sched, self.device, grad_clip=self.args.grad_clip
            )
            print(f"[MoLE][F][Epoch {ep}] loss={loss:.4f}")
        return mole_model

    # -------------------------
    # Summary
    # -------------------------
    def _save_summary(self, results: Iterable[StageResult]):
        summary = []
        for r in results:
            summary.append(
                {
                    "finetune_stage": r.finetune_stage,
                    "predict_stage": r.predict_stage,
                    "predict_metrics": r.predict_metrics,
                    "original_metrics": r.original_metrics,
                    "model_dir": r.model_dir,
                }
            )
        with open(os.path.join(self.args.output_dir, "multistage_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage runner for Bert4Rec_RAIE")
    p.add_argument("--data_dir", type=str, default="/home/zj/code/ml-10M100K_multistage")
    p.add_argument("--output_dir", type=str, default="/home/zj/code/ml-10M100K_multistage")
    p.add_argument("--mode", type=str, default="lora", choices=[
        "base",
        "lora",
        "lora_replay",
        "lora_lwf",
        "lsat",
        "raie",
        "mole",
    ])
    p.add_argument("--stages", type=str, default="f1,f2,f3,f4")
    p.add_argument("--pred_stages", type=str, default="f2,f3,f4,test")

    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--objective", type=str, default="cloze", choices=["cloze", "next"])
    p.add_argument("--mask_prob", type=float, default=0.15)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--finetune_epochs", type=int, default=3)
    p.add_argument("--finetune_batch_size", type=int, default=32)
    p.add_argument("--finetune_lr", type=float, default=5e-4)
    p.add_argument("--do_prompt_mask", action="store_true")
    p.add_argument("--prompt_mask_prob", type=float, default=0.15)

    p.add_argument("--replay_ratio", type=float, default=0.3)
    p.add_argument("--lwf_T", type=float, default=2.0)
    p.add_argument("--lwf_alpha", type=float, default=0.5)
    p.add_argument("--lsat_alpha", type=float, default=0.6)
    p.add_argument("--lsat_long_epochs", type=int, default=1)
    p.add_argument("--lsat_short_epochs", type=int, default=3)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target",
        type=str,
        default="query, key, value, intermediate.dense, output.dense, attention.output.dense",
    )

    p.add_argument("--mole_num_experts", type=int, default=3)
    p.add_argument("--mole_gating_hidden", type=int, default=128)
    p.add_argument("--mole_temp", type=float, default=0.7)
    p.add_argument("--mole_balance", type=float, default=0.01)

    p.add_argument("--topk", type=str, default="5,10,20")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    runner = MultiStageRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
