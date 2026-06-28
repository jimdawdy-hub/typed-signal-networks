from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ai_unity.text_data import resolve_text_dataset, stream_text_rows
from ai_unity.utils import ensure_dir, resolve_device, seed_everything, write_json


GPT2_VOCAB_SIZE = 50_257


@dataclass
class TinyLMConfig:
    vocab_size: int = GPT2_VOCAB_SIZE
    seq_len: int = 128
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.1
    model_kind: str = "transformer"
    num_experts: int = 4
    top_k: int = 2


class FeedForward(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), x.new_zeros(())


class TokenMoEFeedForward(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        if config.top_k < 1 or config.top_k > config.num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.router = nn.Linear(config.d_model, config.num_experts)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(x)[0] for expert in self.experts], dim=2)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        selected = torch.gather(expert_outputs, 2, gather_idx)
        output = (selected * top_probs.unsqueeze(-1)).sum(dim=2)

        avg_probs = router_probs.reshape(-1, self.num_experts).mean(dim=0)
        ideal = torch.ones_like(avg_probs) / self.num_experts
        lb_loss = F.mse_loss(avg_probs, ideal)
        return output, lb_loss

    @torch.no_grad()
    def routing_summary(self, x: torch.Tensor) -> dict[str, float | list[float]]:
        router_probs = F.softmax(self.router(x), dim=-1)
        _, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        usage = torch.bincount(top_idx.reshape(-1), minlength=self.num_experts).float()
        usage = usage / usage.sum().clamp_min(1)
        entropy = -(router_probs * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return {"usage_fraction": usage.cpu().tolist(), "entropy": float(entropy.cpu()), "collapse_rate": float(usage.max().cpu())}


class TinyTransformerBlock(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        if config.model_kind == "moe":
            self.ffn = TokenMoEFeedForward(config)
        elif config.model_kind == "transformer":
            self.ffn = FeedForward(config)
        else:
            raise ValueError(f"Unsupported model_kind: {config.model_kind}")

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        ffn_out, lb_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, lb_loss


class TinyTransformerLM(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TinyTransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured {self.config.seq_len}.")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.drop(self.token_embed(input_ids) + self.pos_embed(positions))
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        lb_loss = x.new_zeros(())
        for block in self.blocks:
            x, block_lb = block(x, causal_mask)
            lb_loss = lb_loss + block_lb
        lb_loss = lb_loss / max(len(self.blocks), 1)
        x = self.norm(x)
        return self.lm_head(x), lb_loss

    @torch.no_grad()
    def routing_summaries(self, input_ids: torch.Tensor) -> list[dict[str, float | list[float]]]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.drop(self.token_embed(input_ids) + self.pos_embed(positions))
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        summaries = []
        for block in self.blocks:
            h = block.attn_norm(x)
            attn_out, _ = block.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
            x = x + block.attn_dropout(attn_out)
            normed = block.ffn_norm(x)
            if isinstance(block.ffn, TokenMoEFeedForward):
                summaries.append(block.ffn.routing_summary(normed))
            ffn_out, _ = block.ffn(normed)
            x = x + ffn_out
        return summaries


class TokenWindowSampler:
    def __init__(self, token_path: Path, seq_len: int, seed: int):
        tokens = np.load(token_path, mmap_mode="r")
        if tokens.ndim != 1:
            raise ValueError(f"Expected flat token array, got shape {tokens.shape}.")
        if len(tokens) <= seq_len + 1:
            raise ValueError(f"Need more than {seq_len + 1} tokens, got {len(tokens)}.")
        self.tokens = tokens
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.seq_len - 1
        starts = self.rng.integers(0, max_start, size=batch_size)
        x = np.stack([self.tokens[start : start + self.seq_len] for start in starts])
        y = np.stack([self.tokens[start + 1 : start + self.seq_len + 1] for start in starts])
        return (
            torch.as_tensor(x, dtype=torch.long, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device),
        )


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def prepare_worker(args: argparse.Namespace) -> None:
    import tiktoken

    out = ensure_dir(args.output_dir)
    token_path = out / "tokens.npy"
    meta_path = out / "metadata.json"
    encoding = tiktoken.get_encoding(args.encoding)
    eos_id = encoding.eot_token

    tokens: list[int] = []
    docs = 0
    chars = 0
    started = time.time()
    for row in stream_text_rows(args.dataset, split=args.split):
        text = row.get(args.text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = encoding.encode(text, disallowed_special=())
        if not encoded:
            continue
        tokens.extend(encoded)
        tokens.append(eos_id)
        docs += 1
        chars += len(text)
        if len(tokens) >= args.max_tokens:
            tokens = tokens[: args.max_tokens]
            break
        if args.max_docs is not None and docs >= args.max_docs:
            break

    arr = np.asarray(tokens, dtype=np.uint16)
    np.save(token_path, arr)
    write_json(
        meta_path,
        {
            "dataset": args.dataset,
            "resolved_dataset": resolve_text_dataset(args.dataset),
            "split": args.split,
            "text_column": args.text_column,
            "encoding": args.encoding,
            "docs": docs,
            "chars": chars,
            "tokens": int(arr.shape[0]),
            "token_path": str(token_path),
            "seconds": time.time() - started,
        },
    )
    print(f"Wrote {arr.shape[0]:,} tokens from {docs:,} docs to {token_path}", flush=True)
    os._exit(0)


def prepare(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    cmd = [
        sys.executable,
        "-m",
        "ai_unity.text_lm",
        "_prepare_worker",
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--text-column",
        args.text_column,
        "--encoding",
        args.encoding,
        "--max-tokens",
        str(args.max_tokens),
        "--output-dir",
        str(out),
    ]
    if args.max_docs is not None:
        cmd.extend(["--max-docs", str(args.max_docs)])
    completed = subprocess.run(cmd, cwd=Path.cwd())
    token_path = out / "tokens.npy"
    meta_path = out / "metadata.json"
    if completed.returncode != 0 or not token_path.exists() or not meta_path.exists():
        raise RuntimeError(f"FineWeb token preparation failed with exit code {completed.returncode}.")
    print(f"Prepared token cache under {out}")


@torch.no_grad()
def evaluate(model: nn.Module, sampler: TokenWindowSampler, batch_size: int, batches: int, device: torch.device) -> dict:
    model.eval()
    losses = []
    aux_losses = []
    correct = 0
    total = 0
    for _ in range(batches):
        x, y = sampler.batch(batch_size, device)
        logits, aux_loss = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
        aux_losses.append(float(aux_loss.item()))
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()
    mean_loss = float(sum(losses) / max(len(losses), 1))
    return {
        "loss": mean_loss,
        "aux_loss": float(sum(aux_losses) / max(len(aux_losses), 1)),
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "token_accuracy": correct / max(total, 1),
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_index)
    out = ensure_dir(args.output_dir)
    config = TinyLMConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        model_kind=args.model_kind,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    model = TinyTransformerLM(config).to(device)
    sampler = TokenWindowSampler(args.token_path, seq_len=args.seq_len, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Params: {count_params(model):,}")
    print(f"Tokens: {len(sampler.tokens):,}")

    history = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sampler.batch(args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            logits, aux_loss = model(x)
            main_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss = main_loss + args.lb_weight * aux_loss
        scaler.scale(loss).backward()
        if args.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            eval_metrics = evaluate(model, sampler, args.batch_size, args.eval_batches, device)
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_main_loss": float(main_loss.item()),
                "train_aux_loss": float(aux_loss.item()),
                "eval_loss": eval_metrics["loss"],
                "eval_aux_loss": eval_metrics["aux_loss"],
                "eval_perplexity": eval_metrics["perplexity"],
                "eval_token_accuracy": eval_metrics["token_accuracy"],
                "seconds": time.time() - started,
            }
            history.append(row)
            print(
                f"step={step}/{args.steps} train_loss={row['train_loss']:.4f} "
                f"eval_loss={row['eval_loss']:.4f} ppl={row['eval_perplexity']:.2f} "
                f"tok_acc={row['eval_token_accuracy']:.2%}"
            )
            model.train()

    payload = {
        "config": asdict(config),
        "params": count_params(model),
        "token_path": str(args.token_path),
        "args": vars(args),
        "history": history,
        "final": history[-1] if history else None,
    }
    if config.model_kind == "moe":
        x, _ = sampler.batch(args.batch_size, device)
        payload["routing"] = model.routing_summaries(x)
    write_json(out / "tiny_transformer_lm_results.json", payload)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "args": vars(args)}, out / "tiny_transformer_lm.pt")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny FineWeb language-modeling tools.")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", default="fineweb-edu-1b")
    prep.add_argument("--split", default="train")
    prep.add_argument("--text-column", default="text")
    prep.add_argument("--encoding", default="gpt2")
    prep.add_argument("--max-tokens", type=int, default=250_000)
    prep.add_argument("--max-docs", type=int, default=None)
    prep.add_argument("--output-dir", type=Path, required=True)

    worker = sub.add_parser("_prepare_worker")
    worker.add_argument("--dataset", required=True)
    worker.add_argument("--split", default="train")
    worker.add_argument("--text-column", default="text")
    worker.add_argument("--encoding", default="gpt2")
    worker.add_argument("--max-tokens", type=int, required=True)
    worker.add_argument("--max-docs", type=int, default=None)
    worker.add_argument("--output-dir", type=Path, required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--token-path", type=Path, required=True)
    tr.add_argument("--output-dir", type=Path, required=True)
    tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    tr.add_argument("--gpu-index", type=int, default=0)
    tr.add_argument("--seed", type=int, default=123)
    tr.add_argument("--steps", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--seq-len", type=int, default=128)
    tr.add_argument("--d-model", type=int, default=192)
    tr.add_argument("--n-heads", type=int, default=6)
    tr.add_argument("--n-layers", type=int, default=4)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--model-kind", choices=["transformer", "moe"], default="transformer")
    tr.add_argument("--num-experts", type=int, default=4)
    tr.add_argument("--top-k", type=int, default=2)
    tr.add_argument("--lb-weight", type=float, default=0.01)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight-decay", type=float, default=0.1)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--log-every", type=int, default=25)
    tr.add_argument("--eval-batches", type=int, default=8)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "_prepare_worker":
        prepare_worker(args)
    elif args.command == "train":
        train(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
