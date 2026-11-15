"""Full fine-tuning for the MoM study (specialists, pooled, tagged-pooled, random partitions).

Design choices (see notebook 2026-10-29):
- Loss on assistant tokens only. The prompt is rendered with the Qwen3 chat template and
  enable_thinking=False exactly as at evaluation time, so train and eval formats match.
- No packing: examples never share an attention window. Batches are length-grouped to limit padding.
- One optimizer update = at least --tokens-per-update supervised+prompt (non-pad) tokens, accumulated
  over micro-batches of at most --micro-tokens padded tokens. Loss is normalised by the number of
  supervised tokens in the whole update.
- AdamW (fp32 master weights via autocast bf16), cosine schedule with linear warmup.

Input: one or more JSONL files of {"id","messages":[{"role":"user",...},{"role":"assistant",...}],"domain",...}.
The mixture/budget is decided upstream (scripts that write the training file); this script trains on
exactly what it is given, for --epochs passes, in a seeded shuffled order.
"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--tokens-per-update", type=int, default=65536)
    p.add_argument("--micro-tokens", type=int, default=32768, help="max padded tokens per micro-batch")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--domain-tag", action="store_true", help="prefix user message with [domain: X]")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--save-epochs", action="store_true", help="also save after each full epoch")
    p.add_argument("--limit", type=int, default=0, help="debug: use first N examples")
    p.add_argument("--lora-rank", type=int, default=0, help="LoRA on all linear layers (0 = full fine-tuning)")
    p.add_argument("--lora-alpha", type=float, default=0, help="default 2 x rank")
    return p.parse_args()


def render(tok, ex, domain_tag):
    msgs = [dict(m) for m in ex["messages"]]
    assert msgs[-1]["role"] == "assistant", ex.get("id")
    if domain_tag:
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"[domain: {ex['domain']}]\n" + m["content"]
                break
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    # Tokenise prompt and answer separately (as at inference: the model continues after the prompt tokens).
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    f_ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    return f_ids, labels


def make_updates(lengths, args, rng):
    """Group example indices into micro-batches (length-sorted within shuffled chunks) and
    micro-batches into optimizer updates of >= tokens_per_update real tokens."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = 2048
    micro = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in part:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > args.micro_tokens:
                micro.append(cur); cur, cur_max = [], 0; m = lengths[i]
            cur.append(i); cur_max = m
        if cur:
            micro.append(cur)
    rng.shuffle(micro)
    updates, cur, cur_tok = [], [], 0
    for mb in micro:
        cur.append(mb); cur_tok += sum(lengths[i] for i in mb)
        if cur_tok >= args.tokens_per_update:
            updates.append(cur); cur, cur_tok = [], 0
    if cur:
        updates.append(cur)
    return updates


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    exs = []
    for path in args.data:
        with open(path) as f:
            for line in f:
                exs.append(json.loads(line))
    if args.limit:
        exs = exs[:args.limit]
    data, dropped = [], 0
    for ex in exs:
        ids, lab = render(tok, ex, args.domain_tag)
        if len(ids) > args.max_len or all(l == -100 for l in lab):
            dropped += 1; continue
        data.append((ids, lab))
    lengths = [len(d[0]) for d in data]
    sup_tokens = sum(sum(1 for l in d[1] if l != -100) for d in data)
    print(json.dumps({"examples": len(data), "dropped": dropped, "tokens": sum(lengths),
                      "supervised_tokens": sup_tokens}), flush=True)

    rng = random.Random(args.seed)
    full_epochs = int(math.floor(args.epochs)); frac = args.epochs - full_epochs
    plan = []
    for e in range(full_epochs):
        plan += [(e, u) for u in make_updates(lengths, args, rng)]
    if frac > 0:
        ups = make_updates(lengths, args, rng)
        plan += [(full_epochs, u) for u in ups[:max(1, int(len(ups) * frac))]]
    total = len(plan)
    warm = max(1, int(total * args.warmup))

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").cuda()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    core = model  # peft injects LoRA layers in place, so core.model / core.lm_head run with the adapters
    if args.lora_rank:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha or 2 * args.lora_rank, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(json.dumps({"lora_rank": args.lora_rank, "trainable_params": n_tr}), flush=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    t0, seen = time.time(), 0
    model.train()
    last_epoch = 0
    for step, (epoch, update) in enumerate(plan):
        if args.save_epochs and epoch != last_epoch:
            save(model, tok, os.path.join(args.out, f"epoch{last_epoch + 1}"), args, step)
            last_epoch = epoch
        n_sup = sum(sum(1 for l in data[i][1] if l != -100) for mb in update for i in mb)
        loss_sum = 0.0
        for mb in update:
            L = max(lengths[i] for i in mb)
            ids = torch.full((len(mb), L), pad, dtype=torch.long)
            lab = torch.full((len(mb), L), -100, dtype=torch.long)
            att = torch.zeros((len(mb), L), dtype=torch.long)
            for r, i in enumerate(mb):
                n = lengths[i]
                ids[r, :n] = torch.tensor(data[i][0]); lab[r, :n] = torch.tensor(data[i][1]); att[r, :n] = 1
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hid = core.model(input_ids=ids, attention_mask=att).last_hidden_state
                # loss only at supervised positions; logits computed in chunks (vocab is ~152K)
                h = hid[:, :-1][lab[:, 1:] != -100]
                tgt = lab[:, 1:][lab[:, 1:] != -100]
                loss = 0.0
                for c in range(0, h.size(0), 4096):
                    lg = core.lm_head(h[c:c + 4096]).float()
                    loss = loss + torch.nn.functional.cross_entropy(lg, tgt[c:c + 4096], reduction="sum")
                loss = loss / n_sup
            loss.backward(); loss_sum += loss.item()
            seen += int(att.sum())
            del hid, h, loss
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "of": total, "epoch": epoch, "loss": round(loss_sum, 5), "gnorm": round(gn, 4),
               "lr": sched.get_last_lr()[0], "tok_s": round(seen / (time.time() - t0))}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 20 == 0 or step + 1 == total:
            print(json.dumps(rec), flush=True)
    save(model, tok, args.out, args, total)
    print("TRAIN_DONE", flush=True)


def save(model, tok, path, args, step):
    os.makedirs(path, exist_ok=True)
    if args.lora_rank:  # keep the adapter (its size is the memory cost) and save a merged copy for serving
        model.save_pretrained(os.path.join(path, "adapter"), safe_serialization=True)
        merged = model.merge_and_unload() if path == args.out else None
        if merged is None:  # epoch checkpoint: adapter only
            tok.save_pretrained(path); return
        model = merged
    # Base checkpoints stop only on <|endoftext|>; we train answers to end with <|im_end|>, so stop on both.
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if im_end not in eos:
        model.generation_config.eos_token_id = [im_end] + [e for e in eos if e is not None]
    model.to(torch.bfloat16).save_pretrained(path, safe_serialization=True) if path == args.out else \
        model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    with open(os.path.join(path, "train_args.json"), "w") as f:
        json.dump({**vars(args), "saved_at_step": step}, f, indent=1)


if __name__ == "__main__":
    main()
