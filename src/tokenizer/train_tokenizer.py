#!/usr/bin/env python3
"""
Train a BPE tokenizer on a representative sample of text + code + chat data.
Uses the HuggingFace `tokenizers` library with ByteLevel BPE encoding.
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


DATA_SOURCES = {
    "text": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "field": "text",
        "max_samples": 200_000,
    },
    "code": {
        "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "config": None,
        "split": "train",
        "field": "solution",
        "filter_lang": "python",
        "max_samples": 50_000,
    },
    "chat": {
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "all",
        "split": "train",
        "field": "messages",
        "max_samples": 50_000,
    },
}

SPECIAL_TOKENS = [
    "<unk>", "<s>", "</s>", "<pad>",
    "<|user|>", "<|assistant|>", "<|system|>",
]


def text_iterator(domain, max_samples):
    cfg = DATA_SOURCES[domain]
    load_kwargs = {"path": cfg["dataset"], "split": cfg["split"], "streaming": True}
    if cfg.get("config"):
        load_kwargs["name"] = cfg["config"]

    ds = load_dataset(**load_kwargs)
    count = 0

    for example in ds:
        if max_samples and count >= max_samples:
            break
        if domain == "code" and cfg.get("filter_lang"):
            if example.get("lang") != cfg["filter_lang"]:
                continue
        field = cfg["field"]
        if field == "messages":
            text = format_chat(example.get(field, []))
        else:
            text = example.get(field, "")
        if text and len(text.strip()) > 20:
            count += 1
            yield text

    print(f"[{domain}] sampled {count} texts")


def format_chat(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        elif role == "system":
            parts.append(f"<|system|>\n{content}")
    return "\n".join(parts)


def train_tokenizer(vocab_size: int, output_path: str, samples_per_domain: dict):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def mixed_iterator():
        for domain, max_samples in samples_per_domain.items():
            if max_samples <= 0:
                continue
            for text in text_iterator(domain, max_samples):
                yield text

    tokenizer.train_from_iterator(mixed_iterator(), trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    tokenizer.save(output_path)
    print(f"\nTokenizer saved to {output_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def test_tokenizer(tokenizer, examples):
    print("\n=== Round-trip tests ===")
    all_ok = True
    for name, text in examples:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        ids = encoded.ids
        print(f"\n--- {name} ---")
        print(f"  Original ({len(text)} chars): {text[:120]}{'...' if len(text)>120 else ''}")
        print(f"  Token count: {len(ids)}")
        print(f"  Token IDs (first 20): {ids[:20]}")
        print(f"  Decoded ({len(decoded)} chars): {decoded[:120]}{'...' if len(decoded)>120 else ''}")
        if text == decoded:
            print(f"  Round-trip: OK")
        else:
            print(f"  Round-trip: MISMATCH!")
            all_ok = False

    status = "All round-trip tests passed" if all_ok else "Some round-trip tests FAILED"
    print(f"\n=== {status} ===")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--output", type=str, default="data/tokenizer/tokenizer.json")
    parser.add_argument("--text-samples", type=int, default=200_000)
    parser.add_argument("--code-samples", type=int, default=50_000)
    parser.add_argument("--chat-samples", type=int, default=50_000)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    if args.test_only:
        if not os.path.exists(args.output):
            print(f"Tokenizer not found at {args.output}", file=sys.stderr)
            sys.exit(1)
        tokenizer = Tokenizer.from_file(args.output)
    else:
        samples_per_domain = {
            "text": args.text_samples,
            "code": args.code_samples,
            "chat": args.chat_samples,
        }
        tokenizer = train_tokenizer(args.vocab_size, args.output, samples_per_domain)

    test_examples = [
        ("English text", "The quick brown fox jumps over the lazy dog. Machine learning is transforming how we build software and understand data."),
        ("French accents", "L'intelligence artificielle révolutionne l'éducation. Les élèves apprennent à coder dès l'âge de 10 ans. Voilà une idée géniale pour l'avenir!"),
        ("Python code", "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n"),
        ("Python with indent", "class Transformer(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.embed = nn.Embedding(config.vocab_size, config.d_model)\n        self.layers = nn.ModuleList([\n            DecoderLayer(config) for _ in range(config.n_layers)\n        ])\n"),
        ("Chat dialogue", "<|user|>\nWhat is the capital of France?\n<|assistant|>\nThe capital of France is Paris.\n<|user|>\nWhat about Lyon?\n<|assistant|>\nLyon is France's third-largest city.\n"),
    ]

    ok = test_tokenizer(tokenizer, test_examples)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
