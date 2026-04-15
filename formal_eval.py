"""
formal_eval.py — formal reasoning benchmark for TensionLM

Tests the model on syllogisms, equations, and proof-style prompts.
Scores each answer and reports accuracy vs a keyword-match ground truth.

Usage:
    python formal_eval.py --checkpoint checkpoints/stage3_math_117m/ckpt_0014000.pt
    python formal_eval.py --checkpoint checkpoints/stage3_math_117m/ckpt_0014000.pt --temp 0.3
"""

import argparse
import sys
import torch


BENCHMARK = [
    # ── Syllogisms ──────────────────────────────────────────────────────────────
    {
        "category": "syllogism",
        "prompt": "All men are mortal. Socrates is a man. Therefore Socrates is",
        "accept": ["mortal"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "All cats are animals. Whiskers is a cat. Therefore Whiskers is",
        "accept": ["animal", "animals"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "No fish are mammals. All sharks are fish. Therefore no sharks are",
        "accept": ["mammal", "mammals"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "All squares are rectangles. All rectangles are quadrilaterals. Therefore all squares are",
        "accept": ["quadrilateral", "quadrilaterals"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "If it is raining then the ground is wet. It is raining. Therefore the ground is",
        "accept": ["wet"],
        "reject": ["dry"],
    },
    {
        "category": "syllogism",
        "prompt": "Either the butler did it or the gardener did it. The butler did not do it. Therefore the gardener",
        "accept": ["did it", "did", "guilty"],
        "reject": [],
    },
    # ── Transitivity ────────────────────────────────────────────────────────────
    {
        "category": "transitivity",
        "prompt": "A is greater than B. B is greater than C. Therefore A is greater than",
        "accept": ["c", "C"],
        "reject": ["b", "B"],
    },
    {
        "category": "transitivity",
        "prompt": "If P implies Q and Q implies R then P implies",
        "accept": ["r", "R"],
        "reject": [],
    },
    {
        "category": "transitivity",
        "prompt": "John is taller than Mary. Mary is taller than Sam. Therefore John is taller than",
        "accept": ["sam", "Sam"],
        "reject": [],
    },
    # ── Basic arithmetic ─────────────────────────────────────────────────────────
    {
        "category": "arithmetic",
        "prompt": "2 plus 2 equals",
        "accept": ["4", "four"],
        "reject": [],
    },
    {
        "category": "arithmetic",
        "prompt": "The square root of 9 is",
        "accept": ["3", "three"],
        "reject": [],
    },
    {
        "category": "arithmetic",
        "prompt": "7 multiplied by 8 equals",
        "accept": ["56", "fifty-six", "fifty six"],
        "reject": [],
    },
    {
        "category": "arithmetic",
        "prompt": "The sum of the angles in a triangle is",
        "accept": ["180", "one hundred and eighty", "π", "pi"],
        "reject": [],
    },
    # ── Calculus ─────────────────────────────────────────────────────────────────
    {
        "category": "calculus",
        "prompt": "The derivative of x squared is",
        "accept": ["2x", "2 x", "2*x"],
        "reject": [],
    },
    {
        "category": "calculus",
        "prompt": "The derivative of a constant is",
        "accept": ["0", "zero"],
        "reject": [],
    },
    {
        "category": "calculus",
        "prompt": "The integral of 1 dx is",
        "accept": ["x", "x +", "x+"],
        "reject": [],
    },
    {
        "category": "calculus",
        "prompt": "The limit as x approaches infinity of 1 over x is",
        "accept": ["0", "zero"],
        "reject": [],
    },
    # ── Algebra ──────────────────────────────────────────────────────────────────
    {
        "category": "algebra",
        "prompt": "If x plus 3 equals 7 then x equals",
        "accept": ["4", "four"],
        "reject": [],
    },
    {
        "category": "algebra",
        "prompt": "The solutions to x squared minus 4 equals 0 are x equals 2 and x equals",
        "accept": ["-2", "negative 2", "minus 2"],
        "reject": [],
    },
    {
        "category": "algebra",
        "prompt": "If 2x equals 10 then x equals",
        "accept": ["5", "five"],
        "reject": [],
    },
    # ── Definitions ──────────────────────────────────────────────────────────────
    {
        "category": "definition",
        "prompt": "A prime number is a number greater than 1 that has no divisors other than 1 and",
        "accept": ["itself", "itself."],
        "reject": [],
    },
    {
        "category": "definition",
        "prompt": "The Pythagorean theorem states that in a right triangle a squared plus b squared equals",
        "accept": ["c squared", "c^2", "c²"],
        "reject": [],
    },
    {
        "category": "definition",
        "prompt": "An even number is divisible by",
        "accept": ["2", "two"],
        "reject": [],
    },
]


def score(output: str, accept: list, reject: list) -> bool:
    out_lower = output.lower()
    # Must contain at least one accept term
    hit = any(a.lower() in out_lower for a in accept)
    # Must not contain any reject terms
    no_reject = not any(r.lower() in out_lower for r in reject)
    return hit and no_reject


def load_model(ckpt_path: str):
    from model import TensionConfig, TensionLM, generate as _generate
    print(f"Loading {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = TensionConfig(**ckpt["cfg"])
    model = TensionLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(ckpt["tok_path"])

    ppl = ckpt.get("val_ppl", "?")
    ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
    print(f"Model  : {model.num_params:,} params  |  val ppl {ppl_str}")
    print(f"Step   : {ckpt.get('step', '?')}\n")
    return model, tokenizer, _generate


def run_eval(ckpt_path: str, max_new: int = 40, temp: float = 0.3, top_p: float = 0.9):
    model, tokenizer, _generate = load_model(ckpt_path)

    results = []
    categories = {}

    for item in BENCHMARK:
        enc     = tokenizer.encode(item["prompt"])
        ids_out = _generate(model, enc.ids, max_new=max_new, temp=temp, top_p=top_p, rep_penalty=1.2)
        # Only score the generated portion (after the prompt)
        gen_ids  = ids_out[len(enc.ids):]
        gen_text = tokenizer.decode(gen_ids)
        correct  = score(gen_text, item["accept"], item["reject"])

        results.append({**item, "output": gen_text, "correct": correct})
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if correct:
            categories[cat]["correct"] += 1

        tick = "✓" if correct else "✗"
        print(f"[{tick}] [{item['category']:12s}] {item['prompt'][:60]}")
        print(f"       Expected: {item['accept']}  |  Got: {gen_text[:80].strip()!r}\n")

    # Summary
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    print("=" * 70)
    print(f"OVERALL: {correct}/{total}  ({100*correct/total:.1f}%)\n")
    print("By category:")
    for cat, d in categories.items():
        pct = 100 * d["correct"] / d["total"]
        print(f"  {cat:14s}: {d['correct']}/{d['total']}  ({pct:.0f}%)")
    print("=" * 70)
    return correct, total, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_new",    default=40,  type=int)
    p.add_argument("--temp",       default=0.3, type=float)
    p.add_argument("--top_p",      default=0.9, type=float)
    args = p.parse_args()
    run_eval(args.checkpoint, args.max_new, args.temp, args.top_p)


if __name__ == "__main__":
    main()
