"""
generate_logic_data.py — Synthetic stage 1 logical constraint dataset generator.

Generates pure logical inference examples under the Thinking System (TS) framework.
Stage 1 of the TS training curriculum: teach the model logical constraint structure
before any mathematical notation, domain knowledge, or ambiguous language.

Under TS:
  - Nodes are propositions (states)
  - Edges are constraints (implications, conjunctions, negations)
  - tau = unresolved tension between premises and conclusions
  - Learning = building a constraint graph that resolves correctly

The model should learn:
  1. Modus ponens: A→B, A ⊢ B
  2. Modus tollens: A→B, ¬B ⊢ ¬A
  3. Transitivity: A→B, B→C ⊢ A→C
  4. Conjunction: A, B ⊢ A∧B
  5. Disjunction: A∨B, ¬A ⊢ B
  6. Contradiction: A, ¬A ⊢ ⊥
  7. Proof by contradiction: assume ¬G, derive ⊥ ⊢ G
  8. Multi-step chains: A→B→C→...→Z, A ⊢ Z
  9. Constraint independence: A→B and A→C simultaneously (TS-specific)
  10. Constraint composition: (A∧B)→C

Output: binary shards of uint16 tokens in the same format as FineWeb/open-web-math.

Usage:
    python generate_logic_data.py \
        --tokenizer data/fineweb-10B/tokenizer.json \
        --out_dir data/logic-stage1 \
        --target_tokens 200_000_000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


# ── Grounded proposition banks ────────────────────────────────────────────────

# Each entry: (positive form, negative form)
PROPOSITIONS = [
    ("it is raining", "it is not raining"),
    ("the ground is wet", "the ground is not wet"),
    ("the path is slippery", "the path is not slippery"),
    ("the door is open", "the door is closed"),
    ("the light is on", "the light is off"),
    ("the engine is running", "the engine is not running"),
    ("the switch is active", "the switch is inactive"),
    ("the system is stable", "the system is unstable"),
    ("the value is positive", "the value is not positive"),
    ("the constraint is satisfied", "the constraint is not satisfied"),
    ("the node is connected", "the node is disconnected"),
    ("the process is complete", "the process is incomplete"),
    ("the signal is strong", "the signal is weak"),
    ("the temperature is high", "the temperature is low"),
    ("the pressure is above threshold", "the pressure is below threshold"),
    ("the lock is engaged", "the lock is released"),
    ("the circuit is closed", "the circuit is open"),
    ("the variable is defined", "the variable is undefined"),
    ("the condition holds", "the condition does not hold"),
    ("the relationship is valid", "the relationship is invalid"),
    ("the path exists", "no path exists"),
    ("the set is non-empty", "the set is empty"),
    ("the function converges", "the function diverges"),
    ("the proof is complete", "the proof is incomplete"),
    ("the assumption leads to a contradiction", "the assumption is consistent"),
]

ABSTRACT = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "A", "B", "C", "D", "E", "F", "G", "H"]


# ── Template generators ───────────────────────────────────────────────────────

def modus_ponens(grounded: bool = True) -> str:
    """A→B, A ⊢ B"""
    if grounded:
        a, b = random.sample(PROPOSITIONS, 2)
        return (
            f"Given that if {a[0]} then {b[0]}. "
            f"We know that {a[0]}. "
            f"Therefore {b[0]}."
        )
    else:
        a, b = random.sample(ABSTRACT, 2)
        return (
            f"If {a} then {b}. "
            f"{a} is true. "
            f"Therefore {b} is true."
        )


def modus_tollens(grounded: bool = True) -> str:
    """A→B, ¬B ⊢ ¬A"""
    if grounded:
        a, b = random.sample(PROPOSITIONS, 2)
        return (
            f"Given that if {a[0]} then {b[0]}. "
            f"We know that {b[1]}. "
            f"Therefore {a[1]}."
        )
    else:
        a, b = random.sample(ABSTRACT, 2)
        return (
            f"If {a} then {b}. "
            f"{b} is false. "
            f"Therefore {a} is false."
        )


def transitivity_chain(length: int = 3, grounded: bool = True) -> str:
    """A→B→C→...→Z ⊢ A→Z"""
    length = max(2, min(length, 6))
    if grounded:
        props = random.sample(PROPOSITIONS, length + 1)
        rules = []
        for i in range(length):
            rules.append(f"if {props[i][0]} then {props[i+1][0]}")
        conclusion = f"if {props[0][0]} then {props[-1][0]}"
        rule_str = ". ".join(f"Rule {i+1}: {r}" for i, r in enumerate(rules))
        return f"{rule_str}. Therefore {conclusion}."
    else:
        nodes = random.sample(ABSTRACT, length + 1)
        rules = [f"if {nodes[i]} then {nodes[i+1]}" for i in range(length)]
        rule_str = ". ".join(rules)
        return f"{rule_str}. Therefore if {nodes[0]} then {nodes[-1]}."


def chain_with_assertion(length: int = 3, grounded: bool = True) -> str:
    """A→B→C, A is true ⊢ B is true, C is true"""
    length = max(2, min(length, 5))
    if grounded:
        props = random.sample(PROPOSITIONS, length + 1)
        rules = [f"if {props[i][0]} then {props[i+1][0]}" for i in range(length)]
        steps = [f"{props[i+1][0]}" for i in range(length)]
        rule_str = ". ".join(rules)
        step_str = ". Therefore ".join(steps)
        return (
            f"{rule_str}. "
            f"We know that {props[0][0]}. "
            f"Therefore {step_str}."
        )
    else:
        nodes = random.sample(ABSTRACT, length + 1)
        rules = [f"if {nodes[i]} then {nodes[i+1]}" for i in range(length)]
        rule_str = ". ".join(rules)
        steps = " and ".join(f"{nodes[i+1]} is true" for i in range(length))
        return (
            f"{rule_str}. "
            f"{nodes[0]} is true. "
            f"Therefore {steps}."
        )


def proof_by_contradiction(grounded: bool = True) -> str:
    """Assume ¬G, derive ⊥ ⊢ G"""
    if grounded:
        goal, intermediate, contra = random.sample(PROPOSITIONS, 3)
        return (
            f"To prove that {goal[0]}, assume the opposite: {goal[1]}. "
            f"From this assumption, it follows that {intermediate[0]}. "
            f"But we know that {intermediate[1]}. "
            f"This is a contradiction. "
            f"Therefore our assumption was wrong. "
            f"Therefore {goal[0]}."
        )
    else:
        g, a, b = random.sample(ABSTRACT, 3)
        return (
            f"To prove {g}, assume {g} is false. "
            f"Then {a} must be true. "
            f"But {a} being true implies {b} is true. "
            f"And {b} being true implies {g} is true. "
            f"This contradicts our assumption. "
            f"Therefore {g} is true."
        )


def conjunction(grounded: bool = True) -> str:
    """A, B ⊢ A∧B — and both constrain C"""
    if grounded:
        a, b, c = random.sample(PROPOSITIONS, 3)
        return (
            f"We know that {a[0]}. "
            f"We also know that {b[0]}. "
            f"Since both {a[0]} and {b[0]}, it follows that {c[0]}."
        )
    else:
        a, b, c = random.sample(ABSTRACT, 3)
        return (
            f"{a} is true. "
            f"{b} is true. "
            f"If {a} and {b} then {c}. "
            f"Therefore {c} is true."
        )


def disjunction(grounded: bool = True) -> str:
    """A∨B, ¬A ⊢ B"""
    if grounded:
        a, b = random.sample(PROPOSITIONS, 2)
        return (
            f"Either {a[0]} or {b[0]}. "
            f"We know that {a[1]}. "
            f"Therefore {b[0]}."
        )
    else:
        a, b = random.sample(ABSTRACT, 2)
        return (
            f"Either {a} is true or {b} is true. "
            f"{a} is false. "
            f"Therefore {b} is true."
        )


def contradiction_detection(grounded: bool = True) -> str:
    """A, ¬A ⊢ contradiction"""
    if grounded:
        a = random.choice(PROPOSITIONS)
        return (
            f"Suppose {a[0]}. "
            f"But we also have {a[1]}. "
            f"These two statements cannot both be true. "
            f"This is a contradiction."
        )
    else:
        a = random.choice(ABSTRACT)
        return (
            f"Suppose {a} is true. "
            f"But {a} is also false. "
            f"A statement cannot be both true and false. "
            f"This is a contradiction."
        )


def ts_independence(grounded: bool = True) -> str:
    """TS-specific: A constrains B AND C simultaneously — no competition"""
    if grounded:
        a, b, c = random.sample(PROPOSITIONS, 3)
        return (
            f"We know that {a[0]}. "
            f"If {a[0]} then {b[0]}. "
            f"If {a[0]} then {c[0]}. "
            f"Therefore {b[0]}. "
            f"Also {c[0]}. "
            f"Both conclusions hold simultaneously."
        )
    else:
        a, b, c = random.sample(ABSTRACT, 3)
        return (
            f"{a} is true. "
            f"If {a} then {b}. "
            f"If {a} then {c}. "
            f"Therefore {b} is true and {c} is true. "
            f"Both constraints are satisfied simultaneously."
        )


def constraint_propagation(grounded: bool = True) -> str:
    """Multi-constraint: A→B, A→C, B∧C→D"""
    if grounded:
        a, b, c, d = random.sample(PROPOSITIONS, 4)
        return (
            f"If {a[0]} then {b[0]}. "
            f"If {a[0]} then {c[0]}. "
            f"If both {b[0]} and {c[0]} then {d[0]}. "
            f"We know that {a[0]}. "
            f"Therefore {b[0]}. "
            f"Therefore {c[0]}. "
            f"Therefore {d[0]}."
        )
    else:
        a, b, c, d = random.sample(ABSTRACT, 4)
        return (
            f"If {a} then {b}. "
            f"If {a} then {c}. "
            f"If {b} and {c} then {d}. "
            f"{a} is true. "
            f"Therefore {b} and {c} are both true. "
            f"Therefore {d} is true."
        )


def biconditional(grounded: bool = True) -> str:
    """A↔B: A→B and B→A"""
    if grounded:
        a, b = random.sample(PROPOSITIONS, 2)
        return (
            f"If {a[0]} then {b[0]}. "
            f"If {b[0]} then {a[0]}. "
            f"Therefore {a[0]} if and only if {b[0]}. "
            f"We know that {a[0]}. "
            f"Therefore {b[0]}."
        )
    else:
        a, b = random.sample(ABSTRACT, 2)
        return (
            f"If {a} then {b}. "
            f"If {b} then {a}. "
            f"{a} is true if and only if {b} is true. "
            f"{a} is true. "
            f"Therefore {b} is true."
        )


def hypothetical_syllogism_extended(grounded: bool = True) -> str:
    """Complex multi-branch reasoning"""
    if grounded:
        props = random.sample(PROPOSITIONS, 5)
        a, b, c, d, e = props
        return (
            f"If {a[0]} then {b[0]}. "
            f"If {b[0]} then {c[0]}. "
            f"If {a[0]} then {d[0]}. "
            f"If {c[0]} and {d[0]} then {e[0]}. "
            f"We know that {a[0]}. "
            f"From this: {b[0]}. "
            f"From this: {c[0]}. "
            f"Also from {a[0]}: {d[0]}. "
            f"Since both {c[0]} and {d[0]}: {e[0]}."
        )
    else:
        nodes = random.sample(ABSTRACT, 5)
        a, b, c, d, e = nodes
        return (
            f"If {a} then {b}. "
            f"If {b} then {c}. "
            f"If {a} then {d}. "
            f"If {c} and {d} then {e}. "
            f"{a} is true. "
            f"Therefore {b}, {c}, {d}, and {e} are all true."
        )


# ── Generator registry ────────────────────────────────────────────────────────

GENERATORS = [
    (modus_ponens,                    3.0),   # most fundamental
    (modus_tollens,                   2.0),
    (lambda: transitivity_chain(2),   1.5),
    (lambda: transitivity_chain(3),   2.0),
    (lambda: transitivity_chain(4),   1.5),
    (lambda: transitivity_chain(5),   1.0),
    (lambda: chain_with_assertion(2), 1.5),
    (lambda: chain_with_assertion(3), 2.0),
    (lambda: chain_with_assertion(4), 1.5),
    (proof_by_contradiction,          2.5),   # core for maths
    (conjunction,                     1.5),
    (disjunction,                     1.5),
    (contradiction_detection,         2.0),
    (ts_independence,                 2.5),   # TS-specific
    (constraint_propagation,          2.0),
    (biconditional,                   1.0),
    (hypothetical_syllogism_extended, 1.5),
]

# Abstract-only versions
ABSTRACT_GENERATORS = [
    (lambda: modus_ponens(False),                    2.0),
    (lambda: modus_tollens(False),                   2.0),
    (lambda: transitivity_chain(3, False),           2.0),
    (lambda: transitivity_chain(5, False),           1.5),
    (lambda: chain_with_assertion(3, False),         2.0),
    (lambda: proof_by_contradiction(False),          2.0),
    (lambda: ts_independence(False),                 2.0),
    (lambda: constraint_propagation(False),          2.0),
    (lambda: contradiction_detection(False),         1.5),
    (lambda: hypothetical_syllogism_extended(False), 1.5),
]

ALL_GENERATORS = GENERATORS + ABSTRACT_GENERATORS
WEIGHTS = [w for _, w in ALL_GENERATORS]
FNS = [f for f, _ in ALL_GENERATORS]


def sample_example() -> str:
    fn = random.choices(FNS, weights=WEIGHTS, k=1)[0]
    return fn()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate TS stage-1 logical constraint dataset")
    p.add_argument("--tokenizer",     default="data/tokenizer.json")
    p.add_argument("--out_dir",       default="data/logic-stage1")
    p.add_argument("--target_tokens", default=200_000_000, type=int)
    p.add_argument("--shard_tokens",  default=10_000_000,  type=int)
    p.add_argument("--val_shards",    default=1,            type=int)
    p.add_argument("--seed",          default=42,           type=int)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    sep = tokenizer.encode("\n\n").ids   # separator between examples

    total_target = args.target_tokens
    shard_size   = args.shard_tokens
    n_shards     = (total_target + shard_size - 1) // shard_size

    print(f"Generating {total_target/1e6:.0f}M tokens of logical constraint data")
    print(f"Shard size: {shard_size/1e6:.0f}M tokens | Total shards: {n_shards}")
    print(f"Output: {args.out_dir}")
    print()

    all_shards = []
    total_tokens = 0
    shard_idx = 0

    buf = []
    examples_generated = 0

    while total_tokens < total_target:
        example = sample_example()
        ids = tokenizer.encode(example).ids + sep
        buf.extend(ids)
        examples_generated += 1

        while len(buf) >= shard_size:
            chunk = buf[:shard_size]
            buf   = buf[shard_size:]

            # Last val_shards shards go to val split
            remaining_shards = n_shards - shard_idx
            split = "val" if remaining_shards <= args.val_shards else "train"

            fname = f"{split}_{shard_idx:04d}.bin"
            path  = os.path.join(args.out_dir, fname)
            np.array(chunk, dtype=np.uint16).tofile(path)
            all_shards.append({"split": split, "path": path, "tokens": shard_size})
            total_tokens += shard_size
            shard_idx    += 1

            print(f"  [{split}] Shard {shard_idx-1}: {shard_size/1e6:.0f}M tokens "
                  f"({examples_generated:,} examples, {total_tokens/1e6:.0f}M total)")

            if total_tokens >= total_target:
                break

    # Flush remainder as val if any
    if buf and total_tokens < total_target:
        fname = f"val_{shard_idx:04d}.bin"
        path  = os.path.join(args.out_dir, fname)
        np.array(buf, dtype=np.uint16).tofile(path)
        all_shards.append({"split": "val", "path": path, "tokens": len(buf)})
        total_tokens += len(buf)
        print(f"  [val] Shard {shard_idx}: {len(buf)/1e6:.1f}M tokens (remainder)")

    # Write metadata.json
    train_shards = [s for s in all_shards if s["split"] == "train"]
    val_shards_  = [s for s in all_shards if s["split"] == "val"]
    meta = {
        "dataset":      "logic-stage1",
        "description":  "Synthetic pure logical inference dataset — TS stage 1 curriculum",
        "vocab_size":   tokenizer.get_vocab_size(),
        "tokenizer":    args.tokenizer,
        "total_tokens": total_tokens,
        "train_tokens": sum(s["tokens"] for s in train_shards),
        "val_tokens":   sum(s["tokens"] for s in val_shards_),
        "shards":       all_shards,
        "generator_types": [
            "modus_ponens", "modus_tollens", "transitivity_chain",
            "chain_with_assertion", "proof_by_contradiction",
            "conjunction", "disjunction", "contradiction_detection",
            "ts_independence", "constraint_propagation",
            "biconditional", "hypothetical_syllogism_extended",
        ]
    }
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Total tokens:    {total_tokens/1e6:.1f}M")
    print(f"  Total examples:  {examples_generated:,}")
    print(f"  Train shards:    {len(train_shards)}")
    print(f"  Val shards:      {len(val_shards_)}")
    print(f"  Metadata:        {meta_path}")
    print(f"\nSample examples:")
    random.seed(0)
    for _ in range(5):
        print(f"  → {sample_example()}")


if __name__ == "__main__":
    main()
