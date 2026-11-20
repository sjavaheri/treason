"""Torch-free entry points for the math-verification subprocess pool.

This module deliberately lives at the repo root, OUTSIDE the ``verl`` package,
and imports nothing but ``math_verify``.

Reason: the reward pool uses the 'spawn' start method, so each worker must
import the module that defines its initializer and its task function. Anything
under ``verl.*`` drags in ``verl/__init__.py`` -> ``verl.protocol`` -> ``torch``,
and importing torch reserves ~9 GB of *virtual address space* on this platform
(aarch64, 64 KB pages). That does not fit under the 2 GB RLIMIT_AS the workers
set to guard against sympy bombs, so every worker died with MemoryError and the
pool was respawned for essentially every sample (~4 s each).

Keeping these two functions torch-free costs ~0.1 GB of address space, which
leaves the 2 GB cap doing its intended job with plenty of headroom.

Do not add imports from ``verl`` (or torch) to this file.
"""

import resource

MAX_WORKER_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def limit_memory():
    """Limit the address space of each worker process to prevent sympy bombs."""
    resource.setrlimit(resource.RLIMIT_AS, (MAX_WORKER_MEMORY_BYTES, MAX_WORKER_MEMORY_BYTES))


def hf_verify_with_try(gold, target):
    from math_verify import parse, verify
    try:
        parsed_target = parse(target)
        parsed_gold = parse(gold)
        return verify(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False
