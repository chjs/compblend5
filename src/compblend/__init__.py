"""compblend — compressed KV cache blending on top of cacheblend-hf-v7.

Public API is intentionally empty at M0. Modules are added per milestone:
    M1: backends.kvzip            (pre-RoPE capture)
    M2: fuse_selective_compblend  (v7 fusor fork + per-head mask + Gated HKVD)
    M3: rag.pipeline              (KVStore-based online blend)
"""
__version__ = "0.5.0.dev0"
