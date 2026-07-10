"""ms-swift integration for gemma-triton-flash-attn (use via --custom_register_path).

Routes ALL attention (Gemma-4's 40 sliding D=256 layers AND 8 global D=512
layers) through this repo's Triton flash kernel while swift stays on
`--attn_impl sdpa`. Anything the kernel doesn't support (dropout>0, softcap,
decode-length N) falls back to stock SDPA transparently.

Why not `pip install` + plain import: this repo's package dir is named
`flash_attn`, which collides with the flash-attn wheel that ms-swift imports
first — a plain `import flash_attn` then resolves to the wheel and this repo is
shadowed. We therefore load the package under a private module name with
importlib, anchored to this file's location (works from any clone path).

Usage:
    swift rlhf ... \
        --attn_impl sdpa \
        --custom_register_path /path/to/repo/integrations/ms_swift/triton_attn_patch.py

Padding note: the kernel builds its own causal/sliding mask and ignores HF's
attention_mask (we pass None). This is safe for right-padded training batches:
padding sits after real tokens, so causal masking already hides it from real
queries, and pad positions carry no loss. Do NOT use with left padding.
"""
import importlib.util
import pathlib
import sys

_REPO = str(pathlib.Path(__file__).resolve().parents[2])

# Load the repo package under a private name to dodge the flash-attn wheel.
_spec = importlib.util.spec_from_file_location(
    '_gtfa', f'{_REPO}/flash_attn/__init__.py',
    submodule_search_locations=[f'{_REPO}/flash_attn'])
_gtfa = importlib.util.module_from_spec(_spec)
sys.modules['_gtfa'] = _gtfa
_spec.loader.exec_module(_gtfa)
triton_gqa_attention = _gtfa.triton_gqa_attention

from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

_HITS = [0]


def _triton_sdpa(module, query, key, value, attention_mask, **kwargs):
    # kernel wants tl.dot dims >=16; generation/decode short-N goes to SDPA.
    if query.shape[-2] >= 64 and kwargs.get('dropout', 0.0) == 0.0 \
            and kwargs.get('softcap') is None:
        try:
            out = triton_gqa_attention(module, query, key, value, None, **kwargs)
            _HITS[0] += 1
            if _HITS[0] == 1:
                print(f'[triton_attn_patch] ENGAGED: D={query.shape[-1]} '
                      f'N={query.shape[-2]} q{tuple(query.shape)} '
                      f'slide={kwargs.get("sliding_window")}', flush=True)
            return out
        except Exception as e:
            if _HITS[0] == 0:
                print(f'[triton_attn_patch] fallback->SDPA: {str(e)[:100]}', flush=True)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


ALL_ATTENTION_FUNCTIONS['sdpa'] = _triton_sdpa
print('[triton_attn_patch] sdpa overridden -> gemma-triton-flash-attn '
      '(both sliding D=256 and global D=512 layers)', flush=True)
