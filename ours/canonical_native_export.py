"""Retain the E4 active graph and fixed base inference contract during export.

WHALE's original FSDP merger still loads, merges and casts every tensor. Only
the model's serialization option is bound to save_original_format=False, avoiding
Transformers' recursive legacy visual-key renaming. A reviewed base config is
used so native training's top-level EOS/pad overrides do not change inference.
"""
from functools import partial
import json
from pathlib import Path

from .audit_native_training_batch import require


def validate_inference_config(base, native):
    """Permit only reviewed training precision and tokenizer-token overrides."""
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    a, b = [AutoConfig.from_pretrained(path, local_files_only=True).to_dict() for path in (base, native)]
    differences = []
    def walk(x, y, path=''):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in x.keys() | y.keys():
                walk(x.get(key), y.get(key), f'{path}.{key}'.lstrip('.'))
        elif x != y:
            differences.append({'path': path, 'base': x, 'native': y})
    walk(a, b)
    allowed = {'_name_or_path', 'dtype', 'text_config.dtype', 'vision_config.dtype', 'eos_token_id', 'pad_token_id'}
    for difference in differences:
        path, before, after = (difference[key] for key in ('path', 'base', 'native'))
        require(path in allowed, f'Unreviewed model configuration difference: {path}')
        if path == 'eos_token_id':
            require(before is None and after == tokenizer.eos_token_id, 'Unexpected native EOS override')
        elif path == 'pad_token_id':
            require(before is None and after == tokenizer.pad_token_id, 'Unexpected native pad override')
        elif path.endswith('dtype'):
            require(before in (None, 'bfloat16') and after == 'float32', 'Unexpected precision override')
    return differences


def canonical_merger_class():
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    class CanonicalFSDPModelMerger(FSDPModelMerger):
        def get_transformers_auto_model_class(self):
            original_factory = super().get_transformers_auto_model_class()
            class ActiveGraphSerializationFactory:
                @staticmethod
                def from_config(*args, **kwargs):
                    model = original_factory.from_config(*args, **kwargs)
                    model.save_pretrained = partial(model.save_pretrained, save_original_format=False)
                    return model
            return ActiveGraphSerializationFactory
    return CanonicalFSDPModelMerger


if __name__ == '__main__':
    import argparse
    from .training_bootstrap import prepare_worker
    prepare_worker()
    from verl.model_merger.base_model_merger import ModelMergerConfig
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    require(not args.target.exists(), 'Export destination already exists')
    differences = validate_inference_config(args.base, args.native / 'huggingface')
    merger = canonical_merger_class()(ModelMergerConfig(operation='merge', backend='fsdp',
        local_dir=str(args.native), target_dir=str(args.target), hf_model_config_path=str(args.base)))
    merger.merge_and_save()
    merger.cleanup()
    report = {'kind': 'canonical_native_export', 'base': str(args.base), 'native': str(args.native),
              'exported': str(args.target), 'save_original_format': False,
              'inference_config_source': 'base checkpoint', 'reviewed_native_config_differences': differences,
              'native_merge_and_bf16_cast_unchanged': True, 'new_model_calls': 0,
              'limitations': ['Parameter values, complete exported graph and live inference require separate checks.']}
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report), flush=True)
