"""Bind executable visual observation formatting to native length filtering.

Shared E3/E4 input channel. Harness code identity enters the native dataset
fingerprint and every row, before generation or any task reward is available.
"""
import os

from .visual_evidence_dataset import VisualEvidenceDataset, PROMPT_VARIABLES
from .visual_harness import configured_visual_harness


def reject_legacy_prompt_environment():
    if any(os.environ.get(name) for name in ('HARNESS_PATH', *PROMPT_VARIABLES)):
        raise ValueError('Use the explicit visual harness config without legacy Chess prompt hooks')


class VisualHarnessDataset(VisualEvidenceDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, **kwargs):
        reject_legacy_prompt_environment()
        self.visual_harness = configured_visual_harness(config)
        super().__init__(data_files, tokenizer, config, processor, **kwargs)

    def _build_messages(self, example):
        self.visual_harness.unchanged()
        messages = super()._build_messages(example)
        user = next(message for message in messages if message['role'] == 'user')
        parts = user['content']
        question = parts if isinstance(parts, str) else ''.join(part['text'] for part in parts if part['type'] == 'text')
        text = self.visual_harness.invoke('format_observation', question=question)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Visual observation formatter must return nonempty text')
        user['content'] = text if isinstance(parts, str) else [
            *[part for part in parts if part['type'] != 'text'], {'type': 'text', 'text': text}]
        systems = [message for message in messages if message['role'] == 'system']
        if len(systems) > 1:
            raise ValueError('Visual harness requires at most one initial system message')
        if systems:
            systems[0]['content'] = self.visual_harness.system_prompt
        else:
            messages.insert(0, {'role': 'system', 'content': self.visual_harness.system_prompt})
        return messages

    def __getitem__(self, item):
        row = super().__getitem__(item)
        row['visual_harness_sha256'] = self.visual_harness.sha256
        return row
