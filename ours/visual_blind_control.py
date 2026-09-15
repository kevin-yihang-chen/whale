"""Development-only image ablation; preserve questions and grading outside requests."""
from .visual_harness_dataset import VisualHarnessDataset


class BlindVisualHarnessDataset(VisualHarnessDataset):
    def _build_messages(self, example):
        messages = super()._build_messages(example)
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                message['content'] = [part for part in content if part.get('type') == 'text']
        return messages
