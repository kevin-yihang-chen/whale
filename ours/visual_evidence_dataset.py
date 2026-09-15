"""Preserve one-image evidence through native dataset and prompt hooks.

This shared environment adapter is outside VETO E1--E3. It keeps the original
prompt override, filtering and row bookkeeping, while retaining image evidence
and binding the prompt environment used by dataset-filter caches.
"""
from copy import deepcopy
from io import BytesIO
import os

from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset

PROMPT_VARIABLES=('_HARNESS_SYSTEM_PROMPT','_HARNESS_USER_PROMPT_TEMPLATE')


class VisualEvidenceDataset(RLHFDataset):
    """A shared single-image input contract for the controlled visual domain.

    Rows contain one system/user exchange and at most one bytes-only image.
    No-image controls use the same class with an empty image list. A new harness
    requires a new dataset instance; the captured prompts also enter the native
    Hugging Face filter fingerprint through the bound instance.
    """
    def __init__(self,*args,**kwargs):
        self.visual_prompt_environment=tuple(os.environ.get(key) for key in PROMPT_VARIABLES)
        super().__init__(*args,**kwargs)

    def _build_messages(self,example):
        if tuple(os.environ.get(key) for key in PROMPT_VARIABLES)!=self.visual_prompt_environment:
            raise ValueError('Visual prompt environment changed after dataset construction')
        source=deepcopy(example);images=source.get(self.image_key) or []
        if len(images)>1 or source.get(self.video_key):
            raise ValueError('This visual domain supports at most one image and no video')
        messages=source[self.prompt_key]
        users=[message for message in messages if message.get('role')=='user']
        if len(users)!=1 or any(m.get('role') not in ('system','user') for m in messages):
            raise ValueError('Expected one visual user question before generation')
        if not isinstance(users[0]['content'],str) or users[0]['content'].count('<image>')!=len(images):
            raise ValueError('Image payload and source prompt placeholder differ')
        evidence=None
        if images:
            payload=images[0]
            if not isinstance(payload,dict) or set(payload)!={'bytes'} or not isinstance(payload['bytes'],bytes):
                raise ValueError('Visual evidence requires a bytes-only image payload')
            with Image.open(BytesIO(payload['bytes'])) as image:
                evidence=image.convert('RGB')
        # Copying matters: the native length filter still needs example[image_key]
        # after this method. The upstream builder pops that field from its input.
        result=super()._build_messages(source)
        user=next(message for message in result if message['role']=='user')
        if evidence is not None and isinstance(user['content'],str):
            user['content']=[{'type':'image','image':evidence},{'type':'text','text':user['content']}]
        visible=[part for message in result if isinstance(message['content'],list)
            for part in message['content'] if part.get('type')=='image']
        if len(visible)!=len(images):raise ValueError('Native prompt override changed visual evidence coverage')
        return result

    def __getitem__(self,item):
        row=super().__getitem__(item)
        # Raw pixels already live in raw_prompt. Match the original row schema
        # instead of duplicating encoded payloads in rollout non-tensor metadata.
        row.pop(self.image_key,None);row.pop(self.video_key,None)
        return row
