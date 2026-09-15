"""Bind the released E4 fused linear operator's token chunk to a frozen budget.

The original autograd forward/backward still compute every vocabulary entry and
every selected loss token. This does not change the SFT minibatch or its mask.
"""
from .audit_native_training_batch import require


def install_chunk_size(chunk_size):
    import verl.utils.experimental.torch_functional as native
    require(type(chunk_size) is int and chunk_size == 128, 'Unreviewed fused token chunk size')
    original = native.FusedLinearForPPO
    existing = getattr(original, '_whale_fused_chunk_size', None)
    if existing is not None:
        require(existing == chunk_size, 'Another fused chunk contract is already installed')
        return

    class BoundedFusedLinearForPPO(original):
        _whale_fused_chunk_size = chunk_size

        def __init__(self, chunk_size=chunk_size):
            require(chunk_size == self._whale_fused_chunk_size, 'Fused chunk override differs from frozen contract')
            super().__init__(chunk_size=chunk_size)

    native.FusedLinearForPPO = BoundedFusedLinearForPPO
