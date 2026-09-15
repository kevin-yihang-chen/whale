"""Read actual E4 sampling RNG state without consuming random numbers."""
class TrialRandomizationWorkerExtension:
    def trial_randomization_state(self):
        import hashlib
        import os
        import torch
        from .experiment_randomization import HASH_PROBE
        digest = lambda state: hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
        return {'pid': os.getpid(), 'model_seed': self.model_config.seed,
                'torch_cpu_initial_seed': torch.initial_seed(),
                'torch_cuda_initial_seed': torch.cuda.initial_seed(),
                'torch_cpu_rng_sha256': digest(torch.get_rng_state()),
                'torch_cuda_rng_sha256': digest(torch.cuda.get_rng_state()),
                'python_hash_probe': hash(HASH_PROBE),
                'device_name': torch.cuda.get_device_name()}

    def sample_updated_embeddings(self, coordinates):
        from .vllm_weight_probe_worker import WeightProbeWorkerExtension
        return WeightProbeWorkerExtension.sample_updated_embeddings(self, coordinates)
