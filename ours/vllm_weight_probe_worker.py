"""E4 worker extension for read-only, numeric parameter handoff measurements."""


class WeightProbeWorkerExtension:
    def sample_updated_embeddings(self, coordinates):
        # vLLM owns model loading; coordinates carry no expected result values.
        from .probe_updated_vllm import worker_embedding_samples
        return worker_embedding_samples(self.model_runner.get_model(), coordinates)
