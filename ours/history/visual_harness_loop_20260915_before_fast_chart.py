"""Execute shared visual harness callbacks inside the original native loop.

The E3 candidate changes observation, crop arguments, feedback, parsing and
bounded continuation. E4 still grades its committed answer with the shared
verifier and uses native masks. VETO selection is not implemented by this file.
"""
import json
import hashlib

from .visual_evidence_tool_loop import VisualEvidenceToolAgentLoop
from .visual_harness import configured_visual_harness
from .visual_harness_dataset import reject_legacy_prompt_environment
from .visual_evidence_reward import compute_score
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.experimental.agent_loop.tool_parser import FunctionCall


@register('visual_harness_agent')
class VisualHarnessAgentLoop(VisualEvidenceToolAgentLoop):
    def __init__(self, *args, **kwargs):
        reject_legacy_prompt_environment()
        super().__init__(*args, **kwargs)
        self.visual_harness = configured_visual_harness(self.data_config)

    async def run(self, sampling_params, **kwargs):
        self.visual_harness.unchanged()
        if kwargs.get('visual_harness_sha256') != self.visual_harness.sha256:
            raise ValueError('Dataset and rollout visual harness identities differ')
        output = await super().run(sampling_params, **kwargs)
        answer = self.visual_harness.invoke('parse_answer', text=output.extra_fields['visual_final_answer'])
        if not isinstance(answer, str):
            raise ValueError('Visual answer parser must return text')
        scores = compute_score(kwargs['data_source'], answer, kwargs['reward_model']['ground_truth'])
        output.reward_score = scores['score']
        output.extra_fields.update(visual_committed_answer=answer, visual_harness_sha256=self.visual_harness.sha256,
            visual_policy_calls=output.extra_fields.get('visual_policy_calls', 0),
            visual_generated_tokens=output.extra_fields.get('visual_generated_tokens', 0),
            visual_harness_nudges=output.extra_fields.get('visual_harness_nudges', 0),
            visual_harness_tool_trace=output.extra_fields.get('visual_harness_tool_trace', []), reward_extra_info=scores)
        return output

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        requested_arguments = tool_call.arguments
        images = agent_data.image_data or []
        image_size = images[0].size if images else None
        if tool_call.name == 'image_zoom_in_tool':
            arguments = json.loads(tool_call.arguments)
            if isinstance(arguments, dict):
                arguments = self.visual_harness.invoke('prepare_tool', arguments=arguments,
                    image_size=image_size)
                if not isinstance(arguments, dict):
                    raise ValueError('Visual tool preparation must return an argument object')
                tool_call = FunctionCall(name=tool_call.name, arguments=json.dumps(arguments, allow_nan=False))
        response, reward, details = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        text = self.visual_harness.invoke('format_feedback', text=response.text or '')
        if not isinstance(text, str):
            raise ValueError('Visual tool feedback must return text')
        if text != (response.text or ''):
            limit = self.max_tool_response_length
            if len(text) > limit:
                if self.tool_response_truncate_side == 'left':
                    text = text[:limit] + '...(truncated)'
                elif self.tool_response_truncate_side == 'right':
                    text = '(truncated)...' + text[-limit:]
                else:
                    text = text[:limit // 2] + '...(truncated)...' + text[-limit // 2:]
            response = response.model_copy(update={'text': text})
        trace = getattr(agent_data, '_visual_harness_tool_trace', [])
        trace.append({'name': tool_call.name, 'requested_arguments': requested_arguments,
            'executed_arguments': tool_call.arguments, 'initial_image_size': image_size,
            'feedback_sha256': hashlib.sha256((response.text or '').encode()).hexdigest(),
            'returned_images': [{'mode': image.mode, 'size': image.size,
                'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest()} for image in (response.image or [])]})
        agent_data._visual_harness_tool_trace = trace
        agent_data.extra_fields['visual_harness_tool_trace'] = trace
        return response, reward, details

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        agent_data.extra_fields['visual_policy_calls'] = agent_data.assistant_turns
        agent_data.extra_fields['visual_generated_tokens'] = sum(agent_data.response_mask)
        if state != AgentState.TERMINATED or agent_data.tool_calls:
            return state
        if ((self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns) or
                (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)):
            return state
        remaining = self._remaining_assistant_tokens(agent_data)
        if remaining is not None and remaining <= 0:
            return state
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        nudge = self.visual_harness.invoke('nudge', text=text, assistant_turns=agent_data.assistant_turns)
        if nudge is None:
            return state
        if not isinstance(nudge, str) or not nudge.strip():
            raise ValueError('Visual continuation must be nonempty text or None')
        messages = [{'role': 'user', 'content': nudge}]
        ids = await self.apply_chat_template(messages, images=None, videos=None, remove_system_prompt=True)
        if len(agent_data.response_mask) + len(ids) >= self.response_length:
            return state
        agent_data.messages.extend(messages)
        agent_data.prompt_ids += ids
        agent_data.response_mask += [0] * len(ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.] * len(ids)
        agent_data.user_turns += 1
        agent_data.extra_fields['visual_harness_nudges'] = agent_data.extra_fields.get('visual_harness_nudges', 0) + 1
        return AgentState.GENERATING
