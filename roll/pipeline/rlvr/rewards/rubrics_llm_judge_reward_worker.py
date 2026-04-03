from typing import Optional, Union, Dict, List, Any
import json
import re
from venv import logger
import torch
import requests
import time
import traceback
import numpy as np
from functools import partial
import tensordict
from tensordict import TensorDict
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_reward_model_provider
from roll.platforms import current_platform
from roll.utils.logging import get_logger
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *
from roll.datasets.chat_template import get_chat_template
from roll.pipeline.rlvr.rewards.ifeval_rule_reward_worker import (
    IF_FUNCTIONS_MAP,
    call_ifeval_function,
)
from roll.pipeline.rlvr.rewards.type2_checkers import TYPE2_CHECKERS
from roll.pipeline.rlvr.rewards.muldimif_checkers import CONSTRAINT_CHECKER_MAP


# Mapping from type1 (IFBench) instruction_id to (ifeval_function_name, param_remap_dict).
# The param_remap_dict maps data kwargs keys to the ifeval function parameter names.
# Keys not in remap are passed through as-is; call_ifeval_function auto-filters
# kwargs to only those matching the function signature.
INSTRUCTION_ID_TO_IFEVAL: Dict[str, tuple] = {
    # No-param functions
    "change_case:english_capital": ("validate_uppercase", {}),
    "change_case:english_lowercase": ("validate_lowercase", {}),
    "combination:two_responses": ("validate_two_responses", {}),
    "detectable_format:json_format": ("validate_json_format", {}),
    "detectable_format:title": ("validate_title", {}),
    "punctuation:no_comma": ("validate_no_commas", {}),
    "startend:quotation": ("validate_quotation", {}),
    # Direct param match
    "detectable_content:postscript": ("verify_postscript", {}),
    "keywords:forbidden_words": ("validate_forbidden_words", {}),
    "language:response_language": ("validate_response_language", {}),
    "startend:end_checker": ("validate_end", {}),
    # Param remapping needed
    "change_case:capital_word_frequency": (
        "validate_frequency_capital_words",
        {"capital_frequency": "N", "capital_relation": "quantifier", "relation": "quantifier"},
    ),
    "combination:repeat_prompt": (
        "validate_repeat_prompt",
        {"prompt_to_repeat": "original_prompt"},
    ),
    "detectable_content:number_placeholders": (
        "validate_placeholders",
        {"num_placeholders": "N"},
    ),
    "detectable_format:multiple_sections": (
        "validate_sections",
        {"section_spliter": "section_splitter", "num_sections": "N"},
    ),
    "detectable_format:number_bullet_lists": (
        "verify_bullet_points",
        {"num_bullets": "N"},
    ),
    "detectable_format:number_highlighted_sections": (
        "validate_highlighted_sections",
        {"num_highlights": "N"},
    ),
    "keywords:existence": (
        "verify_keywords",
        {"keywords": "keyword_list"},
    ),
    "keywords:frequency": (
        "verify_keyword_frequency",
        {"keyword": "word", "frequency": "N", "relation": "quantifier"},
    ),
    "keywords:letter_frequency": (
        "verify_letter_frequency",
        {"let_frequency": "N", "let_relation": "quantifier", "relation": "quantifier"},
    ),
    "length_constraints:nth_paragraph_first_word": (
        "validate_paragraphs",
        {"num_paragraphs": "N", "nth_paragraph": "i"},
    ),
    "length_constraints:number_paragraphs": (
        "verify_paragraph_count",
        {"num_paragraphs": "N"},
    ),
    "length_constraints:number_sentences": (
        "verify_sentence_constraint",
        {"num_sentences": "N", "relation": "quantifier"},
    ),
    "length_constraints:number_words": (
        "validate_word_constraint",
        {"num_words": "N", "relation": "quantifier"},
    ),
    # detectable_format:constrained_response excluded: options are embedded in
    # criteria text with empty kwargs, not parseable for code evaluation.
}


class RubricsLLMJudgeRewardWorker(Worker):
    """
    Reward Worker that uses LLM-as-judge to compute rewards.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # LLM judge相关配置
        self.judge_prompt = self.worker_config.judge_prompt if hasattr(self.worker_config, "judge_prompt") else None
        self.judge_prompt = prompt_maps.get(self.judge_prompt, None)
        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "api"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)
        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            print(f"{self.worker_name} initialized with API model")

        elif self.judge_model_type == "inference":
            async_strategy = self.worker_config.strategy_args.strategy_name in ["vllm", "sglang"]
            if self.worker_config.strategy_args.strategy_name == "sglang":  # not weight sync, need backup weights
                self.worker_config.strategy_args.strategy_config["enable_weights_cpu_backup"] = True
            if self.worker_config.strategy_args.strategy_name == "vllm":
                self.worker_config.strategy_args.strategy_config["sleep_level"] = 1
            self.strategy = create_strategy(worker=self, sync_wrapper=async_strategy)
            self.strategy.initialize(model_provider=default_reward_model_provider)
            self.tokenizer = self.strategy.tokenizer
            print(f"{self.worker_name} initialized with inference model")
            self.strategy.offload_states()
            current_platform.init()
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

    def _call_api_model(self, messages: Dict, retry_times=3) -> str:
        from openai import OpenAI

        ouput = ""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API model type")
        while retry_times > 0:
            retry_times -= 1
            try:
                client = OpenAI(
                    api_key=self.judge_api_key,
                    base_url=self.judge_api_url,
                )
                completion = client.chat.completions.create(model=self.judge_model_name, messages=messages)
                output = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens
                prompt_token = completion.usage.prompt_tokens
                completion_token = completion.usage.completion_tokens
                token_info = {
                    "total_tokens": total_tokens,
                    "prompt_token": prompt_token,
                    "completion_token": completion_token,
                }
                print(token_info)
                if output != None and output != "":
                    break
            except Exception as e:
                print(e)
                continue
        self.logger.info(f"judge model api output: {str(output)}")
        return output

    def _run_local_inference(self, messages: Dict) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")

        template_name = self.worker_config.data_args.template
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        text = chat_template_func(messages)

        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(current_platform.device_type)
        attention_mask = tokenized["attention_mask"].to(current_platform.device_type)

        max_model_len = self.worker_config.strategy_args.strategy_config.get("max_model_len", 8192)
        max_new_tokens = self.worker_config.generating_args.max_new_tokens or 512
        max_input_len = max_model_len - max_new_tokens
        if input_ids.shape[1] > max_input_len:
            self.logger.warning(
                f"Judge prompt ({input_ids.shape[1]} tokens) exceeds limit ({max_input_len}), truncating."
            )
            input_ids = input_ids[:, :max_input_len]
            attention_mask = attention_mask[:, :max_input_len]

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        data = DataProto(
            batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
        )
        data = data.to(current_platform.device_type)
        data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            if isinstance(output, torch.Tensor):
                generate_ids = output[:, len(input_ids[0]) :]
            else:
                generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]

        output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        self.logger.info(f"judge model inference output: {str(output)}")
        return output.strip()

    def _extract_score(self, response: str) -> float:
        try:
            match = re.search("Score: ([0-9.]+)", response)
            if match:
                score = float(match.group(1))
                normalized_score = score / 10
                return normalized_score
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0.5
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0.5

    def _extract_score_v2(self, response: str) -> float:
        response_lower = response.lower()
        try:
            # First try to match {YES} or {NO} format from the prompt
            if "{yes}" in response_lower:
                return 1
            elif "{no}" in response_lower:
                return -1
            # Fallback to plain yes/no if braced format not found
            elif "yes" in response_lower:
                return 1
            elif "no" in response_lower:
                return -1
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return -1
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return -1

    def _format_judge_prompt_v2(self, prompt: str, response: str, rubric: str = None) -> str:
        formatted_prompt = f"""Based on the provided Input (if any) and Generated Text, judge whether the generated text fulfills the Criteria Item with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:

- YES: Select ‘YES’ if the generated text entirely fulfills the condition specified in the Criteria Item. However, note that even minor inaccuracies exclude the text from receiving a ’YES’ rating. As an illustration, consider a Criteria Item ”Each sentence in the generated text uses a second person”. If even one sentence does not use the second person, the answer should NOT be ’YES’. To qualify for a ‘YES’ rating, the generated text must be entirely accurate and satisfy the criteria.

- NO: Opt for ‘NO’ if the generated text fails to meet the criteria or provides no information that could be utilized to judge. For instance, the Criteria Item asks ”Is the second sentence in the generated text a compound sentence?” and the generated text only has one sentence. It offers no relevant information to judge whether this criteria is met. Consequently, the answer should be ‘NO’.

Input:
{prompt}

Generated Text:
{response}

Criteria Item:
{rubric}

You only need to judge whether the generated text satisfiy the given Criteria Item and do NOT affect by other requirements in Input (if any). Return either a ‘YES’ or ‘NO’ choice without any additional text in your response."""
        messages = [{"role": "user", "content": formatted_prompt}]
        return messages, rubric

    def _format_judge_prompt(self, prompt: str, response: str, rubric: str = None) -> str:
        if "user\n" in prompt:
            prompt = prompt.split("user\n")[-1].strip()
        if not self.judge_prompt:
            formatted_prompt = f"""You are an expert judge evaluating the satisfication of a response to a given rubric.

Please evaluate whether the response meets the rubric.
Your evaluation should be "Yes" or "No".
If the response meets the rubric, the evaluation should be "Yes".
If the response does not meet the rubric, the evaluation should be "No".
Your output must follow the following format:
1) Provide an explanation for why the response satisfies the rubric or not.
2) Then provide your final answer in the form of: {{YES}} or {{NO}}

Prompt: {prompt}

Response: {response}

Rubric: {rubric}"""
        else:
            formatted_prompt = self.judge_prompt.format(question=prompt, response=response, rubric=rubric)
        messages = [{"role": "user", "content": formatted_prompt}]
        return messages, rubric

    def _get_llm_judgment(self, prompt: str, response: str, rubric: str = None) -> float:
        messages, rubric = self._format_judge_prompt_v2(prompt, response, rubric)
        if self.judge_model_type == "api":
            llm_response = self._call_api_model(messages)
        elif self.judge_model_type == "inference":
            llm_response = self._run_local_inference(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

        score = self._extract_score_v2(llm_response)
        info = {
            "score": score,
            "prompt": prompt,
            "response": response,
            "rubric": rubric,
            "messages": messages,
            "llm_response": llm_response,
        }
        return score, info

    def _remap_kwargs(self, kwargs: Dict[str, Any], remap: Dict[str, str]) -> Dict[str, Any]:
        """Remap data kwargs keys to match ifeval function parameter names."""
        remapped = {}
        for data_key, value in kwargs.items():
            if data_key in remap:
                remapped[remap[data_key]] = value
            else:
                remapped[data_key] = value
        return remapped

    def _evaluate_type1_rubric(
        self, response_text: str, instruction_id: str, kwargs: Dict[str, Any]
    ) -> Optional[float]:
        """Evaluate a type1 (IFBench) rubric using ifeval code functions.

        Returns score (1.0/-1.0) if code evaluation succeeds, None if
        the instruction_id is not supported and should fall back to LLM.
        """
        if instruction_id not in INSTRUCTION_ID_TO_IFEVAL:
            return None

        func_name, param_remap = INSTRUCTION_ID_TO_IFEVAL[instruction_id]
        if func_name not in IF_FUNCTIONS_MAP:
            self.logger.warning(f"Function {func_name} not found in IF_FUNCTIONS_MAP")
            return None

        func = IF_FUNCTIONS_MAP[func_name]
        remapped_kwargs = self._remap_kwargs(kwargs, param_remap)

        try:
            result = call_ifeval_function(func, response_text, remapped_kwargs)
            return 1.0 if result else -1.0
        except Exception as e:
            self.logger.error(f"Code eval error for {instruction_id}/{func_name}: {e}")
            return None

    def _evaluate_type4_rule(
        self, prompt_text: str, response_text: str, function_code: str
    ) -> Optional[float]:
        """Execute a type4 (VerIF) [rule] checker function.

        The function_code defines check_following(instruction, response) -> bool.
        Returns 1.0/-1.0 on success, None on failure (fall back to LLM).
        """
        try:
            exec_globals: Dict[str, Any] = {"re": re}
            exec(function_code, exec_globals)
            check_fn = exec_globals.get("check_following")
            if check_fn is None:
                self.logger.warning("check_following not found in type4 rule function code")
                return None
            result = check_fn(prompt_text, response_text)
            return 1.0 if result else -1.0
        except Exception as e:
            self.logger.error(f"Type4 rule exec error: {e}")
            return None

    def _evaluate_type2_rubric(
        self, response_text: str, instruction_id: str, kwargs: Dict[str, Any]
    ) -> Optional[float]:
        """Evaluate a type2 (Chat Arena) rubric using custom checker functions.

        Returns score (1.0/-1.0) if code evaluation succeeds, None if
        the instruction_id is not supported and should fall back to LLM.
        """
        checker_fn = TYPE2_CHECKERS.get(instruction_id)
        if checker_fn is None:
            return None

        try:
            result = checker_fn(response_text, kwargs)
            return 1.0 if result else -1.0
        except Exception as e:
            self.logger.error(f"Code eval error for type2 {instruction_id}: {e}")
            return None

    def _evaluate_type3_rubric(
        self, response_text: str, constraint: list
    ) -> Optional[float]:
        """Evaluate a type3 (MulDimIF) constraint using ported checker classes.

        constraint is a [Category, Subcategory, Description] triple.
        Returns score (1.0/-1.0) if code evaluation succeeds, None if
        the constraint type is not supported and should fall back to LLM.
        """
        if not isinstance(constraint, (list, tuple)) or len(constraint) < 3:
            return None

        category, subcategory, description = constraint[0], constraint[1], constraint[2]
        checker_key = f"{category}_{subcategory}"

        checker = CONSTRAINT_CHECKER_MAP.get(checker_key)
        if checker is None:
            return None

        try:
            result = checker.check(description, response_text)
            return 1.0 if result else -1.0
        except Exception as e:
            self.logger.error(f"Code eval error for type3 {checker_key}: {e}")
            return None

    def _evaluate_rubric(
        self,
        prompt_text: str,
        response_raw: str,
        response_stripped: str,
        rubric: str,
        source: str,
        ground_truth_parsed: Dict[str, Any],
        rubric_index: int,
    ) -> tuple:
        """Evaluate a single rubric, dispatching to code or LLM.

        Code evaluation uses response_stripped (after extract_after_last_think).
        LLM evaluation uses response_raw (preserving current behavior).
        Returns (score, info_dict).
        """
        if source == "type1":
            instruction_ids = ground_truth_parsed.get("instruction_id_list", [])
            kwargs_list = ground_truth_parsed.get("kwargs", [])
            if rubric_index < len(instruction_ids) and rubric_index < len(kwargs_list):
                instruction_id = instruction_ids[rubric_index]
                kwargs = kwargs_list[rubric_index]
                code_score = self._evaluate_type1_rubric(response_stripped, instruction_id, kwargs)
                if code_score is not None:
                    info = {
                        "score": code_score,
                        "method": "code_type1",
                        "instruction_id": instruction_id,
                        "rubric": rubric,
                    }
                    return code_score, info

        elif source == "type2":
            instruction_ids = ground_truth_parsed.get("instruction_id_list", [])
            kwargs_list = ground_truth_parsed.get("kwargs", [])
            if rubric_index < len(instruction_ids) and rubric_index < len(kwargs_list):
                instruction_id = instruction_ids[rubric_index]
                kwargs = kwargs_list[rubric_index]
                code_score = self._evaluate_type2_rubric(response_stripped, instruction_id, kwargs)
                if code_score is not None:
                    info = {
                        "score": code_score,
                        "method": "code_type2",
                        "instruction_id": instruction_id,
                        "rubric": rubric,
                    }
                    return code_score, info

        elif source == "type3":
            constraints = ground_truth_parsed.get("constraints", [])
            if rubric_index < len(constraints):
                constraint = constraints[rubric_index]
                code_score = self._evaluate_type3_rubric(response_stripped, constraint)
                if code_score is not None:
                    checker_key = f"{constraint[0]}_{constraint[1]}" if len(constraint) >= 2 else "unknown"
                    info = {
                        "score": code_score,
                        "method": "code_type3",
                        "constraint_type": checker_key,
                        "rubric": rubric,
                    }
                    return code_score, info

        elif source == "type4":
            checkers = ground_truth_parsed.get("checker", [])
            functions = ground_truth_parsed.get("functions", [])
            if rubric_index < len(checkers) and rubric_index < len(functions):
                checker = checkers[rubric_index]
                if checker.startswith("[rule]"):
                    function_code = functions[rubric_index]
                    code_score = self._evaluate_type4_rule(prompt_text, response_stripped, function_code)
                    if code_score is not None:
                        info = {
                            "score": code_score,
                            "method": "code_type4_rule",
                            "checker": checker,
                            "rubric": rubric,
                        }
                        return code_score, info

        # Fall back to LLM judge for: unsupported instruction_ids, type4 [llm] checkers,
        # Language_Chinese constraints, or any code evaluation failure.
        score, info = self._get_llm_judgment(prompt_text, response_raw, rubric)
        info["method"] = "llm_judge"
        return score, info

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}

        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        else:
            return self._compute_rewards_impl(data, metrics)

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict):
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        scores = []
        rubric_scores_list = []
        for prompt_txt, response, rubrics, source, ground_truth_raw in zip(
            data.non_tensor_batch["prompt"],
            response_text_list,
            data.non_tensor_batch["rubrics"],
            data.non_tensor_batch["source"],
            data.non_tensor_batch["ground_truth"],
        ):
            # Parse ground_truth from JSON string
            try:
                if isinstance(ground_truth_raw, str):
                    ground_truth_parsed = json.loads(ground_truth_raw)
                else:
                    ground_truth_parsed = ground_truth_raw
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.error(f"Failed to parse ground_truth: {e}")
                ground_truth_parsed = {}

            # Strip thinking tags for code-based evaluation
            response_stripped = response

            rubric_scores = [-100] * 20
            for r_idx, rubric in enumerate(rubrics[:20]):
                score, info = self._evaluate_rubric(
                    prompt_txt, response, response_stripped, rubric,
                    source, ground_truth_parsed, r_idx,
                )
                rubric_scores[r_idx] = score
                self.logger.info(f"{json.dumps(info, ensure_ascii=False)}")
            if self.pipeline_config.reward_union_type == "mean":
                final_score = np.mean([x for x in rubric_scores if x != -100])
            elif self.pipeline_config.reward_union_type == "all-or-nothing":
                final_score = 1 if all([x == 1 for x in rubric_scores if x != -100]) else 0
            elif self.pipeline_config.reward_union_type == "weighted-mean":
                raise NotImplementedError("Weighted mean reward type is not implemented yet")
            else:
                raise ValueError(f"Unsupported reward type: {self.pipeline_config.reward_union_type}")
            scores.append(final_score)
            rubric_scores_list.append(rubric_scores)
        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        rubric_scores_list = torch.tensor(rubric_scores_list, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
                "rubric_scores_list": rubric_scores_list,
            }
        )

        output.meta_info = {"metrics": metrics}
        return output
