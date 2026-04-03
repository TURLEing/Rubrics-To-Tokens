import copy
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys
from roll.datasets.chat_template import get_chat_template
from roll.distributed.scheduler.generate_scheduler import DynamicSamplingScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger
from roll.utils.metrics.metrics_manager import MetricsManager


logger = get_logger()


class RLVRValidator:
    """Encapsulates validation flows for RLVR pipelines."""

    def __init__(
        self,
        pipeline_config: Any,
        tokenizer: Any,
        actor_infer: Any,
        rewards: Dict[str, Any],
        val_dataset: Optional[Any] = None,
        val_generate_scheduler: Optional[Any] = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.tokenizer = tokenizer
        self.actor_infer = actor_infer
        self.rewards = rewards
        self.val_dataset = val_dataset
        self.val_generate_scheduler = val_generate_scheduler
        self.benchmark_generate_schedulers: Dict[str, Any] = {}
        self.ifeval_inputs: Optional[List[Any]] = None
        self.ifbench_inputs: Optional[List[Any]] = None
        self.muldimif_inputs: Optional[List[Dict[str, Any]]] = None

    def has_validation(self) -> bool:
        benchmark_cfg = getattr(self.pipeline_config, "validation_benchmark", None)
        return self.val_dataset is not None or bool(benchmark_cfg and benchmark_cfg.enabled)

    def shutdown(self) -> None:
        if self.val_generate_scheduler is not None:
            ray.get(self.val_generate_scheduler.shutdown.remote())
        for benchmark_generate_scheduler in self.benchmark_generate_schedulers.values():
            ray.get(benchmark_generate_scheduler.shutdown.remote())

    @torch.no_grad()
    def validate(self, global_step: int) -> Dict[str, float]:
        benchmark_names = self._get_enabled_benchmark_names()
        if benchmark_names:
            return self._validate_benchmarks(global_step=global_step, benchmark_names=benchmark_names)
        return self._validate_reward_model(global_step=global_step)

    def _get_enabled_benchmark_names(self) -> List[str]:
        benchmark_cfg = getattr(self.pipeline_config, "validation_benchmark", None)
        if not benchmark_cfg or not benchmark_cfg.enabled:
            return []

        benchmark_names = benchmark_cfg.name
        if isinstance(benchmark_names, str):
            benchmark_names = [benchmark_names]
        return [benchmark_name.lower() for benchmark_name in benchmark_names]

    def _get_validation_generation_config(self) -> Dict[str, Any]:
        if self.pipeline_config.validation and self.pipeline_config.validation.generating_args:
            return self.pipeline_config.validation.generating_args.to_dict()
        return self.pipeline_config.actor_infer.generating_args.to_dict()

    def _get_validation_template_name(self) -> str:
        if self.pipeline_config.validation and self.pipeline_config.validation.data_args:
            template_name = self.pipeline_config.validation.data_args.template
            if template_name:
                return template_name
        if self.pipeline_config.global_template:
            return self.pipeline_config.global_template
        return self.pipeline_config.actor_train.data_args.template

    def _resolve_repo_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        return os.path.join(repo_root, path)

    @staticmethod
    def _to_object_array(values: List[Any]) -> np.ndarray:
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    def _load_ifeval_evaluation_lib(self):
        benchmark_parent = self._resolve_repo_path("Benchmark")
        if benchmark_parent not in sys.path:
            sys.path.insert(0, benchmark_parent)
        return importlib.import_module("instruction_following_eval.evaluation_lib")

    def _load_ifbench_evaluation_lib(self):
        benchmark_dir = self._resolve_repo_path("Benchmark/IFBench")
        if benchmark_dir not in sys.path:
            sys.path.insert(0, benchmark_dir)
        return importlib.import_module("evaluation_lib")

    def _load_muldimif_evaluation_module(self):
        module_name = "muldimif_evaluation"
        if module_name in sys.modules:
            return sys.modules[module_name]

        benchmark_code_dir = self._resolve_repo_path("Benchmark/MulDimIF/Code")
        benchmark_eval_dir = os.path.join(benchmark_code_dir, "evaluation")
        for path in [benchmark_eval_dir, benchmark_code_dir]:
            if path not in sys.path:
                sys.path.insert(0, path)

        evaluation_path = os.path.join(benchmark_eval_dir, "evaluation.py")
        spec = importlib.util.spec_from_file_location(module_name, evaluation_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load MulDimIF evaluation module from {evaluation_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _build_infer_batch_from_prompts(
        self, prompts: List[str], generation_config: Dict[str, Any], global_step: int
    ) -> DataProto:
        chat_template_func = get_chat_template(self._get_validation_template_name(), self.tokenizer)
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        text_list = [chat_template_func(conversation) for conversation in conversations]
        encodings = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.pipeline_config.val_prompt_length,
        )
        attention_mask = encodings["attention_mask"].to(torch.long)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        return DataProto.from_dict(
            tensors={
                "input_ids": encodings["input_ids"],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors={"prompt": self._to_object_array(prompts)},
            meta_info={
                "is_offload_states": False,
                "generation_config": generation_config,
                "global_step": global_step,
            },
        )

    def _get_benchmark_domain(self) -> str:
        if self.rewards:
            return next(iter(self.rewards.keys()))
        raise ValueError("Benchmark validation requires at least one reward domain for scheduler setup")

    def _get_benchmark_input_data_path(self, benchmark_name: str) -> str:
        benchmark_cfg = self.pipeline_config.validation_benchmark
        input_data = benchmark_cfg.input_data
        if isinstance(input_data, dict):
            benchmark_input_data = input_data.get(benchmark_name)
            if benchmark_input_data:
                return self._resolve_repo_path(benchmark_input_data)
        elif input_data:
            return self._resolve_repo_path(input_data)

        default_input_paths = {
            "ifbench": "Benchmark/IFBench/data/IFBench_test.jsonl",
            "muldimif": "Benchmark/MulDimIF/Data/test.json",
        }
        if benchmark_name in default_input_paths:
            return self._resolve_repo_path(default_input_paths[benchmark_name])

        raise ValueError(f"validation_benchmark.input_data must be set when {benchmark_name} validation is enabled")

    def _build_instruction_following_dataset(self, inputs: List[Any], tag: str) -> datasets.Dataset:
        if not inputs:
            raise ValueError(f"{tag} validation inputs are empty")

        benchmark_domain = self._get_benchmark_domain()
        chat_template_func = get_chat_template(self._get_validation_template_name(), self.tokenizer)
        text_list = [chat_template_func([{"role": "user", "content": item.prompt}]) for item in inputs]
        encodings = self.tokenizer(
            text_list,
            truncation=True,
            max_length=self.pipeline_config.val_prompt_length,
        )

        rows = []
        for item, input_ids, attention_mask in zip(inputs, encodings["input_ids"], encodings["attention_mask"]):
            rows.append(
                {
                    "key": item.key,
                    "prompt": item.prompt,
                    "instruction_id_list": item.instruction_id_list,
                    "kwargs": item.kwargs,
                    "domain": benchmark_domain,
                    "tag": tag,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
        return datasets.Dataset.from_list(rows)

    def _build_ifeval_dataset(self) -> datasets.Dataset:
        benchmark_cfg = self.pipeline_config.validation_benchmark
        evaluation_lib = self._load_ifeval_evaluation_lib()
        input_data_path = self._get_benchmark_input_data_path("ifeval")
        inputs = evaluation_lib.read_prompt_list(input_data_path)
        if benchmark_cfg.max_samples > 0:
            inputs = inputs[: benchmark_cfg.max_samples]
        self.ifeval_inputs = inputs
        return self._build_instruction_following_dataset(inputs, "ifeval")

    def _build_ifbench_dataset(self) -> datasets.Dataset:
        benchmark_cfg = self.pipeline_config.validation_benchmark
        evaluation_lib = self._load_ifbench_evaluation_lib()
        input_data_path = self._get_benchmark_input_data_path("ifbench")
        inputs = evaluation_lib.read_prompt_list(input_data_path)
        if benchmark_cfg.max_samples > 0:
            inputs = inputs[: benchmark_cfg.max_samples]
        self.ifbench_inputs = inputs
        return self._build_instruction_following_dataset(inputs, "ifbench")

    def _build_muldimif_dataset(self) -> datasets.Dataset:
        benchmark_cfg = self.pipeline_config.validation_benchmark
        input_data_path = self._get_benchmark_input_data_path("muldimif")
        with open(input_data_path, "r", encoding="utf-8") as f:
            inputs = json.load(f)

        if benchmark_cfg.max_samples > 0:
            inputs = inputs[: benchmark_cfg.max_samples]
        self.muldimif_inputs = inputs

        benchmark_domain = self._get_benchmark_domain()
        chat_template_func = get_chat_template(self._get_validation_template_name(), self.tokenizer)
        text_list = [chat_template_func(item["conversations"]) for item in inputs]
        encodings = self.tokenizer(
            text_list,
            truncation=True,
            max_length=benchmark_cfg.muldimif_max_length,
        )

        rows = []
        for item, input_ids, attention_mask in zip(inputs, encodings["input_ids"], encodings["attention_mask"]):
            prompt = next(
                (message["content"] for message in reversed(item["conversations"]) if message.get("role") == "user"),
                "",
            )
            rows.append(
                {
                    "id": item["id"],
                    "prompt": prompt,
                    "domain": benchmark_domain,
                    "tag": "muldimif",
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
        return datasets.Dataset.from_list(rows)

    def _get_or_create_benchmark_scheduler(self, benchmark_name: str):
        if benchmark_name in self.benchmark_generate_schedulers:
            return self.benchmark_generate_schedulers[benchmark_name]

        if benchmark_name == "ifeval":
            benchmark_dataset = self._build_ifeval_dataset()
        elif benchmark_name == "ifbench":
            benchmark_dataset = self._build_ifbench_dataset()
        elif benchmark_name == "muldimif":
            benchmark_dataset = self._build_muldimif_dataset()
        else:
            raise ValueError(f"Unsupported benchmark validation name: {benchmark_name}")
        val_pipeline_config = copy.deepcopy(self.pipeline_config)
        val_pipeline_config.is_use_additional_prompts = False
        benchmark_generate_scheduler = ray.remote(DynamicSamplingScheduler).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        ).remote(pipeline_config=val_pipeline_config)
        ray.get(
            benchmark_generate_scheduler.set_scheduler.remote(
                actor_cluster=self.actor_infer,
                reward_clusters=self.rewards,
                dataset=benchmark_dataset,
                collect_fn_cls=DataCollatorWithPaddingForPaddedKeys,
                collect_fn_kwargs=dict(max_length=self.pipeline_config.val_prompt_length, padding="max_length"),
                is_val=True,
            )
        )
        self.benchmark_generate_schedulers[benchmark_name] = benchmark_generate_scheduler
        return benchmark_generate_scheduler

    @staticmethod
    def _summarize_instruction_following_outputs(outputs: List[Any], prefix: str) -> Dict[str, float]:
        prompt_total = len(outputs)
        prompt_correct = sum(output.follow_all_instructions for output in outputs)
        instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
        instruction_correct = sum(sum(output.follow_instruction_list) for output in outputs)
        return {
            f"{prefix}/prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
            f"{prefix}/instruction_accuracy": instruction_correct / instruction_total if instruction_total else 0.0,
            f"{prefix}/prompt_total": float(prompt_total),
            f"{prefix}/instruction_total": float(instruction_total),
        }

    @staticmethod
    def _build_val_score_metrics(score: float, score_name: str) -> Dict[str, float]:
        return {
            f"val_score/{score_name}": score,
        }

    def _save_instruction_following_outputs(
        self,
        benchmark_name: str,
        evaluation_lib: Any,
        global_step: int,
        response_records: List[Dict[str, str]],
        strict_outputs: List[Any],
        loose_outputs: List[Any],
    ) -> None:
        benchmark_cfg = self.pipeline_config.validation_benchmark
        if not benchmark_cfg.save_outputs:
            return
        output_dir = self._resolve_repo_path(benchmark_cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        response_path = os.path.join(output_dir, f"{benchmark_name}_responses_step_{global_step}.jsonl")
        strict_path = os.path.join(output_dir, f"{benchmark_name}_eval_results_strict_step_{global_step}.jsonl")
        loose_path = os.path.join(output_dir, f"{benchmark_name}_eval_results_loose_step_{global_step}.jsonl")

        with open(response_path, "w", encoding="utf-8") as f:
            for record in response_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        evaluation_lib.write_outputs(strict_path, strict_outputs)
        evaluation_lib.write_outputs(loose_path, loose_outputs)

    @staticmethod
    def _summarize_muldimif_outputs(judge_data: List[Dict[str, Any]], prefix: str) -> Dict[str, float]:
        prompt_total = len(judge_data)
        prompt_correct = sum(sum(item["judges"]) == len(item["judges"]) for item in judge_data)
        constraint_total = sum(len(item["judges"]) for item in judge_data)
        constraint_correct = sum(sum(item["judges"]) for item in judge_data)
        return {
            f"{prefix}/prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
            f"{prefix}/constraint_accuracy": constraint_correct / constraint_total if constraint_total else 0.0,
            f"{prefix}/prompt_total": float(prompt_total),
            f"{prefix}/constraint_total": float(constraint_total),
        }

    def _save_muldimif_outputs(
        self,
        global_step: int,
        generated_records: List[Dict[str, Any]],
        score: Dict[str, Any],
        judge_data: List[Dict[str, Any]],
    ) -> None:
        benchmark_cfg = self.pipeline_config.validation_benchmark
        if not benchmark_cfg.save_outputs:
            return

        output_dir = self._resolve_repo_path(benchmark_cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        response_path = os.path.join(output_dir, f"muldimif_responses_step_{global_step}.jsonl")
        score_path = os.path.join(output_dir, f"muldimif_score_step_{global_step}.json")
        judge_path = os.path.join(output_dir, f"muldimif_judged_step_{global_step}.jsonl")

        with open(response_path, "w", encoding="utf-8") as f:
            for record in generated_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(score, f, ensure_ascii=False, indent=2)

        with open(judge_path, "w", encoding="utf-8") as f:
            for record in judge_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def _validate_instruction_following_benchmark(
        self,
        global_step: int,
        benchmark_name: str,
        evaluation_lib: Any,
        benchmark_inputs: List[Any],
    ) -> Dict[str, float]:
        benchmark_scheduler = self._get_or_create_benchmark_scheduler(benchmark_name)

        generation_config = self._get_validation_generation_config()
        if generation_config.get("num_return_sequences", 1) != 1:
            raise ValueError(
                f"{benchmark_name} validation requires validation.generating_args.num_return_sequences == 1"
            )

        batch = DataProto(
            meta_info={
                "is_offload_states": False,
                "generation_config": generation_config,
                "global_step": global_step,
                "skip_rewards": True,
            }
        )
        generate_output: DataProto = ray.get(
            benchmark_scheduler.get_batch.remote(
                data=batch,
                global_step=global_step,
                batch_size=len(benchmark_inputs),
            ),
            timeout=self.pipeline_config.rpc_timeout,
        )

        prompts = list(generate_output.non_tensor_batch["prompt"])
        responses = self.tokenizer.batch_decode(generate_output.batch["responses"], skip_special_tokens=True)
        prompt_to_response = {prompt: response.strip() for prompt, response in zip(prompts, responses)}
        response_records = [{"prompt": prompt, "response": prompt_to_response[prompt]} for prompt in prompts]
        strict_outputs = [
            evaluation_lib.test_instruction_following_strict(item, prompt_to_response) for item in benchmark_inputs
        ]
        loose_outputs = [
            evaluation_lib.test_instruction_following_loose(item, prompt_to_response) for item in benchmark_inputs
        ]

        metrics: Dict[str, float] = {}
        metrics.update(self._summarize_instruction_following_outputs(strict_outputs, f"val/{benchmark_name}/strict"))
        metrics.update(self._summarize_instruction_following_outputs(loose_outputs, f"val/{benchmark_name}/loose"))
        metrics.update(
            self._build_val_score_metrics(
                score=metrics[f"val/{benchmark_name}/strict/prompt_accuracy"],
                score_name=f"{benchmark_name}_strict_prompt_level",
            )
        )
        logger.info(json.dumps(metrics, ensure_ascii=False))
        self._save_instruction_following_outputs(
            benchmark_name=benchmark_name,
            evaluation_lib=evaluation_lib,
            global_step=global_step,
            response_records=response_records,
            strict_outputs=strict_outputs,
            loose_outputs=loose_outputs,
        )
        return metrics

    @torch.no_grad()
    def _validate_benchmarks(self, global_step: int, benchmark_names: List[str]) -> Dict[str, float]:
        benchmark_metrics: Dict[str, float] = {}

        for benchmark_name in benchmark_names:
            if benchmark_name == "ifeval":
                metrics = self._validate_ifeval(global_step=global_step)
            elif benchmark_name == "ifbench":
                metrics = self._validate_ifbench(global_step=global_step)
            elif benchmark_name == "muldimif":
                metrics = self._validate_muldimif(global_step=global_step)
            else:
                raise ValueError(f"Unsupported benchmark validation name: {benchmark_name}")

            benchmark_metrics.update(metrics)

        return benchmark_metrics

    @torch.no_grad()
    def _validate_ifeval(self, global_step: int) -> Dict[str, float]:
        evaluation_lib = self._load_ifeval_evaluation_lib()
        if self.ifeval_inputs is None:
            self._get_or_create_benchmark_scheduler("ifeval")
        return self._validate_instruction_following_benchmark(
            global_step=global_step,
            benchmark_name="ifeval",
            evaluation_lib=evaluation_lib,
            benchmark_inputs=self.ifeval_inputs,
        )

    @torch.no_grad()
    def _validate_ifbench(self, global_step: int) -> Dict[str, float]:
        evaluation_lib = self._load_ifbench_evaluation_lib()
        if self.ifbench_inputs is None:
            self._get_or_create_benchmark_scheduler("ifbench")
        return self._validate_instruction_following_benchmark(
            global_step=global_step,
            benchmark_name="ifbench",
            evaluation_lib=evaluation_lib,
            benchmark_inputs=self.ifbench_inputs,
        )

    @torch.no_grad()
    def _validate_muldimif(self, global_step: int) -> Dict[str, float]:
        evaluation_module = self._load_muldimif_evaluation_module()
        if self.muldimif_inputs is None:
            self._get_or_create_benchmark_scheduler("muldimif")

        generation_config = self._get_validation_generation_config()
        if generation_config.get("num_return_sequences", 1) != 1:
            raise ValueError("muldimif validation requires validation.generating_args.num_return_sequences == 1")

        benchmark_scheduler = self._get_or_create_benchmark_scheduler("muldimif")
        batch = DataProto(
            meta_info={
                "is_offload_states": False,
                "generation_config": generation_config,
                "global_step": global_step,
                "skip_rewards": True,
            }
        )
        generate_output: DataProto = ray.get(
            benchmark_scheduler.get_batch.remote(
                data=batch,
                global_step=global_step,
                batch_size=len(self.muldimif_inputs),
            ),
            timeout=self.pipeline_config.rpc_timeout,
        )

        responses = self.tokenizer.batch_decode(generate_output.batch["responses"], skip_special_tokens=True)
        if len(responses) != len(self.muldimif_inputs):
            raise ValueError(
                f"MulDimIF response count mismatch: got {len(responses)} responses for {len(self.muldimif_inputs)} inputs"
            )

        # Build id->response mapping to handle shuffled dataset ordering in the scheduler
        output_ids = list(generate_output.non_tensor_batch["id"])
        id_to_response = {item_id: response.strip() for item_id, response in zip(output_ids, responses)}

        generated_records = []
        for item in self.muldimif_inputs:
            item_id = item["id"]
            if item_id not in id_to_response:
                raise ValueError(f"MulDimIF: missing response for item id={item_id}")
            generated_item = copy.deepcopy(item)
            generated_item["conversations"].append(
                {
                    "role": "assistant",
                    "content": id_to_response[item_id],
                }
            )
            generated_records.append(generated_item)

        # Debug: log a few samples to inspect prompt truncation and thinking blocks
        for debug_item in generated_records[:3]:
            item_id = debug_item["id"]
            user_prompt = next(
                (m["content"] for m in debug_item["conversations"] if m.get("role") == "user"), ""
            )
            response = debug_item["conversations"][-1]["content"]
            logger.info(
                f"[MulDimIF debug] id={item_id} "
                f"prompt_tail={repr(user_prompt[-200:])} "
                f"response_head={repr(response[:300])}"
            )

        judge_data = evaluation_module.check(copy.deepcopy(generated_records))
        score = evaluation_module.get_score(judge_data)

        metrics = self._summarize_muldimif_outputs(judge_data, "val/muldimif")
        metrics.update(
            self._build_val_score_metrics(
                score=metrics["val/muldimif/prompt_accuracy"],
                score_name="muldimif_prompt_level",
            )
        )
        logger.info(json.dumps(metrics, ensure_ascii=False))
        self._save_muldimif_outputs(
            global_step=global_step,
            generated_records=generated_records,
            score=score,
            judge_data=judge_data,
        )
        return metrics

    @torch.no_grad()
    def _validate_reward_model(self, global_step: int) -> Dict[str, float]:
        if self.val_generate_scheduler is None or self.val_dataset is None:
            raise ValueError("Reward-model validation requires both val_dataset and val_generate_scheduler")

        val_metrics_mgr = MetricsManager()
        batch = DataProto()

        with Timer(name="step_generate", logger=None) as step_generate_timer:
            reward_system_config = copy.deepcopy(self.pipeline_config.reward_system_config)
            experiment_id = reward_system_config.get("experiment_id", "roll_default_task_id")
            reward_system_config["experiment_id"] = experiment_id + "_val"
            batch.meta_info = {
                "is_offload_states": False,
                "generation_config": self.pipeline_config.validation.generating_args.to_dict(),
                "global_step": global_step,
                "reward_system_config": reward_system_config,
            }

            generate_output: DataProto = ray.get(
                self.val_generate_scheduler.get_batch.remote(
                    data=batch,
                    global_step=global_step,
                    batch_size=len(self.val_dataset),
                ),
                timeout=self.pipeline_config.rpc_timeout,
            )

            generate_output.meta_info.pop("is_offload_states", None)
            val_metrics_mgr.add_metric("time/step_generate", step_generate_timer.last)

        batch = generate_output
        val_correct_mean = (batch.batch["scores"] == 1).detach().float().mean().item()
        val_metrics_mgr.add_metric("val_correct/all/mean", val_correct_mean)
        logger.info(json.dumps({"val_correct/all/mean": val_correct_mean}, ensure_ascii=False))

        epoch_batch = batch.pop(batch_keys=["scores"], non_tensor_batch_keys=["tag"])
        grouped_batch = epoch_batch.group_by("tag")
        for group_key, group_batch in grouped_batch.items():
            score_mean = group_batch.batch["scores"].mean().item()
            logger.info(f"val_correct/{group_key}:  {score_mean}")
            val_metrics_mgr.add_domain_metrics(
                "val_correct",
                {f"{group_key}/mean": (group_batch.batch["scores"] == 1).detach().float().mean().item()},
            )

        return val_metrics_mgr.get_metrics()
