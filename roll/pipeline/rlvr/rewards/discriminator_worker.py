import inspect
import os
import random
import threading
import time
from typing import Dict, Optional, Union, List

import ray
import torch
from codetiming import Timer
from tqdm import tqdm
import numpy as np

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    default_value_model_provider,
    default_actor_model_provider
)
from roll.platforms import current_platform
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_managers import state_offload_manger, log_gpu_memory_usage
from roll.utils.dynamic_batching import make_mini_batch_iter_for_dynamic_batching
from roll.utils.functionals import agg_loss, append_to_dict, compute_approx_kl, masked_mean, postprocess_generate
from roll.utils.offload_nccl import reload_process_groups
from roll.utils.offload_states import OffloadStateType
from roll.utils.prompt import prompt_maps
from tensordict import TensorDict
from roll.utils.logging import get_logger

logger = get_logger()

class DiscriminatorWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.strategy = create_strategy(worker=self)
        self.strategy.initialize(model_provider=default_value_model_provider)
        self.tokenizer = self.strategy.tokenizer
        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_values(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'values': values})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_values",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )
                values = results["values"]
                logger.info(f"value_size: {values.size()}")
                weights = torch.sigmoid(values) # 0 - 1
                logger.info(f"value after sigmoid: {weights.size()}")
            output = DataProto.from_dict(tensors={"weights": weights})
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, :] # B, SEQ_LENGTH
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}


class DiscriminatorWorker_P(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.strategy = create_strategy(worker=self)
        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def compute_log_probs(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """

        data = self.strategy.get_data_input(data)
        data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", False)
        metrics = {}

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
        ):
            data = data.to(current_platform.device_type)
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_log_probs
                )
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            output = DataProto.from_dict(tensors={"log_probs": results["log_probs"]})
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["attention_mask"]
        )

        return log_probs, {"log_probs": log_probs}

def discriminator_prompt(rubcic):
    return f"Identify which tokens in the response are relevant to the criteria.\n\nCriteria: {rubcic}\n\nResponse:\n\n"

# adiadas
# adi[token] adas[token] --> discrimiator eval
# adiadas [token] --> train
def  prepare_weights_input(responses_ids, data: DataProto, tokenizer, dp_size):
    logger.info(f"dp_size: {dp_size}")
    input_ids_list = []
    indexs = []
    response_idxs = []
    response_ids_list = []
    rubric_list  = []
    MAX_RUBRIC_COUNT = 20

    for idx, (prompt_id, response_ori_ids, rubrics) in enumerate(zip(
        data.non_tensor_batch["id"], responses_ids, data.non_tensor_batch["rubrics"])
    ):
        response_ori_ids = response_ori_ids.tolist()
        response_ids = []
        for reponse_id in response_ori_ids:
            if reponse_id in tokenizer.added_tokens_decoder:
                break
            response_ids.append(reponse_id)
        
        for rubric in rubrics[:MAX_RUBRIC_COUNT]:
            input_text = discriminator_prompt(rubric["description"])  # 这里应该只需要文本？对的✅
            prompt_ids = tokenizer.encode(input_text)
            indexs.append(idx)
            response_ids_list.append(response_ids)
            input_ids = prompt_ids + response_ids
            response_idx = range(len(prompt_ids), len(input_ids))
            response_idxs.append(response_idx)
            rubric_list.append(rubric)
            input_ids_list.append(input_ids)

    
    # Pad prompts to be a multiple of dp_size using the last element
    valid_index = len(indexs)
    remainder = valid_index % dp_size
    if remainder != 0:
        padding_needed = dp_size - remainder
        last_input_ids = input_ids_list[-1]
        input_ids_list.extend([last_input_ids] * padding_needed)

    # Pad input_ids_list to the same maximum length
    max_length = max(len(input_ids) for input_ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []

    for input_ids in input_ids_list:
        padding_length = max_length - len(input_ids)
        # Pad with tokenizer.pad_token_id
        padded_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)
    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
    # (>> 512 [0,0,0,1,1,2,2,2])
    return DataProto.from_dict(tensors={"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}),  \
        {"indexs": indexs, "response_idxs": response_idxs, "response_ids_list": response_ids_list,"rubric_list": rubric_list}, valid_index


def discriminator_prompt_p(rubcic, prompt):
    return f"{rubcic}\n\nAnswer following question:\n{prompt}"

def baseline_prompt_p(prompt):
    return f"Answer following question:\n{prompt}"

# adiadas
# adi[token] adas[token] --> discrimiator eval
# adiadas [token] --> train
def prepare_weights_input_for_p(responses_ids, data: DataProto, tokenizer, dp_size):
    logger.info(f"dp_size: {dp_size}")
    rubrics_list = []
    prompts = []
    input_ids_list = []
    indexs = []
    response_idxs = []
    response_ids_list = []
    MAX_RUBRIC_COUNT = 20

    for idx, (response_ori_ids, prompt) in enumerate(zip(responses_ids, data.non_tensor_batch["prompt"])):
        indexs.append(idx)
        response_ids = [reponse_id for reponse_id in response_ori_ids.tolist() if reponse_id != tokenizer.pad_token_id]
        response_ids_list.append(response_ids)
        prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": baseline_prompt_p(prompt)}], tokenize=True, add_generation_prompt=True)
        input_ids = prompt_ids + response_ids
        response_idx = range(len(prompt_ids), len(input_ids))
        response_idxs.append(response_idx)
        input_ids_list.append(input_ids)
        prompts.append(prompt)
        rubrics_list.append(None)

    for idx, (response_ori_ids, rubrics, prompt) in enumerate(zip(
                responses_ids, data.non_tensor_batch["criteria"], data.non_tensor_batch["prompt"])
    ):
        response_ids = [reponse_id for reponse_id in response_ori_ids.tolist() if reponse_id != tokenizer.pad_token_id]
        for rubric in rubrics[:MAX_RUBRIC_COUNT]:
            user_prompt = discriminator_prompt_p(rubric, prompt)
            prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=True, add_generation_prompt=True)
            indexs.append(idx)
            response_ids_list.append(response_ids)
            input_ids = prompt_ids + response_ids
            response_idx = range(len(prompt_ids), len(input_ids))
            response_idxs.append(response_idx)
            input_ids_list.append(input_ids)
            prompts.append(prompt)
            rubrics_list.append(rubric)

    
    # Pad prompts to be a multiple of dp_size using the last element
    valid_index = len(indexs)
    remainder = valid_index % dp_size
    if remainder != 0:
        padding_needed = dp_size - remainder
        last_input_ids = input_ids_list[-1]
        input_ids_list.extend([last_input_ids] * padding_needed)

    # Pad input_ids_list to the same maximum length
    max_length = max(len(input_ids) for input_ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []

    for input_ids in input_ids_list:
        padding_length = max_length - len(input_ids)
        # Pad with tokenizer.pad_token_id
        padded_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)
    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
    return DataProto.from_dict(tensors={"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}), {"indexs": indexs, "response_idxs": response_idxs, "response_ids_list": response_ids_list, "prompts": prompts, "rubrics": rubrics_list}, valid_index



def post_procsess_weights(actor_tokenizer, batch: DataProto, weights_addtion_info: Dict, weights: DataProto, valid_index: int):
    weights = weights.batch["weights"]
    indexs = weights_addtion_info["indexs"]
    response_idxs = weights_addtion_info["response_idxs"]
    response_ids_list = weights_addtion_info["response_ids_list"]
    rubric_list = weights_addtion_info["rubric_list"]
    input_ids_list = batch.batch["input_ids"]
    rubric_scores_list = batch.batch["rubric_scores_list"]
    input_ids_weight = torch.zeros_like(input_ids_list)
    weight_group = [[] for _ in range(len(batch))]
    score_group = [[] for _ in range(len(batch))]

    logger.info(f"len(batch): {len(batch)}")
    rubric_scores_list = [x for x in torch.flatten(rubric_scores_list).tolist() if x != -100] # > 512 <==> weights <==> indexs
    assert len(rubric_scores_list) == len(indexs)

    # 限制打印的样本数量，避免日志过多
    max_log_samples = 20
    sample_infos = []
    
    for idx_in_loop, (index, response_ids, response_idx, weight, rubric, rubric_score) in enumerate(zip(indexs, response_ids_list, response_idxs, weights[:valid_index], rubric_list, rubric_scores_list)):
        input_ids = input_ids_list[index]
        input_ids_weight = torch.zeros_like(input_ids) # input 部分无权重
        assert len(response_idx) == len(response_ids) 
        response_weight = weight[response_idx] # response_length, 1
        
        # 收集样本信息
        response_weight_list = response_weight.tolist() if isinstance(response_weight, torch.Tensor) else response_weight
        response_text = actor_tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 收集 token 级别的信息
        token_info_list = []
        for token_id, token_weight in zip(response_ids, response_weight_list):
            token_text = actor_tokenizer.decode([token_id], skip_special_tokens=False)
            token_info_list.append((token_text, token_weight))
        
        sample_infos.append({
            'index': index,
            'rubric': rubric,
            'response_text': response_text,
            'response_weight_list': response_weight_list,
            'token_info_list': token_info_list
        })

        
        response_start_pos = None
        input_ids = input_ids.tolist()
        for i in range(len(input_ids) - len(response_ids) + 1):
            if input_ids[i:i+len(response_ids)] == response_ids:
                response_start_pos = i
                break
        
        # If match found, place response weights in corresponding positions
        # if response_start_pos is None:
        #     input_text = actor_tokenizer.decode(input_ids)
        #     response_text = actor_tokenizer.decode(response_ids)
        #     logger.info(f"input_text: {input_text}\nresponse_text: {response_text}\ninput_ids: {input_ids}\nresponse_ids: {response_ids}")

        assert response_start_pos is not None
        response_end_pos = response_start_pos + len(response_ids)
        input_ids_weight[response_start_pos: response_end_pos] = response_weight
        # if index >= len(weight_group):
        #     logger.info(f"len group: {len(weight_group)}, index: {index}")
        token_weights = input_ids_weight
        # logger.info(f"index: {index}, token_weights: {token_weights.size()}")
        weight_group[index].append(token_weights.tolist())
        score_group[index].append(rubric_score)

    # 随机采样并打印样本信息
    if len(sample_infos) > 0:
        num_samples_to_log = min(max_log_samples, len(sample_infos))
        sampled_indices = random.sample(range(len(sample_infos)), num_samples_to_log)
        logger.info(f"Randomly sampling {num_samples_to_log} from {len(sample_infos)} samples.")
        
        for sampled_idx in sampled_indices:
            sample_info = sample_infos[sampled_idx]
            index = sample_info['index']
            rubric = sample_info['rubric']
            response_text = sample_info['response_text']
            response_weight_list = sample_info['response_weight_list']
            token_info_list = sample_info['token_info_list']
            
            # 打印 rubric
            logger.info(f"[Index {sampled_idx}] Rubric: {rubric}")
            
            # 打印 response 文本
            logger.info(f"[Index {sampled_idx}] Response text: {response_text}")
            
            # 打印 response weights
            logger.info(f"[Index {sampled_idx}] Response weights: {response_weight_list}")
            logger.info(f"[Index {sampled_idx}] Weight Shape: {len(response_weight_list)}")
            
            # 打印 token 级别的权重
            logger.info(f"[Index {sampled_idx}] Token-level weights: ")
            for token_text, token_weight in token_info_list:
                logger.info(f"[Index {sampled_idx}] Text: {repr(token_text)},  Weight: {token_weight:.6f}")

    # logger.info(f"len(weight_group): {len(weight_group)}")
    # rubric2token_scores = []
    # for token_rubric_rewards in weight_group:
    #     rubric2token_scores.append(torch.stack(token_rubric_rewards, dim=0).tolist())
    logger.info(f"weight_group len: {len(weight_group)}")
    logger.info(f"weight_group sample: {torch.tensor(weight_group[0]).size()}")
    logger.info(f"weight_group sample: {torch.tensor(weight_group[-1]).size()}")

    # [batch, rubric_size, seq_len], 每个样本的 rubric_size 可能不同。
    return weight_group, score_group

 
def post_procsess_weights_for_p(tokenizer, batch: DataProto, weights_addtion_info: Dict, log_probs_batch: DataProto, valid_index: int):
    log_probs = log_probs_batch.batch["log_probs"]
    logger.info(log_probs.size())
    batch_size = len(batch)

    base_line_log_probs = log_probs[:batch_size]
    base_response_idxs = weights_addtion_info["response_idxs"][:batch_size]
    base_line_response_log_probs = []
    for base_line_log_prob, base_response_idx in zip(base_line_log_probs, base_response_idxs):
        base_line_response_log_probs.append(base_line_log_prob[np.array(base_response_idx) - 1])
    logger.info(log_probs.size())

    logger.info(batch_size)
    logger.info(valid_index)
    rubric_log_probs = log_probs[batch_size: valid_index]
    indexs = weights_addtion_info["indexs"][batch_size:]
    response_idxs = weights_addtion_info["response_idxs"][batch_size:]
    response_ids_list = weights_addtion_info["response_ids_list"][batch_size:]
    prompts =  weights_addtion_info["prompts"][batch_size:]
    rubrics =  weights_addtion_info["rubrics"][batch_size:]
    input_ids_list = batch.batch["input_ids"]
    rubric_scores_list = batch.batch["rubric_scores_list"]
    input_ids_weight = torch.zeros_like(input_ids_list)
    weight_group = [[] for _ in range(len(batch))]
    score_group = [[] for _ in range(len(batch))]
    
    rubric_scores_list = [x for x in torch.flatten(rubric_scores_list).tolist() if x != -100] # > 512 <==> weights <==> indexs
    logger.info(f"len(rubric_scores_list): {len(rubric_scores_list)}")
    logger.info(f"len(indexs): {len(indexs)}")
    assert len(rubric_scores_list) == len(indexs)

    logger.info(len(indexs))
    logger.info(len(response_ids_list))
    logger.info(len(response_idxs))
    logger.info(len(rubric_log_probs))
    logger.info(len(rubric_scores_list))
    logger.info(len(weight_group))

    for index, response_ids, response_idx, log_prob, rubric_score, prompt, rubric in zip(indexs, response_ids_list, response_idxs, rubric_log_probs, rubric_scores_list, prompts, rubrics):
        input_ids = input_ids_list[index]
        input_ids_weight = torch.ones_like(input_ids, dtype=torch.float32)
        assert len(response_idx) == len(response_ids)
        rubric_response_log_prob = log_prob[np.array(response_idx) - 1]
        base_line_response_log_prob = base_line_response_log_probs[index]
        response_weight = rubric_response_log_prob - base_line_response_log_prob

        if index <= 5:
            logger.info("==========================================")
            logger.info(f"【PROMPT】\n{prompt}\n【RUBRIC】\n{rubric}【SCORE】\n{rubric_score}")
            logger.info("------------------------------------------")
            temp_weight = torch.softmax(response_weight, -1)
            logger.info([(tokenizer.decode([response_id]), weight) for response_id, weight in zip(response_ids, temp_weight)])

        response_start_pos = None
        input_ids = input_ids.tolist()
        for i in range(len(input_ids) - len(response_ids) + 1):
            if input_ids[i:i+len(response_ids)] == response_ids:
                response_start_pos = i
                break
        
        # If match found, place response weights in corresponding positions
        # if response_start_pos is None:
        #     input_text = actor_tokenizer.decode(input_ids)
        #     response_text = actor_tokenizer.decode(response_ids)
        #     logger.info(f"input_text: {input_text}\nresponse_text: {response_text}\ninput_ids: {input_ids}\nresponse_ids: {response_ids}")

        assert response_start_pos is not None
        response_end_pos = response_start_pos + len(response_ids)
        input_ids_weight[response_start_pos - 1: response_end_pos - 1] = response_weight
        # if index >= len(weight_group):
        #     logger.info(f"len group: {len(weight_group)}, index: {index}")
        token_weights = input_ids_weight
        # logger.info(f"index: {index}, token_weights: {token_weights.size()}")
        weight_group[index].append(token_weights)
        score_group[index].append(rubric_score)

    # rubric2token_scores = []
    # for token_rubric_rewards in weight_group:
    #     token_rubric_rewards = torch.stack(token_rubric_rewards, dim=0).reshape(len(token_rubric_rewards), -1)
    #     # token_rubric_weights = torch.softmax(token_rubric_rewards, dim=0)
    #     rubric2token_scores.append(token_rubric_rewards)
    # logger.info(f"weight_group len: {len(rubric2token_scores)}")
    # logger.info(f"weight_group sample: {torch.tensor(rubric2token_scores[0]).size()}")
    # logger.info(f"weight_group sample: {torch.tensor(rubric2token_scores[-1]).size()}")
    return weight_group, score_group