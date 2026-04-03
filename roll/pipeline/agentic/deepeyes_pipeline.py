import json
import os
from typing import Any, Dict, List

import ray
import torch
import numpy as np
from codetiming import Timer
from ray.util.timer import _Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider, get_extra_data_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.utils import compute_response_level_rewards
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.agentic.agentic_pipeline import get_episode_scores, compute_data_metrics
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    apply_kl_penalty,
    expand_to_token_level,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    agg_loss,
    compute_token_reward,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.offload_states import OffloadStateType
from roll.utils.logging import get_logger



logger = get_logger()


class DeepEyesPipeline(BasePipeline):
    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        # rollout_batch_size should be not less than `num_env_groups * group_size`.
        # If bigger than it, envs would run multiple episodes to produce more than one trajectory.
        # While we request them to be equal to match data and rollout items order simplely
        assert (
            self.pipeline_config.rollout_batch_size
            == self.pipeline_config.train_env_manager.num_env_groups
            * self.pipeline_config.train_env_manager.group_size
        ), "rollout_batch_size should equal to `num_env_groups * group_size` currently"
        self.train_batch_size = (
            self.pipeline_config.rollout_batch_size // self.pipeline_config.train_env_manager.group_size
        )

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        download_clusters = [self.actor_train, self.actor_infer, self.reference]
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)

        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        download_clusters.extend(self.rewards.values())
        if self.rewards:
            assert len(self.rewards) == 1, "only support one reward currently"
            self.reward = self.rewards[list(self.rewards.keys())[0]]

        self.download_models(*download_clusters)
        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        self.tokenizer = self.processor.tokenizer

        self.train_rollout_scheduler = (
            ray.remote(RolloutScheduler)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            )
            .remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.train_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="train",
            )
        )
        self.val_rollout_scheduler = None
        if self.pipeline_config.eval_steps > 0:
            self.val_rollout_scheduler = (
                ray.remote(RolloutScheduler)
                .options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(), soft=False
                    )
                )
                .remote(
                    config=self.pipeline_config,
                    env_manager_config=self.pipeline_config.val_env_manager,
                    resource_manager=self.resource_manager,
                    infer_cluster=self.actor_infer,
                    mode="val",
                )
            )
            self.val_dataset_manager = GlobalDatasetManager.options(
                name=f"val_dataset_manager", get_if_exists=True, namespace=RAY_NAMESPACE
            ).remote()
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True))
        if self.rewards:
            refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=True))
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

    @torch.no_grad()
    def run(self):
        tps_timer = _Timer(window_size=5)
        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}

            # Add overall step timing
            with Timer(name="pipeline_step_total", logger=None) as step_timer:
                with tps_timer:
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    ray.get(self.train_rollout_scheduler.suspend.remote())
                    if self.pipeline_config.async_pipeline:
                        self.actor_infer.offload_states(include=OffloadStateType.other_params)

                    with Timer(name="model_update", logger=None) as model_update_timer:
                        model_update_metrics: Dict = self.model_update(global_step)
                    metrics["time/step_model_update"] = model_update_timer.last
                    metrics.update(model_update_metrics)

                    self.actor_infer.load_states()

                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    if self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0:
                        metrics.update(self.val(global_step=global_step))

                    with Timer(name="rollout", logger=None) as rollout_timer:
                        batch.meta_info["is_offload_states"] = True
                        batch = ray.get(
                            self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size)
                        )
                    metrics["time/rollout"] = rollout_timer.last
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    batch.meta_info["global_step"] = global_step
                    # mark here to make megatron get_data_input broadcast with non_batch_tensor
                    batch.meta_info["_broadcast_non_tensor_batch"] = True
                    if not (self.pipeline_config.async_pipeline > 0):
                        self.actor_infer.offload_states()

                    if self.rewards:
                        for reward_cluster in self.rewards.values():
                            reward_cluster.load_states()
                        with Timer(name="compute_rewards", logger=None) as cal_timer:
                            rewards_refs: List[ray.ObjectRef] = self.reward.compute_rewards(
                                batch.select(batch_keys=[], non_tensor_batch_keys=["message", "question", "ground_truth"]),
                                blocking=False,
                            )
                            rewards = DataProto.materialize_concat(data_refs=rewards_refs)
                            # `compute_response_level_rewards` uses 2D "scores" to compute rewards
                            # `compute_data_metrics` uses 1D "episode_scores" to compute metrics
                            batch.non_tensor_batch["episode_scores"] = np.array(
                                rewards.batch["scores"].tolist(), dtype=object
                            )
                            batch.batch["scores"] = rewards.batch["scores"].unsqueeze(1)
                            metrics.update(reduce_metrics(rewards.meta_info.pop("metrics", {})))
                        for reward_cluster in self.rewards.values():
                            reward_cluster.offload_states()
                        metrics["time/compute_rewards"] = cal_timer.last

                    with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                        ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                        ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                        ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        batch = batch.union(ref_log_probs)
                        avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                        metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                        metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                    metrics["time/ref_log_probs_values"] = cal_timer.last

                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        batch.meta_info["is_offload_states"] = False
                        old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                        if self.pipeline_config.adv_estimator == "gae":
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                        old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                        batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                        avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                        metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})

                        agg_entropy = agg_loss(
                            loss_mat=old_log_probs.batch["entropy"],
                            loss_mask=batch.batch["response_mask"][:, 1:],
                            loss_agg_mode="token-mean",
                        )
                        metrics.update({"critic/entropy/mean": agg_entropy.item()})

                        metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                    metrics["time/old_log_probs_values"] = cal_old_logpb_timer.last

                    with Timer(name="adv", logger=None) as timer:
                        # Rewards need to be processed after grouping
                        # We can group by tag(env_type)/traj_group_id(group)/batch(rollout_batch)... to compute rewards / advantages
                        # The compute_response_level_rewards function injects a response_level_rewards key into batch.batch.
                        batch, reward_metrics = compute_response_level_rewards(
                            batch=batch, pipeline_config=self.pipeline_config
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        metrics.update(reward_metrics)

                        batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
                        metrics.update(token_level_metrics)

                        # Is the advantage calculated globally across the batch, or within each group?
                        batch = compute_advantage(
                            data=batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    metrics["time/adv"] = timer.last

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    # implement critic warmup
                    if self.pipeline_config.critic_warmup <= global_step:
                        # update actor
                        actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                        actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                        metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                    tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                data_metrics = compute_data_metrics(batch=batch)

            metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
            metrics.update(data_metrics)
            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]

            # do ckpt
            self.state.step = global_step
            self.state.log_history.append(metrics)

            self.do_checkpoint(global_step=global_step)

            metrics["time/step_total"] = step_timer.last
            self.tracker.log(values=metrics, step=global_step)

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                    )

                prompt_mask = batch.batch["prompt_mask"]
                non_prompt_mask = torch.logical_not(batch.batch["prompt_mask"])
                input_ids = batch.batch["input_ids"]
                prompt_ids = torch.where(
                    prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                )
                response_ids = torch.where(
                    non_prompt_mask.bool(), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id)
                )

                generate_res = []
                prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                episode_scores = batch.non_tensor_batch["episode_scores"].tolist()
                for prompt, response, episode_score in zip(prompts, responses, episode_scores):
                    generate_res.append({"prompt": prompt, "response": response, "episode_score": episode_score})
                logger.info(json.dumps(generate_res[:10], ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")

        ray.get(self.train_rollout_scheduler.shutdown.remote())
        if self.val_rollout_scheduler:
            ray.get(self.val_rollout_scheduler.shutdown.remote())

        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self, global_step):
        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step
        ray.get(self.val_dataset_manager.reset.remote())
        eval_batches = []
        # make env return some flags to indicate val dataset done
        dataset_done = False
        while not dataset_done:
            # GroupQueueManager guarantees get_batch episode by episode
            eval_batch = ray.get(
                self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size)
            )
            eval_batches.append(eval_batch)
            dataset_done = eval_batch.meta_info.get("is_last_in_epoch", False)
        eval_batch = DataProto.concat(eval_batches)
        if self.rewards:
            rewards_refs: List[ray.ObjectRef] = self.reward.compute_rewards(
                eval_batch.select(batch_keys=[], non_tensor_batch_keys=["message", "question", "ground_truth"]),
                blocking=False,
            )
            rewards = DataProto.materialize_concat(data_refs=rewards_refs)
            eval_batch.non_tensor_batch["episode_scores"] = np.array(rewards.batch["scores"].tolist(), dtype=object)
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()
        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
        return metrics
