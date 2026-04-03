import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl
from roll.pipeline.agentic.utils import compute_segment_masked_mean


class ActorWorker(BaseActorWorker):
    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
        infer_log_probs = data.batch.get("infer_logprobs", old_log_probs)
        infer_log_probs = infer_log_probs if len(infer_log_probs) > 0 else old_log_probs

        # for train infer diff
        train_infer_ratio = (old_log_probs - infer_log_probs).exp()
        train_infer_diff = old_log_probs.exp() - infer_log_probs.exp()
        train_infer_ratio_segment = compute_segment_masked_mean(old_log_probs - infer_log_probs, response_mask).exp()
        train_infer_diff_segment = compute_segment_masked_mean(
            old_log_probs.exp() - infer_log_probs.exp(), response_mask
        )

        if self.pipeline_config.ratio_type == "segment":
            # 计算序列级别的 ratio：对每段连续的1分别计算 masked_mean，不连续的段不相乘
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = compute_segment_masked_mean(log_ratio, response_mask)
            ratio = masked_log_ratio.exp()
        else:
            ratio = (log_probs - old_log_probs).exp()

        pg_clip_low = (
            self.pipeline_config.pg_clip_low
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        pg_clip_high = (
            self.pipeline_config.pg_clip_high
            if self.pipeline_config.use_pg_clip_range
            else self.pipeline_config.pg_clip
        )
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
        pg_loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

        # train-infer part
        train_infer_metric = {}

        # train infer is correct
        if self.pipeline_config.is_correct.is_ratio_type:
            if self.pipeline_config.is_correct.is_ratio_type == "token":
                is_ratio = train_infer_ratio
            elif self.pipeline_config.is_correct.is_ratio_type == "segment":
                is_ratio = train_infer_ratio_segment
            else:
                raise ValueError(
                    f"Unsupported is_ratio_type: {self.pipeline_config.is_correct.is_ratio_type}. "
                    f"Supported types are 'token' and 'segment'."
                )

            pg_loss = is_ratio.clamp(0, self.pipeline_config.is_correct.is_upper_bound) * pg_loss
            # TODO ：是否需要mask？
            train_infer_metric["actor/rollout_importance_sampling_clip"] = (
                (is_ratio > self.pipeline_config.is_correct.is_upper_bound).float().mean().detach().item()
            )

        # train-infer token filter
        if self.pipeline_config.token_filter.is_mask:
            token_is_ratio_mask = (train_infer_ratio <= self.pipeline_config.token_filter.is_clamp_high).float() * (
                train_infer_ratio >= self.pipeline_config.token_filter.is_clamp_low
            ).float()
            train_infer_metric["actor/train_infer_token_is_ratio_mask_mean"] = (
                masked_mean(token_is_ratio_mask, response_mask, dim=-1).mean().detach().item()
            )
            response_mask = response_mask * token_is_ratio_mask

        if self.pipeline_config.token_filter.infer_diff_mask:
            token_infer_diff_mask = (
                train_infer_diff <= self.pipeline_config.token_filter.infer_diff_clamp_high
            ).float() * (train_infer_diff >= self.pipeline_config.token_filter.infer_diff_clamp_low).float()
            train_infer_metric["actor/train_infer_token_infer_diff_mask_mean"] = (
                masked_mean(token_infer_diff_mask, response_mask, dim=-1).mean().detach().item()
            )
            response_mask = response_mask * token_infer_diff_mask

        if self.pipeline_config.segment_filter.is_mask:
            segment_infer_ratio_mask = (
                train_infer_ratio_segment <= self.pipeline_config.segment_filter.is_clamp_high
            ).float() * (train_infer_ratio_segment >= self.pipeline_config.segment_filter.is_clamp_low).float()
            train_infer_metric["actor/train_infer_segment_infer_ratio_mask_mean"] = (
                masked_mean(segment_infer_ratio_mask, response_mask, dim=-1).mean().detach().item()
            )
            response_mask = response_mask * segment_infer_ratio_mask

        if self.pipeline_config.segment_filter.infer_diff_mask:
            segment_infer_diff_seq_mask = (
                train_infer_diff_segment <= self.pipeline_config.segment_filter.infer_diff_clamp_high
            ).float() * (train_infer_diff_segment >= self.pipeline_config.segment_filter.infer_diff_clamp_low).float()
            train_infer_metric["actor/train_infer_segment_infer_diff_seq_mask_mean"] = (
                masked_mean(segment_infer_diff_seq_mask, response_mask, dim=-1).mean().detach().item()
            )
            response_mask = response_mask * segment_infer_diff_seq_mask

        pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

        kl_loss = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
        )
        kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode)

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )
        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = pg_loss
        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(
                logits=output_tensor, attention_mask=data.batch["response_mask"]
            )
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        # train infer data filter
        train_infer_metric.update(
            {
                "actor/train_infer_ratio_mean": masked_mean(train_infer_ratio, response_mask, dim=-1)
                .mean()
                .detach()
                .item(),
                "actor/train_infer_ratio_segment_mean": masked_mean(train_infer_ratio_segment, response_mask, dim=-1)
                .mean()
                .detach()
                .item(),
                "actor/train_infer_diff_mean": masked_mean(train_infer_diff, response_mask, dim=-1)
                .mean()
                .detach()
                .item(),
            }
        )

        pg_metrics = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": agg_loss(
                loss_mat=torch.lt(surr2, surr1).float(),
                loss_mask=response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
            )
            .detach()
            .item(),
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": agg_loss(
                loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode
            )
            .detach()
            .item(),
            "actor/policykl": agg_loss(
                loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode
            )
            .detach()
            .item(),
            **train_infer_metric,
        }

        return total_loss, pg_metrics
