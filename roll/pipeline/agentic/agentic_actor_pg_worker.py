import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl
from roll.pipeline.agentic.utils import compute_segment_masked_mean
from roll.pipeline.agentic.agentic_pipeline import get_episode_scores
from roll.platforms import current_platform


class ActorWorker(BaseActorWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 缓存PG变体的配置参数
        self._pg_config_cache = {}
        self._pg_variant_logged = True
        self._topr_sample_logged = False
        self._cispo_config_logged = False
        self._kimi15_config_logged = False

    def _get_or_cache_config(self, key, default_value):
        """获取或缓存配置值"""
        if key not in self._pg_config_cache:
            self._pg_config_cache[key] = getattr(self.pipeline_config.actor_train, key, default_value)
        return self._pg_config_cache[key]

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

        pg_variant = self._get_or_cache_config("pg_variant", "vanilla")
        self._cached_metrics = {
            "pg_variant": pg_variant,
            "ratio": ratio,
            "response_mask": response_mask,
        }

        if pg_variant == "vanilla":  # Basic Policy Gradient
            pg_loss = self._compute_vanilla_pg_loss(ratio, log_probs, advantages)
        elif pg_variant == "ppo":  # Proximal Policy Optimization
            pg_loss = self._compute_ppo_loss(ratio, advantages, response_mask)
        elif pg_variant == "tis":  # Truncated Importance Sampling
            pg_loss = self._compute_tis_loss(ratio, log_probs, old_log_probs, advantages, data)
        elif pg_variant == "topr":  # Tapered off-policy REINFORCE
            pg_loss = self._compute_topr_loss(ratio, log_probs, old_log_probs, advantages, data)
        elif pg_variant == "cispo":  # Clipped Importance Sampling Policy Optimization    Minimax-M1
            pg_loss = self._compute_cispo_loss(ratio, log_probs, advantages)
        elif pg_variant == "kimi15":  # Kimi15
            pg_loss = self._compute_kimi15_loss(ratio, log_probs, old_log_probs, advantages)
        else:
            raise ValueError(f"Unsupported pg_variant: {pg_variant}")

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
        # 缓存损失相关指标
        self._cached_metrics.update({"original_pg_loss": pg_loss})

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

        self._cached_metrics.update(
            {
                "kl_loss": kl_loss,
                "approxkl": approxkl,
                "policykl": policykl,
                "train_infer_ratio_mean": masked_mean(train_infer_ratio, response_mask, dim=-1).mean().detach().item(),
                "train_infer_diff_mean": masked_mean(train_infer_diff, response_mask, dim=-1).mean().detach().item(),
            }
        )

        self._cached_metrics["total_loss"] = total_loss

        # 使用缓存的指标
        pg_metrics = self._get_pg_metrics(data)
        pg_metrics.update(train_infer_metric)
        return total_loss, pg_metrics

    def _compute_ppo_loss(self, ratio: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor):
        """
        计算PPO损失
        """
        pg_clip = self.pipeline_config.pg_clip
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
        loss = -torch.min(surr1, surr2)
        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-loss, (1 + pg_clip * 2) * advantages)
            loss = torch.where(advantages < 0, dual_clip_loss, loss)

        # 缓存PPO相关指标
        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        self._cached_metrics.update(
            {
                "ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
                "ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
                "ppo_ratio_clipfrac": clipped.mean().detach().item(),
                "clipfrac": agg_loss(
                    loss_mat=torch.lt(surr2, surr1).float(),
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                )
                .detach()
                .item(),
            }
        )

        return loss

    def _compute_vanilla_pg_loss(self, ratio: torch.Tensor, log_probs: torch.Tensor, advantages: torch.Tensor):
        """
        计算原始Policy Gradient损失

        Args:
            ratio: 重要性采样比率 π(a|s) / π_old(a|s)
            advantages: 优势函数值

        Returns:
            pg_loss: Policy Gradient损失
        """

        return -log_probs * advantages

    def _compute_tis_loss(
        self,
        ratio: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        data: DataProto,
    ):
        """
        计算Truncated Importance Sampling (TIS) 损失
        根据论文: Truncated Importance Sampling for Value-based Reinforcement Learning
        TIS将重要性采样比率截断在[0, 1]范围内
        """
        # 缓存TIS配置
        tis_lower_bound = self._get_or_cache_config("tis_lower_bound", 0.0)
        tis_upper_bound = self._get_or_cache_config("tis_upper_bound", 1.0)

        # 截断重要性采样比率
        clipped_ratio = torch.clamp(ratio, min=tis_lower_bound, max=tis_upper_bound)

        TIS_loss = -clipped_ratio.detach() * advantages * log_probs

        # 缓存TIS相关指标
        lower_clipped = (ratio < tis_lower_bound).float()
        upper_clipped = (ratio > tis_upper_bound).float()
        total_clipped = (lower_clipped + upper_clipped).float()

        self._cached_metrics.update(
            {
                "tis_lower_bound": tis_lower_bound,
                "tis_upper_bound": tis_upper_bound,
                "tis_lower_clipfrac": lower_clipped.mean().detach().item(),
                "tis_upper_clipfrac": upper_clipped.mean().detach().item(),
                "tis_total_clipfrac": total_clipped.mean().detach().item(),
                "tis_clipped_ratio": clipped_ratio.detach(),
            }
        )

        return TIS_loss

    def _compute_topr_loss(
        self,
        ratio: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        data: DataProto,
    ):
        """
        计算TOPR (Tapered off-policy REINFORCE) 损失. https://arxiv.org/abs/2503.14286

        根据论文公式(8):
        ∇J_TOPR(π) = Σ_{τ∈T^+} μ(τ)R(τ)∇log π(τ) + Σ_{τ∈T^-} μ(τ)[π(τ)/μ(τ)]_0^1 R(τ)∇log π(τ)

        - 正样本(T^+): SFT更新, 直接对log π(τ)求导, 不使用importance sampling
        - 负样本(T^-): TIS更新, 使用clipped importance sampling ratio [0,1]

        Args:
            ratio: 重要性采样比率 π(a|s) / π_old(a|s) [batch_size, seq_len]
            log_probs: 当前策略的log概率 [batch_size, seq_len]
            old_log_probs: 旧策略的log概率 [batch_size, seq_len]
            advantages: 优势函数值 [batch_size, seq_len]
            data: 数据，包含奖励/分数信息

        Returns:
            topr_loss: TOPR损失 [batch_size, seq_len]
        """
        # 缓存TOPR配置
        positive_weight = self._get_or_cache_config("topr_positive_weight", 1.0)
        negative_weight = self._get_or_cache_config("topr_negative_weight", 1.0)

        # scores = data.batch['scores']dim=@).to(current_platform.device_type)
        scores = get_episode_scores(data).to(current_platform.device_type)
        self.logger.info(
            f"shape!!!!!! - scores: {scores.shape}, log_probs: {log_probs.shape}, advantages: {advantages.shape}"
        )
        positive_mask = (scores > 0).float()
        negative_mask = (scores <= 0).float()

        if not self._topr_sample_logged:
            total_samples = len(scores)
            positive_count = positive_mask.sum().item()
            negative_count = negative_mask.sum().item()
            self.logger.info(
                f"TOPR样本分布 - 总样本: {total_samples}, 正样本: {positive_count} ({positive_count/total_samples*100:.1f}%), 负样本: {negative_count} ({negative_count/total_samples*100:.1f}%)"
            )
            self.logger.info(
                f"TOPR奖励统计 - 平均: {scores.mean().item():.4f}, 标准差: {scores.std().item():.4f}, 最大: {scores.max().item():.4f}, 最小: {scores.min().item():.4f}"
            )
            self.logger.info(f"TOPR权重配置 - 正样本权重: {positive_weight}, 负样本权重: {negative_weight}")
            self._topr_sample_logged = True

        # 计算损失组件
        positive_token_mask = positive_mask.unsqueeze(-1).expand_as(log_probs)
        negative_token_mask = negative_mask.unsqueeze(-1).expand_as(log_probs)

        positive_loss = -advantages * log_probs * positive_token_mask

        # 负样本: TIS更新，使用clipped importance sampling ratio
        # 梯度是: -[π(τ)/μ(τ)]_0^1 * R(τ) * ∇log π(τ)
        clipped_ratio = torch.clamp(ratio, min=0.0, max=1.0).detach()
        negative_loss = -clipped_ratio * advantages * log_probs * negative_token_mask

        weighted_positive_loss = positive_weight * positive_loss
        weighted_negative_loss = negative_weight * negative_loss

        topr_loss = weighted_positive_loss + weighted_negative_loss

        # 缓存TOPR相关指标
        negative_lower_clipped = ((ratio < 0.0) & (negative_token_mask > 0)).float()
        negative_upper_clipped = ((ratio > 1.0) & (negative_token_mask > 0)).float()
        negative_total_clipped = negative_lower_clipped + negative_upper_clipped
        self._cached_metrics.update(
            {
                "topr_positive_loss": positive_loss,
                "topr_negative_loss": negative_loss,
                "topr_weighted_positive_loss": weighted_positive_loss,
                "topr_weighted_negative_loss": weighted_negative_loss,
                "topr_positive_weight": positive_weight,
                "topr_negative_weight": negative_weight,
                "topr_positive_samples": positive_mask.sum().detach().item(),
                "topr_negative_samples": negative_mask.sum().detach().item(),
                "topr_positive_ratio": (positive_mask.sum() / (positive_mask.size(0) + 1e-8)).detach().item(),
                "topr_negative_ratio": (negative_mask.sum() / (negative_mask.size(0) + 1e-8)).detach().item(),
                "topr_negative_lower_clipfrac": negative_lower_clipped.mean().detach().item(),
                "topr_negative_upper_clipfrac": negative_upper_clipped.mean().detach().item(),
                "topr_negative_total_clipfrac": negative_total_clipped.mean().detach().item(),
                "topr_scores_mean": scores.mean().detach().item(),
                "topr_scores_std": scores.std().detach().item(),
            }
        )

        return topr_loss

    def _compute_cispo_loss(self, ratio: torch.Tensor, log_probs: torch.Tensor, advantages: torch.Tensor):
        """
        计算CISPO (Clipped Importance Sampling Policy Optimization) 损失

        根据论文: https://arxiv.org/abs/2503.14286
        CISPO使用截断的重要性采样权重, 同时使用stop-gradient操作来稳定训练

        公式: J_CISPO(θ) = E[sg(r̂_t(θ)) * Â_t * log π_θ(a_t|s_t)]
        其中: r̂_t(θ) = clip(r_t(θ), 1-ε_low^IS, 1+ε_high^IS)

        Args:
            ratio: 重要性采样比率 π(a|s) / π_old(a|s) [batch_size, seq_len]
            log_probs: 当前策略的log概率 [batch_size, seq_len]
            advantages: 优势函数值 [batch_size, seq_len]

        Returns:
            cispo_loss: CISPO损失 [batch_size, seq_len]
        """
        # 缓存CISPO配置
        epsilon_low = self._get_or_cache_config("cispo_epsilon_low", 0.1)
        epsilon_high = self._get_or_cache_config("cispo_epsilon_high", 0.1)
        use_unified_mask = self._get_or_cache_config("cispo_use_unified_mask", False)

        clip_lower = 1.0 - epsilon_low
        clip_upper = 1.0 + epsilon_high

        if not self._cispo_config_logged:
            self.logger.info(f"CISPO配置 - epsilon_low: {epsilon_low}, epsilon_high: {epsilon_high}")
            self.logger.info(f"CISPO截断范围: [{clip_lower:.3f}, {clip_upper:.3f}]")
            self.logger.info(f"CISPO使用统一mask: {use_unified_mask}")
            self._cispo_config_logged = True

        clipped_ratio = torch.clamp(ratio, min=clip_lower, max=clip_upper)

        # 缓存CISPO相关指标
        lower_clipped = (ratio < clip_lower).float()
        upper_clipped = (ratio > clip_upper).float()
        total_clipped = (lower_clipped + upper_clipped).float()

        if use_unified_mask:
            # 使用统一mask公式 (论文公式7). 实际上应该和PPO一致了
            # M_t = 0 if (A_t > 0 and r_t > 1+ε_high) or (A_t < 0 and r_t < 1-ε_low), else 1
            positive_advantages = advantages > 0
            negative_advantages = advantages < 0

            mask_positive = positive_advantages & (ratio > clip_upper)
            mask_negative = negative_advantages & (ratio < clip_lower)
            token_mask = ~(mask_positive | mask_negative)

            cispo_loss = -clipped_ratio.detach() * advantages * log_probs * token_mask.float()
        else:
            cispo_loss = -clipped_ratio.detach() * advantages * log_probs

        cispo_metrics = {
            "cispo_epsilon_low": epsilon_low,
            "cispo_epsilon_high": epsilon_high,
            "cispo_clip_lower": clip_lower,
            "cispo_clip_upper": clip_upper,
            "cispo_use_unified_mask": float(use_unified_mask),
            "cispo_lower_clipfrac": lower_clipped.mean().detach().item(),
            "cispo_upper_clipfrac": upper_clipped.mean().detach().item(),
            "cispo_total_clipfrac": total_clipped.mean().detach().item(),
            "cispo_clipped_ratio": clipped_ratio.detach(),
        }
        if use_unified_mask:
            cispo_metrics.update(
                {
                    "cispo_masked_positive_tokens": mask_positive.float().mean().detach().item(),
                    "cispo_masked_negative_tokens": mask_negative.float().mean().detach().item(),
                    "cispo_kept_tokens": token_mask.float().mean().detach().item(),
                }
            )

        self._cached_metrics.update(cispo_metrics)
        return cispo_loss

    def _compute_kimi15_loss(
        self, ratio: torch.Tensor, log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor
    ):
        """
        计算Kimi15损失 https://arxiv.org/pdf/2501.12599

        根据论文公式(3):
        1/k Σ (∇_θ log π_θ(y_j, z_j|x)(r(x, y_j, y*) - r̄) - τ/2 ∇_θ (log π_θ(y_j, z_j|x)/π_θ_i(y_j, z_j|x))^2)

        这相当于最小化损失函数的负值:
        L = -[(r - r̄) * log π_θ - τ/2 * (log π_θ/π_θ_i)^2]
        """
        # 缓存Kimi15配置
        tau = self._get_or_cache_config("kimi15_tau", 0.1)

        if not self._kimi15_config_logged:
            self.logger.info(f"Kimi15配置 - tau (正则化参数): {tau}")
            self._kimi15_config_logged = True

        # 计算并缓存指标
        log_ratio = torch.log(ratio + 1e-8)
        policy_grad_magnitude = (advantages * log_ratio).abs().mean().item()
        kl_reg_magnitude = (tau * log_ratio.pow(2) * 0.5).mean().item()

        kimi15_loss = -advantages * log_probs + tau * 0.5 * (log_probs - old_log_probs).pow(2)

        self._cached_metrics.update(
            {
                "kimi15_tau": tau,
                "kimi15_log_ratio_mean": log_ratio.mean().item(),
                "kimi15_log_ratio_std": log_ratio.std().item(),
                "kimi15_log_ratio_abs_mean": log_ratio.abs().mean().item(),
                "kimi15_policy_grad_magnitude": policy_grad_magnitude,
                "kimi15_kl_reg_magnitude": kl_reg_magnitude,
                "kimi15_reg_ratio": kl_reg_magnitude / (policy_grad_magnitude + 1e-8),
            }
        )

        return kimi15_loss

    def _get_pg_metrics(self, data: DataProto):
        """
        获取Policy Gradient相关的指标，使用缓存的值避免重复计算
        """
        # 从缓存中获取基础值
        cached = self._cached_metrics
        ratio = cached["ratio"]
        response_mask = cached["response_mask"]

        # 构建基础指标
        base_metrics = {
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/pg_loss": cached["original_pg_loss"].detach().item(),
            "actor/kl_loss": cached["kl_loss"].detach().item(),
            "actor/total_loss": cached["total_loss"].detach().item(),
            "actor/approxkl": agg_loss(
                loss_mat=cached["approxkl"], loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode
            )
            .detach()
            .item(),
            "actor/policykl": agg_loss(
                loss_mat=cached["policykl"], loss_mask=response_mask, loss_agg_mode=self.pipeline_config.loss_agg_mode
            )
            .detach()
            .item(),
            "actor/train_infer_ratio_mean": cached["train_infer_ratio_mean"],
            "actor/train_infer_diff_mean": cached["train_infer_diff_mean"],
        }

        # 根据PG变体添加特定指标
        pg_variant = cached["pg_variant"]

        if pg_variant == "ppo":
            ppo_metrics = {
                "actor/ppo_ratio_high_clipfrac": cached["ppo_ratio_high_clipfrac"],
                "actor/ppo_ratio_low_clipfrac": cached["ppo_ratio_low_clipfrac"],
                "actor/ppo_ratio_clipfrac": cached["ppo_ratio_clipfrac"],
            }
            base_metrics.update(ppo_metrics)

        elif pg_variant == "tis":
            tis_metrics = {
                "actor/tis_lower_clipfrac": cached["tis_lower_clipfrac"],
                "actor/tis_upper_clipfrac": cached["tis_upper_clipfrac"],
                "actor/tis_total_clipfrac": cached["tis_total_clipfrac"],
                "actor/tis_clipped_ratio_mean": masked_mean(cached["tis_clipped_ratio"], response_mask, dim=-1)
                .mean()
                .item(),
                "actor/tis_lower_bound": cached["tis_lower_bound"],
                "actor/tis_upper_bound": cached["tis_upper_bound"],
            }
            base_metrics.update(tis_metrics)

        elif pg_variant == "topr":
            # 计算TOPR损失组件的聚合指标
            topr_loss_metrics = {
                "actor/topr_positive_loss": agg_loss(
                    loss_mat=cached["topr_positive_loss"],
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                ).item(),
                "actor/topr_negative_loss": agg_loss(
                    loss_mat=cached["topr_negative_loss"],
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                ).item(),
                "actor/topr_weighted_positive_loss": agg_loss(
                    loss_mat=cached["topr_weighted_positive_loss"],
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                ).item(),
                "actor/topr_weighted_negative_loss": agg_loss(
                    loss_mat=cached["topr_weighted_negative_loss"],
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                ).item(),
            }

            topr_metrics = {
                "actor/topr_positive_samples": cached["topr_positive_samples"],
                "actor/topr_negative_samples": cached["topr_negative_samples"],
                "actor/topr_positive_ratio": cached["topr_positive_ratio"],
                "actor/topr_negative_ratio": cached["topr_negative_ratio"],
                "actor/topr_negative_lower_clipfrac": cached["topr_negative_lower_clipfrac"],
                "actor/topr_negative_upper_clipfrac": cached["topr_negative_upper_clipfrac"],
                "actor/topr_negative_total_clipfrac": cached["topr_negative_total_clipfrac"],
                "actor/topr_scores_mean": cached["topr_scores_mean"],
                "actor/topr_scores_std": cached["topr_scores_std"],
                "actor/topr_positive_weight": cached["topr_positive_weight"],
                "actor/topr_negative_weight": cached["topr_negative_weight"],
                **topr_loss_metrics,
            }
            base_metrics.update(topr_metrics)

        elif pg_variant == "cispo":
            cispo_metrics = {
                f"actor/cispo_{key}": value
                for key, value in cached.items()
                if key.startswith("cispo_") and key != "cispo_clipped_ratio"
            }

            # 特殊处理需要计算的指标
            cispo_metrics["actor/cispo_clipped_ratio_mean"] = (
                masked_mean(cached["cispo_clipped_ratio"], response_mask, dim=-1).mean().item()
            )
            base_metrics.update(cispo_metrics)

        elif pg_variant == "kimi15":
            kimi15_metrics = {
                f"actor/kimi15_{key}": value for key, value in cached.items() if key.startswith("kimi15_")
            }
            base_metrics.update(kimi15_metrics)

        return base_metrics
