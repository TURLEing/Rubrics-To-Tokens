import os
from abc import ABC

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.utils.distributed_sampler import DistributedSampler


class TokenLevelValueTrainer(ABC):
    """
    Trainer for training a token-level value model.

    This trainer trains a model to output binary labels (0/1) for each token in the response,
    indicating whether the token is relevant to the prompt requirement.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        disable_ds_ckpt=False,
        save_hf_ckpt=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.disable_ds_ckpt = disable_ds_ckpt
        self.save_hf_ckpt = save_hf_ckpt

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.compute_fp32_loss = getattr(self.strategy.args, "compute_fp32_loss", False)

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def compute_loss(self, logits, token_labels, response_mask):
        """
        Compute token-level binary cross-entropy loss.

        Args:
            logits: (batch_size, seq_len) - Model output logits
            token_labels: (batch_size, seq_len) - Ground truth labels, -1 for ignored positions
            response_mask: (batch_size, seq_len) - 1 for response tokens, 0 for prompt/padding

        Returns:
            loss: Scalar loss value
        """
        # Only compute loss for response tokens (response_mask > 0) with valid labels (token_labels >= 0)
        valid_mask = (token_labels >= 0) & (response_mask > 0)

        if valid_mask.sum() == 0:
            # No valid tokens, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Extract valid logits and labels
        valid_logits = logits[valid_mask]
        valid_labels = token_labels[valid_mask].float()

        if self.compute_fp32_loss:
            valid_logits = valid_logits.float()

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, reduction="mean")

        return loss

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        loss_sum = 0
        acc_sum = 0
        print_samples = getattr(args, "print_samples", True)  # Default to True, can be disabled via args
        print_sample_steps = getattr(args, "print_sample_steps", [0, 1, 10, 100])  # Steps to print samples
        
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            # train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                input_ids, attention_mask, token_labels, response_mask, response_start_indices = data
                input_ids = input_ids.to(torch.cuda.current_device())
                attention_mask = attention_mask.to(torch.cuda.current_device())
                token_labels = token_labels.to(torch.cuda.current_device())
                response_mask = response_mask.to(torch.cuda.current_device())
                
                # Print sample data for debugging (only on rank 0 and at specified steps)
                if print_samples and self.strategy.is_rank_0() and step in print_sample_steps:
                    self._print_sample_data(
                        input_ids, attention_mask, token_labels, response_mask, 
                        response_start_indices, step, epoch
                    )

                # Forward pass
                logits, output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )
                
                # Print sample data with predictions (after forward pass)
                if print_samples and self.strategy.is_rank_0() and step in print_sample_steps:
                    self._print_sample_data_with_logits(
                        input_ids, attention_mask, token_labels, response_mask, 
                        response_start_indices, logits, step, epoch
                    )

                # Compute loss
                loss = self.compute_loss(logits, token_labels, response_mask)

                # Auxiliary loss (e.g., MoE balancing loss)
                aux_loss = 0
                if self.aux_loss and "aux_loss" in output:
                    aux_loss = output.aux_loss
                    loss = loss + aux_loss * self.args.aux_loss_coef

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Compute accuracy (only for response tokens)
                with torch.no_grad():
                    valid_mask = (token_labels >= 0) & (response_mask > 0)
                    if valid_mask.sum() > 0:
                        preds = (torch.sigmoid(logits[valid_mask]) > 0.5).long()
                        targets = token_labels[valid_mask].long()
                        acc = (preds == targets).float().mean().item()
                    else:
                        acc = 0.0

                acc_sum += acc
                loss_sum += loss.item()

                # optional info
                logs_dict = {
                    "loss": loss.item(),
                    "acc": acc,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if (
            global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            loss_sum = 0
            total_valid_tokens = 0
            for data in eval_dataloader:
                input_ids, attention_mask, token_labels, response_mask, response_start_indices = data
                input_ids = input_ids.to(torch.cuda.current_device())
                attention_mask = attention_mask.to(torch.cuda.current_device())
                token_labels = token_labels.to(torch.cuda.current_device())
                response_mask = response_mask.to(torch.cuda.current_device())

                # Forward pass
                logits, output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )
                
                # Print sample data with model predictions for debugging (only on rank 0 and at specified steps)
                if getattr(self, "print_samples", False) and self.strategy.is_rank_0():
                    # Get print_sample_steps from args or default
                    print_sample_steps = getattr(self, "print_sample_steps", [0, 100, 500, 1000])
                    # step_bar.n + 1 为当前eval真实step号, 这里step是for循环index
                    real_eval_step = getattr(self, "current_eval_step", steps)
                    if real_eval_step in print_sample_steps:
                        if hasattr(self, "_print_sample_data_with_logits"):
                            self._print_sample_data_with_logits(
                                input_ids, attention_mask, token_labels, response_mask, 
                                response_start_indices, logits, real_eval_step, getattr(self, "current_epoch", 0)
                            )

                # Compute loss
                loss = self.compute_loss(logits, token_labels, response_mask)
                loss_sum += loss.item()

                # Compute accuracy
                valid_mask = (token_labels >= 0) & (response_mask > 0)
                if valid_mask.sum() > 0:
                    preds = (torch.sigmoid(logits[valid_mask]) > 0.5).long()
                    targets = token_labels[valid_mask].long()
                    batch_acc = (preds == targets).float().mean().item()
                    batch_tokens = valid_mask.sum().item()
                    acc += batch_acc * batch_tokens
                    total_valid_tokens += batch_tokens

                step_bar.update()

            acc_mean = acc / total_valid_tokens if total_valid_tokens > 0 else 0.0
            loss_mean = loss_sum / eval_dataloader.__len__()

            bar_dict = {
                "eval_loss": loss_mean,
                "eval_acc": acc_mean,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def _print_sample_data(
        self, input_ids, attention_mask, token_labels, response_mask, 
        response_start_indices, step, epoch
    ):
        """Print sample data before forward pass for debugging."""
        print(f"\n{'='*80}")
        print(f"训练样例 - Epoch {epoch}, Step {step}")
        print(f"{'='*80}")
        
        batch_size = input_ids.shape[0]
        # Print first sample in the batch
        num_samples = min(1, batch_size)
        
        for i in range(num_samples):
            print(f"\n--- 样例 {i+1}/{batch_size} ---")
            
            # Decode input_ids to text
            sample_input_ids = input_ids[i].cpu()
            sample_attention_mask = attention_mask[i].cpu()
            # Remove padding tokens
            valid_length = sample_attention_mask.sum().item()
            sample_input_ids = sample_input_ids[:valid_length]
            
            # Decode full text
            try:
                full_text = self.tokenizer.decode(sample_input_ids, skip_special_tokens=False)
                
                # Try to split prompt and response
                response_start_idx = response_start_indices[i] if i < len(response_start_indices) else 0
                if response_start_idx > 0 and response_start_idx < len(sample_input_ids):
                    prompt_ids = sample_input_ids[:response_start_idx]
                    response_ids = sample_input_ids[response_start_idx:]
                    prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                    
                    print(f"\n【Prompt】")
                    print(f"{prompt_text}")
                    print(f"\n【Response】")
                    print(f"{response_text}")
                else:
                    print(f"\n【完整文本】")
                    print(f"{full_text[:500]}..." if len(full_text) > 500 else f"{full_text}")
            except Exception as e:
                print(f"解码文本失败: {e}")
            
            # Show token labels and response mask
            sample_token_labels = token_labels[i].cpu()[:valid_length]
            sample_response_mask = response_mask[i].cpu()[:valid_length]
            response_start_idx = response_start_indices[i] if i < len(response_start_indices) else 0
            
            print(f"\n【统计信息】")
            print(f"  序列长度: {valid_length}")
            print(f"  Response起始位置: {response_start_idx}")
            print(f"  Response token数量: {sample_response_mask.sum().item()}")
            
            # Show token-level information for response tokens
            response_tokens = sample_response_mask > 0
            if response_tokens.sum() > 0:
                response_labels = sample_token_labels[response_tokens]
                valid_labels = response_labels[response_labels >= 0]
                
                if len(valid_labels) > 0:
                    label_counts = torch.bincount(valid_labels.long(), minlength=2)
                    print(f"  Label=0 (不相关) token数: {label_counts[0].item()}")
                    print(f"  Label=1 (相关) token数: {label_counts[1].item() if len(label_counts) > 1 else 0}")
                
                # Show visualization of labels for response tokens
                response_indices = torch.where(response_tokens)[0]
                if len(response_indices) > 0:
                    print(f"\n【Token标签可视化】(前50个response tokens)")
                    print(f"  位置 | Token | 标签")
                    print(f"  {'-'*60}")
                    for j, idx in enumerate(response_indices[:50]):
                        token_id = sample_input_ids[idx].item()
                        label = sample_token_labels[idx].item()
                        try:
                            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            # Clean up token string
                            token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')
                            if len(token_str) > 20:
                                token_str = token_str[:17] + "..."
                        except:
                            token_str = f"<id:{token_id}>"
                        
                        label_str = "✓" if label == 1 else ("✗" if label == 0 else "?")
                        print(f"  {idx:4d} | {token_str:20s} | {label_str} ({label})")
        
        print(f"\n{'='*80}\n")

    def _print_sample_data_with_logits(
        self, input_ids, attention_mask, token_labels, response_mask, 
        response_start_indices, logits, step, epoch
    ):
        """Print sample data with model predictions after forward pass."""
        print(f"\n{'='*80}")
        print(f"模型预测样例 - Epoch {epoch}, Step {step}")
        print(f"{'='*80}")
        
        batch_size = input_ids.shape[0]
        # Print first sample in the batch
        num_samples = min(1, batch_size)
        
        for i in range(num_samples):
            print(f"\n--- 样例 {i+1}/{batch_size} ---")
            
            # Decode input_ids to text
            sample_input_ids = input_ids[i].cpu()
            sample_attention_mask = attention_mask[i].cpu()
            sample_logits = logits[i].cpu()
            # Remove padding tokens
            valid_length = sample_attention_mask.sum().item()
            sample_input_ids = sample_input_ids[:valid_length]
            sample_logits = sample_logits[:valid_length]
            
            # Show token labels and response mask
            sample_token_labels = token_labels[i].cpu()[:valid_length]
            sample_response_mask = response_mask[i].cpu()[:valid_length]
            response_start_idx = response_start_indices[i] if i < len(response_start_indices) else 0
            
            # Show token-level information for response tokens
            response_tokens = sample_response_mask > 0
            if response_tokens.sum() > 0:
                response_labels = sample_token_labels[response_tokens]
                response_logits = sample_logits[response_tokens]
                response_probs = torch.sigmoid(response_logits)
                response_preds = (response_probs > 0.5).long()
                
                print(f"\n【预测统计】")
                print(f"  Response token总数: {response_labels.shape[0]}")
                
                valid_mask = response_labels >= 0
                if valid_mask.sum() > 0:
                    valid_labels = response_labels[valid_mask]
                    valid_preds = response_preds[valid_mask]
                    valid_probs = response_probs[valid_mask]
                    
                    label_counts = torch.bincount(valid_labels.long(), minlength=2)
                    pred_counts = torch.bincount(valid_preds.long(), minlength=2)
                    
                    print(f"  真实标签分布: Label=0: {label_counts[0].item()}, Label=1: {label_counts[1].item() if len(label_counts) > 1 else 0}")
                    print(f"  预测分布: Pred=0: {pred_counts[0].item()}, Pred=1: {pred_counts[1].item() if len(pred_counts) > 1 else 0}")
                    
                    # Calculate accuracy
                    acc = (valid_labels == valid_preds).float().mean().item()
                    print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
                    
                    # Show some tokens with their labels and predictions
                    response_indices = torch.where(response_tokens)[0][:30]
                    print(f"\n【Token预测详情】(前30个response tokens)")
                    print(f"  {'位置':<6} {'Token':<25} {'标签':<6} {'Logit':<10} {'概率':<8} {'预测':<6} {'匹配':<6}")
                    print(f"  {'-'*80}")
                    for idx in response_indices:
                        token_id = sample_input_ids[idx].item()
                        label = sample_token_labels[idx].item()
                        logit_val = sample_logits[idx].item()
                        prob_val = torch.sigmoid(torch.tensor(logit_val)).item()
                        pred_val = 1 if prob_val > 0.5 else 0
                        
                        try:
                            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                            token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')
                            if len(token_str) > 20:
                                token_str = token_str[:17] + "..."
                            token_str = repr(token_str)
                        except:
                            token_str = f"<id:{token_id}>"
                        
                        label_str = str(label) if label >= 0 else "N/A"
                        match_str = "✓" if label >= 0 and label == pred_val else ("✗" if label >= 0 else "?")
                        print(f"  {idx:<6} {token_str:<25} {label_str:<6} {logit_val:<10.4f} {prob_val:<8.4f} {pred_val:<6} {match_str:<6}")
        
        print(f"\n{'='*80}\n")

