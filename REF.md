<div align='center'>
<h2>Replay Failures as Successes:<br>Sample-Efficient Reinforcement Learning for Instruction Following</h2>

<!-- TODO:  Thread,Paper,Dataset,Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.23457)
[![Dataset](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/sastpg/HIR-16K)
</div>

## Overview
Reinforcement Learning (RL) has shown promise for aligning Large Language Models (LLMs) to follow instructions with various constraints. Despite the encouraging results, RL improvement inevitably relies on sampling successful, high-quality responses; however, the initial model often struggles to generate responses that satisfy all constraints due to its limited capabilities, yielding sparse or indistinguishable rewards that impede learning. In this work, we propose ***H***indsight ***i***nstruction ***R***eplay (HiR), a novel sample-efficient RL framework for complex instruction following tasks, which employs a *select*-then-*rewrite* strategy to *replay failed attempts as successes* based on the constraints that have been satisfied in hindsight. We perform RL on these replayed samples as well as the original ones, theoretically framing the objective as dual-preference learning at both the instruction- and response-level to enable efficient optimization using only a binary reward signal.

![](./images/framework.png)

## Acknowledgement
We thank the [verl](https://github.com/volcengine/verl) for providing the awesome open-source RL infrastructure. We also thank the developers of [Qwen](https://github.com/QwenLM) and [Llama](https://github.com/meta-llama) for their awesome open-source models.