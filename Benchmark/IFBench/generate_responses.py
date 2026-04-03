#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate responses for IFEval dataset using vLLM API.
Reads input_data.jsonl and generates responses using vLLM's OpenAI-compatible API,
then saves to input_response_data_{model}.jsonl

Usage example:
    # Start vLLM server first:
    vllm serve /home/xutianze.xtz/rubrics_to_tokens/LlamaFactory/saves/qwen3-4b-instruct/sft --port 8000 \
    --served-model-name Qwen/Qwen3-4B-Instruct-2507
    
    python generate_responses.py \
        --input_data data/IFBench_test.jsonl \
        --api_base http://localhost:8000/v1 \
        --model_name Qwen/Qwen3-4B-Instruct-2507 \
        --max_concurrent 20 
"""

import argparse
import asyncio
import json
import os
from datetime import datetime

from openai import AsyncOpenAI
from tqdm import tqdm


async def generate_response_async(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = -1
) -> str:
    """
    Generate response using OpenAI-compatible API (vLLM).
    
    Args:
        client: AsyncOpenAI client instance
        model_name: Model name to use
        prompt: Input prompt
        max_token: max token
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter (-1 means no limit)
    
    Returns:
        Generated response text
    """
    try:
        create_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        if top_k > 0:
            create_kwargs["extra_body"] = {"top_k": top_k}
        response = await client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")


async def process_dataset(
    input_file: str,
    output_file: str,
    api_base: str,
    model_name: str,
    max_concurrent: int = 10,
    max_tokens: int = 4096,
    temperature: float = 0,
    top_p: float = 1,
    top_k: int = -1
):
    """
    Process the dataset and generate responses using vLLM API.
    
    Args:
        input_file: Path to input_data.jsonl
        output_file: Path to output input_response_data_{model}.jsonl
        api_base: Base URL for vLLM API (e.g., http://localhost:8000/v1)
        model_name: Model name to use
        max_concurrent: Maximum number of concurrent requests
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    """
    # Load input data
    print(f"Loading input data from {input_file}...")
    inputs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))
    
    print(f"Loaded {len(inputs)} prompts")
    
    # Initialize OpenAI client
    print(f"Connecting to API at {api_base}...")
    client = AsyncOpenAI(
        base_url=api_base,
        api_key="not-needed"  # vLLM doesn't require API key
    )
    
    # Test connection
    try:
        await client.models.list()
        print("Successfully connected to API")
    except Exception as e:
        print(f"Warning: Could not verify API connection: {e}")
        print("Continuing anyway...")
    
    # Process prompts asynchronously
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(prompt_data):
        async with semaphore:
            prompt = prompt_data["prompt"]
            try:
                response = await generate_response_async(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                return {
                    "prompt": prompt,
                    "response": response
                }
            except Exception as e:
                print(f"Error processing prompt: {e}")
                return {
                    "prompt": prompt,
                    "response": f"Error: {str(e)}"
                }
    
    # Process all prompts with progress bar
    print("Generating responses...")
    
    # Create tasks with indices to preserve order
    async def process_with_index(idx, prompt_data):
        result = await process_single(prompt_data)
        return idx, result
    
    tasks = [process_with_index(i, inp) for i, inp in enumerate(inputs)]
    results_dict = {}
    
    # Process with progress bar, maintaining order
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        idx, result = await coro
        results_dict[idx] = result
    
    # Reconstruct results in original order
    results = [results_dict[i] for i in range(len(inputs))]
    
    # Save results
    print(f"Saving results to {output_file}...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Generated {len(results)} responses")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses for IFEval dataset using vLLM API"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input_data.jsonl"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for vLLM API (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use (as registered in vLLM server)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output file (default: input_response_data_{model}.jsonl in same dir as input_data)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent API requests (default: 10)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (default: -1, means no limit)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input file not found: {args.input_data}")
    
    # Determine output file path
    if args.output_file is None:
        input_dir = os.path.dirname(os.path.abspath(args.input_data))
        model_name = args.model_name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            input_dir,
            f"input_response_data_{model_name}_{timestamp}.jsonl"
        )
    else:
        output_file = os.path.abspath(args.output_file)
    
    # Run async processing
    asyncio.run(
        process_dataset(
            input_file=os.path.abspath(args.input_data),
            output_file=output_file,
            api_base=args.api_base,
            model_name=args.model_name,
            max_concurrent=args.max_concurrent,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
    )


if __name__ == "__main__":
    main()

