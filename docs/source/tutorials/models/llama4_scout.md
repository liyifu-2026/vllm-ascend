# Llama-4-Scout-17B-16E-Instruct

This tutorial provides instructions on how to deploy and run the `Llama-4-Scout-17B-16E-Instruct` model using vLLM on Huawei Ascend NPU hardware.

## Run Docker Container

:::{note}
Running this 17B-16E MoE model requires at least 4 NPUs (TP=4) to accommodate the weights and KV cache.
:::

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=16g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /data/models:/data/models \
-v /root/.cache:/root/.cache \
-p 8188:8188 \
-it $IMAGE bash
```

## Launch vLLM Server

You can start the vLLM server with the following command. Note that **Tensor Parallel (TP) size 4** is required for this model on Atlas A2.

```bash
export MODEL_PATH=/data/models/Llama-4-Scout-17B-16E-Instruct

python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --enforce-eager \
    --trust-remote-code \
    --port 8188
```

## Verify the Model

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8188/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data/models/Llama-4-Scout-17B-16E-Instruct",
        "messages": [{"role": "user", "content": "Explain Mixture-of-Experts in one sentence."}],
        "max_tokens": 64,
        "temperature": 0.0
    }'
```

## Offline Inference

You can also execute offline inference using the `LLM` class. Ensure `tensor_parallel_size` is set to 4.

```python
from vllm import LLM, SamplingParams

prompts = [
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

# Initialize the model with TP=4
llm = LLM(
    model="/data/models/Llama-4-Scout-17B-16E-Instruct",
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=2048,
    enforce_eager=True,
    trust_remote_code=True
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
