# llmperf-bench

`llmperf-bench` is a benchmarking tool designed to evaluate the performance of open Large Language Models (LLMs) using Hugging Face Text Generation. This tool is based on a fork of `llmperf` which can be found [here](https://github.com/philschmid/llmperf). Users can define test scenarios using YAML configuration files, which makes it versatile for different testing needs.

## Features

`llmperf-bench` allows you to measure various performance metrics for LLMs, including:

- Throughput: Measures how many tokens can be processed in a given time frame.
- First Time to Token: Tracks the time taken to generate the first token in response to a request.
- Latency (Inter-Token Latency): Measures the time elapsed between generating successive tokens.
- Requests Per Minute: Evaluates the number of requests that can be handled by the model per minute.

These metrics provide a comprehensive view of the performance characteristics of LLMs under various conditions, facilitating better understanding and optimization of model deployments.

## Getting Started

### Prerequisites

- Docker
- Python requirements

You can install the necessary Python packages using:

```bash
pip install -r requirements.txt
```

### Configuration

Create a YAML configuration file to define your test scenarios. An example of the configuration structure is provided below:

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
num_gpus: 1
memory_per_gpu: 24 
tgi:
  max_batch_prefill_tokens: 6144
  max_input_length: 3072
  max_total_tokens: 4096
concurrency: 1,2,4,8,16,32,64,128
num_requests: 100
input_token_length: 750
output_token_length: 150
```

### Running Benchmarks

To run benchmarks, use the provided Python script with the path to your YAML configuration:

```bash
python main.py --config configs/llama3_8b_tp_1.yaml
```

The script will parse the YAML file, start the Docker container, and run the benchmarks. The results will be saved in a JSON file.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
