import argparse
from dataclasses import dataclass, field
import os
from typing import Optional
import yaml
import docker
import requests
import subprocess
import time
import json
from huggingface_hub import HfFolder
import glob 

@dataclass
class TGIConfig:
    max_batch_prefill_tokens: int 
    max_input_length: int
    max_total_tokens: int
    quantize: Optional[bool] = None

@dataclass
class Config:
    model_id: str
    num_gpus: int
    memory_per_gpu: int
    tgi_config: TGIConfig 
    concurrency: list = field(default_factory=list)
    num_requests: int = 100  # Default value if not specified
    input_token_length: int = 500  # Default value if not specified
    output_token_length: int = 200  # Default value if not specified


def fetch_model_config(model_id,num_gpus,memory_per_gpu) -> dict:
    """Retrieve model configuration from an API."""
    api_url = f"https://huggingface.co/api/integrations/tgi/v1/provider/hf/recommend?model_id={model_id}&gpu_memory={memory_per_gpu*num_gpus}&num_gpus={num_gpus}"
    response = requests.get(api_url)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code.
    response = response.json()
    if "error" in response:
        raise ValueError(f"Error fetching model configuration: {response['error']}")
    return response["configuration"]

def parse_yaml_file(filename) -> Config:
    """Load and return the contents of a YAML file as a Config dataclass."""
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        data['concurrency'] = list(map(int, data['concurrency'].split(',')))  # Convert the string to a list of integers
        if 'tgi' in data:
            tgi_conf = TGIConfig(**data.pop('tgi'))
        else:
            tgi_conf = fetch_model_config(data['model_id'], data['num_gpus'], data['memory_per_gpu'])
        return  Config(**data, tgi_config=tgi_conf)



def wait_for_server_to_be_ready(url, timeout=600):
    """Check every 5 seconds if the server is up by calling /health endpoint."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            pass
        print("Waiting for server to be ready...")
        time.sleep(5)
    return False

def run_performance_tests(config):
    """Run the performance script for each concurrency level."""
    results = {}
    detailed_results = {}
    for concurrency in config.concurrency:
        print(f"Running test with concurrency: {concurrency}")
        os.environ["HUGGINGFACE_API_BASE"] = "http://localhost"
        output_dir = f"result_outputs_{concurrency}"
        cmd = [
            "python", "llmperf/token_benchmark_ray.py", 
            "--model", config.model_id,
            "--mean-input-tokens", str(config.input_token_length),
            "--stddev-input-tokens", "0",
            "--mean-output-tokens", str(config.output_token_length),
            "--stddev-output-tokens", "0",
            "--max-num-completed-requests", str(config.num_requests),
            "--timeout", "600",
            "--num-concurrent-requests", str(concurrency),
            "--results-dir", output_dir,
            "--llm-api", "huggingface",
            "--additional-sampling-params", "{}",
        ]
        subprocess.run(cmd)
        with open(glob.glob(f'{output_dir}/*summary.json')[0], 'r') as file:
            data = json.load(file)
        c_detailed_results = {
                  "concurrency": concurrency,
                  "input_token_length": data["mean_input_tokens"],
                  "output_token_length": data["mean_output_tokens"],
                  "first-time-to-token_mean_in_ms": data['results_ttft_s_mean']*1000,
                  "throughput_token_per_s": data['results_mean_output_throughput_token_per_s'],
                  "latency_ms_per_token": data['results_inter_token_latency_s_mean']*1000,
        }
        # append results
        results[concurrency] = data
        detailed_results[concurrency] = c_detailed_results
        with open(f'{config.model_id.replace("/","_")}_tp_{config.num_gpus}_cur_{concurrency}.json', "w") as file:
            json.dump(detailed_results[concurrency], file, indent=2)
        # remove the output directory
        subprocess.run(["rm", "-rf", output_dir])
    return results, detailed_results

def main():
    parser = argparse.ArgumentParser(description="Manage Docker, run tests, and process results.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--image-uri", type=str, default="ghcr.io/huggingface/text-generation-inference:2.0.1", help="URI of the Docker image to run.")
    args = parser.parse_args()

    # Parse the YAML file
    config = parse_yaml_file(args.config)
    print("Config:", config)
    
    # Start the Docker container
    client = docker.from_env()
    gpu_device_requests = [docker.types.DeviceRequest(count=config.num_gpus, capabilities=[['gpu']])]
    environment_vars = {
      "MODEL_ID": config.model_id, 
      "MAX_INPUT_LENGTH": str(config.tgi_config.max_input_length),
      "MAX_TOTAL_TOKENS": str(config.tgi_config.max_total_tokens),
      "MAX_BATCH_PREFILL_TOKENS": str(config.tgi_config.max_batch_prefill_tokens),
      "HUGGING_FACE_HUB_TOKEN": HfFolder.get_token(),
      "NUM_SHARD": str(config.num_gpus),
      }
    if config.tgi_config.quantize is not None:
        environment_vars["QUANTIZE"] = str(config.tgi_config.quantize)
    print("Starting Docker container...")
    container = client.containers.run(
        args.image_uri, 
        ports={'80/tcp': 80}, 
        detach=True, 
        device_requests=gpu_device_requests, 
        environment=environment_vars  # Pass environment variables
    )
    
    try:
        if wait_for_server_to_be_ready("http://localhost/health"):
            results, detailed_results = run_performance_tests(config) 
            # create detailed results
            with open(f'{config.model_id.replace("/","_")}_tp_{config.num_gpus}_summary.json', "w") as file:
                json.dump(detailed_results, file, indent=2)
            with open(f'{config.model_id.replace("/","_")}_tp_{config.num_gpus}_results.json', "w") as file:
                json.dump(results, file, indent=2)
        else:
            print("Server did not become ready in time.")
    finally:
        container.stop()
        container.remove()

if __name__ == "__main__":
    main()
