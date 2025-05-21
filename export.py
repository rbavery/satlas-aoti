import json
import torch
from torch.export import Dim
import os
from huggingface_hub import hf_hub_download
#from torch._inductor.test_case import TestCase
import typer
import satlas.model.evaluate
import satlas.model.model
import time
import logging

def check_nvidia_drivers():
    nvidia_smi = os.popen('nvidia-smi').read()
    if 'NVIDIA-SMI' in nvidia_smi:
        return nvidia_smi
    return "NVIDIA drivers not found."

def check_cuda_installation():
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-11.0',
        '/usr/local/cuda-11.1',
        '/usr/local/cuda-11.2',
        '/usr/local/cuda-11.3',
        '/usr/local/cuda-11.4',
        '/usr/local/cuda-11.5',
        '/usr/local/cuda-11.6',
        '/usr/local/cuda-11.7',
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-12.0'
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            return f"CUDA installation found at: {path}"
    return "CUDA installation not found."

if not torch.cuda.is_available():
    print("CUDA is not available. Please run this script on a machine with a CUDA-compatible GPU.")
    print("Debugging Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"CUDA devices: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'None'}")
    print(check_nvidia_drivers())
    print(check_cuda_installation())
    raise Exception("CUDA is not available.")

torch._logging.set_logs(all=logging.ERROR)

full_path = os.path.dirname(os.path.abspath(__file__))

torch.set_float32_matmul_precision("high")
# os.environ["TORCH_LOGS"] = "+dynamic"
config_path = f"{full_path}/src/configs/satlas_explorer_solar_farm.txt"
size = 1024

with open(config_path, "r") as f:
    config = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for spec in config["Tasks"]:
    if "Task" not in spec:
        spec["Task"] = satlas.model.dataset.tasks[spec["Name"]]
model = satlas.model.model.Model(
    {
        "config": config["Model"],
        "channels": config["Channels"],
        "tasks": config["Tasks"],
    }
)

app = typer.Typer()

def export_to_torchep(model, name: str, img_size: int = 1024, save_dir: str = './', load: bool = True, compare: bool = False):
    "Save the model to pytorch ExportedProgram format."
    bs_min = 2
    example_inputs =  (torch.randn(bs_min, 9*4, img_size, img_size).to("cuda"), )

    # dynamic shapes for model export
    batch_size = Dim("batch", min=2, max=13)
    #height = Dim("height", min=2, max=2048)
    #width = Dim("width", min=2, max=2048)
    dynamic_shapes = {
        "batch_tensor": {0: batch_size},
    }

    # Export the model to pytorch ExportedProgram format
    ep = torch.export.export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    package_path = f"{os.path.abspath(save_dir)}/{name}"
    if os.path.exists(package_path) and load:
        pass
    else:
        start_time = time.time()
        torch._inductor.aoti_compile_and_package(
            ep, package_path=package_path
        )
        aot_compile_time = time.time() - start_time
        print(f"AOT model compile time: {aot_compile_time:.6f} seconds")

    loaded = torch._inductor.aoti_load_package(package_path)
    # Measure inference time for the AOT model
    start_time = time.time()
    compile_results = loaded(*example_inputs)
    aot_inference_time = time.time() - start_time
    print(f"AOT model inference time: {aot_inference_time:.6f} seconds")

    if compare:
        # we don't always do this because eager mode can OOM when aoti doesn't
        # Measure inference time for the eager mode model
        start_time = time.time()
        nocompile_results = model.eval()(*example_inputs)
        eager_inference_time = time.time() - start_time
        print(f"Eager mode model inference time: {eager_inference_time:.6f} seconds")
        # TODO move to tests, this screws up Typer
        # testcase = TestCase()
        # testcase.assertEqual(compile_results, nocompile_results)
        print(
            f"Model exported to pytorch AOT .pt2 format: {os.path.abspath(save_dir)}/{name}"  # noqa: E501
        )
    return package_path

@app.command()
def main(
    name: str,
    img_size: int = 1024,
    save_dir: str = './',
    load: bool = True,
    compare: bool = False
):
    device = "cuda"

    print(f"Downloading from Hugging Face Hub or loading from local dir...")
    try:
        # Provide the repository ID and filename on the Hub
        weights_path = hf_hub_download(
            repo_id="allenai/satlas-pretrain",
            subfolder="finetuned_satlas_explorer_models_2023-07-24",
            filename="finetuned_satlas_explorer_sentinel2_solar_farm.pth",
            local_dir=full_path,
            revision='5f6eff89d0675b7601bbe8c8d68956163ae07dd0'
        )
        print(f"Weights downloaded to: {weights_path}")
    except Exception as e:  # Catch potential download errors
        print(f"Error downloading weights: {e}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Exporting model")
    export_to_torchep(model, name, img_size, save_dir, load, compare)

if __name__ == "__main__":
    app()
