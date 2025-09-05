from src.ssunet.datasets import BinomDataset, ValidationDataset
from src.ssunet.models import Bit2Bit
from src.ssunet.configs import MasterConfig
from src.ssunet.constants import DEFAULT_CONFIG_PATH

config = MasterConfig.from_config(DEFAULT_CONFIG_PATH)
config.copy_config(DEFAULT_CONFIG_PATH)

# Load data for training and validation, ground truth for validation metrics
data = config.path_config.load_data_only()
# validation_data = config.path_config.load_reference_and_ground_truth()
validation_data = config.path_config.load_reference_only()

# Load data configurations and disable augmentation for validation
data_config = config.data_config
validation_config = data_config.validation_config

# Create training and validation datasets
training_data = BinomDataset(data, data_config, config.split_params)
validation_data = ValidationDataset(validation_data, validation_config)

# Create model
model = Bit2Bit(config.model_config)

# Create data loaders
training_loader = config.loader_config.loader(training_data)
validation_loader = config.loader_config.loader(validation_data)

# Print the input size
print(f"input_size: {tuple(next(iter(training_loader))[1].shape)}")

# Train the model
trainer = config.trainer
trainer.fit(model, training_loader, validation_loader)
trainer.save_checkpoint(config.train_config.default_root_dir / "model.ckpt")

from src.tools.tools import clear_vram
clear_vram()

from src.tools.gpuinference import gpu_patch_inference
import numpy as np
from tifffile import imwrite

output = gpu_patch_inference(
    model,
    data.primary_data.astype(np.float32),
    min_overlap=48, # Already 3x rn. TODO: This is the overlap between patches. Increase it! We have more VRAM now. 
    initial_patch_depth=64, # Already 2x rn. TODO: This is the initial patch depth. Increase it! We have more VRAM now. 
    device=config.device,
)

imwrite(config.train_config.default_root_dir / "input.tif", data.primary_data)
imwrite(config.train_config.default_root_dir / "inference.tif", output)