import os

image_size = (256,256)
num_frames = 12
fps = 3
save_dir = "./samples"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 4
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=os.environ["DCCP_WORLD_MODEL_PATH"],
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    action_dim=7,
    pad_action_num = 4,
)
vae = dict(
    type="SDXL",
    from_pretrained=os.environ["DCCP_VAE_PATH"],
    micro_frame_size=8,
    micro_batch_size=64,
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)
