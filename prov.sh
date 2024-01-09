#!/bin/false

printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
function download() {
    wget -q --show-progress -e dotbytes="${3:-4M}" -O "$2" "$1"
}

## Set paths
nodes_dir=/opt/ComfyUI/custom_nodes
models_dir=/opt/ComfyUI/models
checkpoints_dir=${models_dir}/checkpoints
vae_dir=${models_dir}/vae
controlnet_dir=${models_dir}/controlnet
loras_dir=${models_dir}/loras
upscale_dir=${models_dir}/upscale_models
animated_models_dir=${nodes_dir}/ComfyUI-AnimateDiff-Evolved/models
motion_models_dir=${nodes_dir}/ComfyUI-AnimateDiff-Evolved/motion_lora
face_restore_models_dir=${models_dir}/facerestore_models

### Install custom nodes

# ComfyUI-AnimateDiff-Evolved
this_node_dir=${nodes_dir}/ComfyUI-AnimateDiff-Evolved
if [[ ! -d $this_node_dir ]]; then
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved $this_node_dir
else
    (cd $this_node_dir && git pull)
fi

# ComfyUI-Advanced-ControlNet
this_node_dir=${nodes_dir}/ComfyUI-Advanced-ControlNet
if [[ ! -d $this_node_dir ]]; then
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet $this_node_dir
else
    (cd $this_node_dir && git pull)
fi

## Animated
model_file=${animated_models_dir}/mm_sd_v15_v2.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "mm_sd_v15_v2.ckpt...\n"
    download ${model_url} ${model_file}
fi

model_file=${animated_models_dir}/v3_sd15_adapter.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v3_sd15_adapter.ckpt...\n"
    download ${model_url} ${model_file}
fi

model_file=${animated_models_dir}/v3_sd15_mm.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v3_sd15_mm.ckpt...\n"
    download ${model_url} ${model_file}
fi

model_file=${animated_models_dir}/mm_sdxl_v10_beta.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "mm_sdxl_v10_beta.ckpt...\n"
    download ${model_url} ${model_file}
fi

model_file=${animated_models_dir}/v3_sd15_sparsectrl_scribble.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_scribble.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v3_sd15_sparsectrl_scribble.ckpt...\n"
    download ${model_url} ${model_file}
fi

model_file=${motion_models_dir}/v2_lora_RollingAnticlockwise.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v2_lora_RollingAnticlockwise...\n"
    download ${model_url} ${model_file}
fi

model_file=${motion_models_dir}/v2_lora_RollingClockwise.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v2_lora_RollingClockwise...\n"
    download ${model_url} ${model_file}
fi

model_file=${motion_models_dir}/v2_lora_ZoomIn.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v2_lora_ZoomIn...\n"
    download ${model_url} ${model_file}
fi

model_file=${motion_models_dir}/v2_lora_ZoomOut.ckpt
model_url=https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt
if [[ ! -e ${model_file} ]]; then
    printf "v2_lora_ZoomOut...\n"
    download ${model_url} ${model_file}
fi

## Standard
# v1-5-pruned-emaonly
# model_file=${checkpoints_dir}/v1-5-pruned-emaonly.ckpt
# model_url=https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

# if [[ ! -e ${model_file} ]]; then
#     printf "Downloading Stable Diffusion 1.5...\n"
#     download ${model_url} ${model_file}
# fi

### Download controlnet
# done below
# model_file=${controlnet_dir}/control_canny-fp16.safetensors
# model_url=https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors
# if [[ ! -e ${model_file} ]]; then
#    printf "Downloading Canny...\n"
#    download ${model_url} ${model_file}
# fi

### Download loras

model_file=${loras_dir}/epi_noiseoffset2.safetensors
model_url=https://civitai.com/api/download/models/16576
if [[ ! -e ${model_file} ]]; then
   printf "Downloading epi_noiseoffset2 lora...\n"
   download ${model_url} ${model_file}
fi

model_file=${loras_dir}/lcm-lora-sdxl.safetensors
model_url=https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors
if [[ ! -e ${model_file} ]]; then
   printf "Downloading lcm-lora-sdxl lora...\n"
   download ${model_url} ${model_file}
fi

NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    # "https://github.com/Gourieff/comfyui-reactor-node" this one is not working, I think it needs to download and install onnxruntime-gpu
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/FizzleDorf/ComfyUI_FizzNodes"
)

CHECKPOINT_MODELS=(
    "https://civitai.com/api/download/models/128713" # dreamshaper_8 sd 1.5
    #"https://civitai.com/api/download/models/251662" # Dreamshaper_XL sd xl
    "https://civitai.com/api/download/models/245598" # Realistic Vision V6.0 B1 sd 1.5
    #"https://civitai.com/api/download/models/247444" # nightvision sd xl
    "https://civitai.com/api/download/models/132760" # absolute reality sd 1.5
    "https://civitai.com/api/download/models/289073" # real dream sd 1.5
    #"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    #"https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
)

LORA_MODELS=(

)

VAE_MODELS=(
    # "https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
)

ESRGAN_MODELS=(
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
    "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_RealisticRescaler_100000_G.pth"
    "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth"
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
    "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth"
    "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
    "https://huggingface.co/Akumetsu971/SD_Anime_Futuristic_Armor/resolve/main/4x_NMKD-Siax_200k.pth"
)

CONTROLNET_MODELS=(
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth"
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors"
    "https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors"
    "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors"
    #"https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors"
)

FACERESTORE_MODELS=(
    "https://huggingface.co/nlightcho/gfpgan_v14/resolve/main/GFPGANv1.4.pth"
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

# nodes_dir=/opt/ComfyUI/custom_nodes
# models_dir=/opt/ComfyUI/models
# checkpoints_dir=${models_dir}/checkpoints
# vae_dir=${models_dir}/vae
# controlnet_dir=${models_dir}/controlnet
# loras_dir=${models_dir}/loras
# upscale_dir=${models_dir}/upscale_models
# animated_models_dir=${nodes_dir}/ComfyUI-AnimateDiff-Evolved/models
# motion_models_dir=${nodes_dir}/ComfyUI-AnimateDiff-Evolved/motion_lora
# face_restore_models_dir=${models_dir}/facerestore_models

function provisioning_start() {
    DISK_GB_AVAILABLE=$(($(df --output=avail -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_USED=$(($(df --output=used -m "${WORKSPACE}" | tail -n1) / 1000))
    DISK_GB_ALLOCATED=$(($DISK_GB_AVAILABLE + $DISK_GB_USED))
    provisioning_print_header
    provisioning_get_nodes
    provisioning_get_models \
        "${checkpoints_dir}" \
        "${CHECKPOINT_MODELS[@]}"
    provisioning_get_models \
        "${loras_dir}" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${controlnet_dir}" \
        "${CONTROLNET_MODELS[@]}"
    provisioning_get_models \
        "${vae_dir}" \
        "${VAE_MODELS[@]}"
    provisioning_get_models \
        "${upscale_dir}" \
        "${ESRGAN_MODELS[@]}"
    provisioning_get_models \
        "${face_restore_models_dir}" \
        "${FACERESTORE_MODELS[@]}"
    provisioning_print_end
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${repo}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                    micromamba -n comfyui run ${PIP_INSTALL} -r "$requirements"
                fi
            fi
        else
            printf "Downloading node: %s...\n" "${repo}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                micromamba -n comfyui run ${PIP_INSTALL} -r "${requirements}"
            fi
        fi
    done
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    dir="$1"
    mkdir -p "$dir"
    shift
    if [[ $DISK_GB_ALLOCATED -ge $DISK_GB_REQUIRED ]]; then
        arr=("$@")
    else
        printf "WARNING: Low disk space allocation - Only the first model will be downloaded!\n"
        arr=("$1")
    fi
    
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        printf "Downloading: %s\n" "${url}"
        provisioning_download "${url}" "${dir}"
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#          Provisioning container            #\n#                                            #\n#         This will take some time           #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
    if [[ $DISK_GB_ALLOCATED -lt $DISK_GB_REQUIRED ]]; then
        printf "WARNING: Your allocated disk size (%sGB) is below the recommended %sGB - Some models will not be downloaded\n" "$DISK_GB_ALLOCATED" "$DISK_GB_REQUIRED"
    fi
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Web UI will start now\n\n"
}

# Download from $1 URL to $2 file path
function provisioning_download() {
    wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
}

provisioning_start