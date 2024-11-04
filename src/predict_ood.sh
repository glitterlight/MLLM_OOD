ROOT_PATH="/home/ma-user/work/code_dev/siming"
export PYTHONPATH=$ROOT_PATH:$PYTHONPATH


# export LOCAL_WORLD_SIZE=$(echo $MA_NUM_GPUS);
# export WORLD_SIZE=$(echo $MA_NUM_HOSTS);
# export RANK=$(echo $VC_TASK_INDEX);
# export MASTER_ADDR=$(echo $VC_WORKER_HOSTS);
# export MASTER_PORT=9899;
# export NCCL_DEBUG=INFO;

# test path config
export LOCAL_WORLD_SIZE=8
export WORLD_SIZE=1
export RANK=0



source /home/ma-user/work/code_dev/miniconda3/bin/activate /home/ma-user/work/code_dev/siming/miniconda3/envs/mllm


# # imagenet10
export CUDA_VISIBLE_DEVICES="6,7"
method="baseline_split_20"
model_name="qwen2-vl-72b"
# src_dir="${ROOT_PATH}/datasets/iNaturalist/images"
# dst_dir="${ROOT_PATH}/ood_labels/iNaturalist"
src_dir="${ROOT_PATH}/datasets/imagenet100/val"
dst_dir="${ROOT_PATH}/ood_labels/imagenet100"
classes_dataset_name="imagenet100"
# few_shot_path="${ROOT_PATH}/few_shots/few_shot_vllm_cot2.json"
num_workers=2
# batch_size=1000
deal_num=30
save_flag=0

python ${ROOT_PATH}/src/predict_ood.py \
    --method ${method} \
    --model_name ${model_name} \
    --classes_dataset_name ${classes_dataset_name} \
    --src_dir ${src_dir} \
    --dst_dir ${dst_dir} \
    --num_workers ${num_workers} \
    --save_flag ${save_flag} \
    --deal_num ${deal_num} \
    # --few_shot_path ${few_shot_path} \
    # --batch_size ${batch_size} \

# # imagenet20
# src_dir="${ROOT_PATH}/datasets/imagenet20/val"
# dst_dir="${ROOT_PATH}/ood_labels/imagenet20"
# classes_dataset_name="imagenet20"
# num_workers=8

# python ${ROOT_PATH}/src/predict_ood.py \
#     --model_name ${model_name} \
#     --classes_dataset_name ${classes_dataset_name} \
#     --src_dir ${src_dir} \
#     --dst_dir ${dst_dir} \
#     --num_workers ${num_workers} \

# # imagenet100
# src_dir="${ROOT_PATH}/datasets/imagenet100/val"
# dst_dir="${ROOT_PATH}/ood_labels/imagenet100"
# classes_dataset_name="imagenet100"
# num_workers=8

# python ${ROOT_PATH}/src/predict_ood.py \
#     --model_name ${model_name} \
#     --classes_dataset_name ${classes_dataset_name} \
#     --src_dir ${src_dir} \
#     --dst_dir ${dst_dir} \
#     --num_workers ${num_workers} \

# # imagenet1k
# src_dir="${ROOT_PATH}/datasets/imagenet1k/val"
# dst_dir="${ROOT_PATH}/ood_labels/imagenet1k"
# classes_dataset_name="imagenet1k"

# python ${ROOT_PATH}/src/predict_ood.py \
#     --model_name ${model_name} \
#     --classes_dataset_name ${classes_dataset_name} \
#     --src_dir ${src_dir} \
#     --dst_dir ${dst_dir} \