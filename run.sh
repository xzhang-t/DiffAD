MVTEC_DATA_DIR='/path/to/mvtec'
VISA_DATA_DIR='/path/to/visa/dataset'
DTD_DATA_DIR='/path/to/dtd/images'
DATASET='mvtec'
DATA_DIR=$VISA_DATA_DIR

# MVTEC CLASSES
if [ ${DATASET} == "mvtec" ]; then
declare -a arr=('bottle' 'carpet' 'cable' 'grid' 'leather' 'tile' 'wood' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')
fi

# VISA CLASSEES
if [ ${DATASET} == "visa" ]; then
declare -a arr=('candle'  'capsules'  'cashew'  'chewinggum'  'fryum'  'macaroni1'  'macaroni2'  'pcb1'  'pcb2'  'pcb3'  'pcb4'  'pipe_fryum')
fi

## now loop through the above array
for OBJ_NAME in "${arr[@]}"

do

python rec_network/main.py --base ./configs/kl.yaml -t --gpus 0,  --obj_name ${OBJ_NAME} --dataset ${DATASET}

python rec_network/main.py --base ./configs/mvtec.yaml -t --gpus 0, -max_epochs 300, --obj_name ${OBJ_NAME} --dataset ${DATASET}

python seg_network/train.py --gpu_id 0 --lr 0.001 --bs 32 --epochs 300 --data_path ${DATA_DIR} --anomaly_source_path ${DTD_DATA_DIR} --checkpoint_path ./checkpoints_${DATASET}/${OBJ_NAME} --obj_name  ${OBJ_NAME} --log_path ./logs/${OBJ_NAME} --dataset ${DATASET}

done

# test epoch 100
python seg_network/test.py --base_model_name "DRAEM_test_0.001_300_bs32" --data_path ${DATA_DIR} --checkpoint_path ./checkpoints_${DATASET}/ --gpu_id "0" --epoch "100" --dataset ${DATASET}
# test last epoch
python seg_network/test.py --base_model_name "DRAEM_test_0.001_300_bs32" --data_path ${DATA_DIR} --checkpoint_path ./checkpoints_${DATASET}/ --gpu_id "0" --epoch "" --dataset ${DATASET}