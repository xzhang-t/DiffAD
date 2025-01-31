MVTEC_DATA_DIR='/path/to/mvtec_anomaly_detection'
DTD_DATA_DIR='/path/to/dtd/images'
 
declare -a arr=('bottle' 'carpet' 'cable' 'grid' 'leather' 'tile' 'wood' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')

## now loop through the above array
for OBJ_NAME in "${arr[@]}"

do

python rec_network/main.py --base ./configs/kl.yaml -t --gpus 0,  --obj_name ${OBJ_NAME}

python rec_network/main.py --base ./configs/mvtec.yaml -t --gpus 0, -max_epochs 300, --obj_name ${OBJ_NAME}

python seg_network/train.py --gpu_id 0 --lr 0.001 --bs 32 --epochs 300 --data_path ${MVTEC_DATA_DIR} --anomaly_source_path ${DTD_DATA_DIR} --checkpoint_path ./checkpoints/${OBJ_NAME} --obj_name  ${OBJ_NAME} --log_path ./logs/${OBJ_NAME}

done

# test epoch 100
python seg_network/test.py --base_model_name "DRAEM_test_0.001_300_bs32" --data_path "/mnt/isilon/shicsonmez/ad/data/mvtec_anomaly_detection" --checkpoint_path "./checkpoints/" --gpu_id "0" --epoch "100"
# test last epoch
python seg_network/test.py --base_model_name "DRAEM_test_0.001_300_bs32" --data_path "/mnt/isilon/shicsonmez/ad/data/mvtec_anomaly_detection" --checkpoint_path "./checkpoints/" --gpu_id "0" --epoch ""