INPUT_FILE_OR_FOLDER=demo/demo.mp4
CONFIG_FILE=configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_bdd100k.py
OUTPUT_FILE_OR_FOLDER=demo/local_output.mp4
CHECKPOINT_FILE=configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.pth
FPS=5
echo "python tools/inference.py ${CONFIG_FILE} --input ${INPUT_FILE_OR_FOLDER} --output ${OUTPUT_FILE_OR_FOLDER} --checkpoint ${CHECKPOINT_FILE} --fps ${FPS}"
python tools/inference.py ${CONFIG_FILE} --input ${INPUT_FILE_OR_FOLDER} --output ${OUTPUT_FILE_OR_FOLDER} --checkpoint ${CHECKPOINT_FILE} --fps ${FPS}
