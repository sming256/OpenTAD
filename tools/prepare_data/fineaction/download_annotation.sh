DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for FineAction-1.3
gdown --folder https://drive.google.com/drive/folders/1R2ji2InJONWzBfLaEmp2CiT3FHHiZSJI -O $DATA_DIR --folder