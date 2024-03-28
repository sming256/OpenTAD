DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for ActivityNet-1.3
gdown https://drive.google.com/drive/folders/1HwLFaUdrTLkTcx0oKm_z9WPyQiD2moJU -O $DATA_DIR --folder
