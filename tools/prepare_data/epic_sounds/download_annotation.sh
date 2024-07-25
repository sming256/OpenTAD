DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Epic-Kitchens-100
gdown --folder https://drive.google.com/drive/folders/1_uFPyDElFaAENPeMB3PM_gAymATMNyWs -O $DATA_DIR --folder