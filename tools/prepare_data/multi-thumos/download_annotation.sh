DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Multi-THUMOS
gdown --folder https://drive.google.com/drive/folders/1benWupqjUEqxsup514hopBX1mHU2nrmd -O $DATA_DIR --folder