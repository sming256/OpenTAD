DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Charades
gdown --folder https://drive.google.com/drive/folders/1oON5K5hSa5jexnB4IGdsW6qDB3w1I_QG -O $DATA_DIR --folder