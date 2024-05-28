DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Ego4D-MQ
gdown https://drive.google.com/drive/folders/12DIU_htIlKOQINRwYCOK9bA57awJxHfA -O $DATA_DIR --folder
