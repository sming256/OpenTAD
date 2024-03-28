DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for HACS-1.1
gdown --folder https://drive.google.com/drive/folders/1K5tPK-HYkKHYFsvkOydAGCnM9R5Nrrqx -O $DATA_DIR --folder