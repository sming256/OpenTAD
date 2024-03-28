DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for THUMOS-14
gdown --folder https://drive.google.com/drive/folders/1sGTFuJ-G08sOZi9SHCBR7W3Q8IIryHKN -O $DATA_DIR --folder