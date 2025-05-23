import os
import gdown

os.makedirs("data", exist_ok=True)

files = {
    "UTKFaces_dataset_1.tar.gz": "1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW",
    "UTKFaces_dataset_2.tar.gz": "19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b",
    "UTKFaces_dataset_3.tar.gz": "1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b"
}

for filename, file_id in files.items():
    filepath = os.path.join("data", filename)
    if os.path.exists(filepath):
        print(f"{filename} already exists. Skipping.")
    else:
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
