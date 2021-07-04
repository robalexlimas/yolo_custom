import os, wget
from file_manager import create_path, dir_exist, get_files
from yolo_custom import save_model


def download_files(urls):
    path = os.path.join(os.getcwd(), 'weights')
    path_download = os.path.join(path, 'originals')
    if not dir_exist(path_download):
        create_path(path_download)
        download(path_download, urls)
    else:
        files = get_files(path_download)
        download_bool = False
        for url in urls:
            for file in files:
                if file in url:
                    download_bool = True
                    break
            print(url, 'Downloaded: ', download_bool)
            if not download_bool:
                download(path_download, [url])
                name = url.split('/')[-1]
                save_model(name)
    

def download(path, urls):
    for url in urls:
        file_path = os.path.join(path, url.split('/')[-1])
        wget.download(url, file_path)
