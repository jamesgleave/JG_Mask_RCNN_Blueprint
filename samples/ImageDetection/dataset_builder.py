import requests
import os
import random
import cv2


def download_image(url, filename, save_path):
    if save_path is None:
        save_path = ""

    url = url
    filename = (save_path + "/" + filename) + "." + url.split('.')[-1]
    print(filename)
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)


def scan_dir(file_path):
    file_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".txt"):
                f = open(os.path.join(root, file), "r")
                name = root.split("/")[-1]
                file_list.append((f, name))
    return parse_txt(file_list)


def parse_txt(txt_list):
    url_list = []

    for txt in txt_list:
        name = txt[1]
        for url in txt[0].readlines():
            url_list.append((url.split("\n")[0], name))
    return url_list


def extract_random_images(url_list, number_of_images, save_path):
    rand_urls = random.sample(url_list, number_of_images)
    index = 0
    for url in rand_urls:
        index += 1
        download_image(url[0], url[1] + str(index), save_path=save_path)


def rename_all(dir_path, name, use_index=True):
    index = 0
    if os.path.isdir(dir_path):
        for root, dirs, file in os.walk(dir_path):
            for f in file:
                index += 1
                print(file)
                if use_index:
                    i = str(index)
                else:
                    i = ""
                path = dir_path + f
                os.rename(path, dir_path + name + i + ".jpg")

    else:
        print("Please enter an absolute path to a directory")


def get_images(file_path, number_of_images, save_location=None, make_dir=False):

    if make_dir:
        dir_name = save_location.split("/")[-1]
        os.makedirs(dir_name, exist_ok=True)

    url_list = scan_dir(file_path)
    extract_random_images(url_list, number_of_images, save_path=save_location)


if __name__ == '__main__':
    import argparse

    rename_all("/Users/martingleave/Downloads/JG_Mask_RCNN_Blueprint-master/samples/ImageDetection/images/",
               "PHOTO2", True)

    get_images(file_path="/Users/martingleave/Downloads/JG_Mask_RCNN_Blueprint-master/datasets/nsfw_data_source_urls-master/raw_data/body-parts_lower-body_asshole",
               number_of_images=300,
               save_location="/Users/martingleave/Downloads/JG_Mask_RCNN_Blueprint-master/samples/ImageDetection/images",
               make_dir=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='This downloads and renames image directories from urls')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'download' or 'rename'")
    parser.add_argument('--path', required=True,
                        metavar="/path/to/url/directory/",
                        help='Path to directory')
    parser.add_argument('--num_images', required=False,
                        metavar=int,
                        help='number of images to download')
    parser.add_argument('--save_path', required=False,
                        metavar="/path/to/save/location",
                        help='path to desired save location')
    parser.add_argument('--make_dir', required=False,
                        metavar="",
                        help='create new dir, naming it with the end of --path')
    parser.add_argument('--name', required=False,
                        metavar="str",
                        help='name to rename file')
    parser.add_argument('--use_index', required=False,
                        metavar="Bool",
                        help='Use indexing in renaming')

    args = parser.parse_args()

    if args.command == "download":
        assert args.num_images and args.save_path, "num_images and save_path are required for download"
        get_images(file_path=args.path, number_of_images=int(args.num_images),
                   save_location=args.save_path, make_dir=args.make_dir)

    if args.command == "rename":
        assert args.name and args.path, "name and path are required for download"
        rename_all(args.path, args.name, args.use_index)


get_images(file_path="datasets/nsfw_data_source_urls-master/raw_data/body-parts_lower-body_genitalia_penis/",
           number_of_images=200,
           save_location="samples/ImageDetection/images/", make_dir=True)
