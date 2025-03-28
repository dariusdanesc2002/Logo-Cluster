import os
import tarfile
import pandas as pd
import splitfolders
from PIL import Image


def extractTarImages(tarFileName, destinationFolder):
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)

    with tarfile.open(tarFileName, 'r:gz') as tar:
        tar.extractall(destinationFolder)


# verificam daca imagina se poate deschide si nu este corupta
def checkValidImg(imgPath):
    try:
        if Image.open(imgPath).verify():
            return True
    except (IOError, SyntaxError):
        return False


def deleteBadTrainingExamples(folder_path, annotation_file_path):
    total_images_before = len(os.listdir(folder_path))

    with open(annotation_file_path, 'r') as file:
        annotations = file.readlines()

    validAnnotations = []
    for line in annotations:
        img_name, class_name, xmin, ymin, xmax, ymax = line.split()

        imgPath = os.path.join(folder_path, img_name)
        if checkValidImg(imgPath) == False:
            continue

        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        if xmin >= xmax or ymin >= ymax:
            continue

        validAnnotations.append(line)

    # Facem update la folderul unde se afla anotatiile ca sa le avem mereu pe cele bune/functionale
    with open(annotation_file_path, 'w') as file:
        file.writelines(validAnnotations)
    # O sa avem treaba doar cu root(dirpath) si files
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            impPath = os.path.join(dirpath, file)
            if checkValidImg(impPath) == False:
                os.remove(impPath)
            else:
                annotation_exists = any(file in x for x in validAnnotations)
                if not annotation_exists:
                    os.remove(impPath)

    total_images_after = len(os.listdir(folder_path))

    print(f"Total number of images before Removal: {total_images_before}")
    print(f"Total number of images after Removal: {total_images_after}")

def prepareData(df, ImagesFolderPath, output_images_folder,output_labels_folder):
    for index, row in df.iterrows():
        fileName = row['filename']
        className = row['class']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        image_path = os.path.join(ImagesFolderPath,fileName)
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        # Normalizam coordonatele
        b_center_x = (xmin + xmax) / 2
        b_center_y = (ymin + ymax) / 2
        b_width = (xmax - xmin)
        b_height = (ymax - ymin)

        b_center_x /= W
        b_center_y /= H
        b_width /= W
        b_height /= H
        output_image_path = os.path.join(output_images_folder, fileName)
        image.save(output_image_path)

        labelFilename = os.path.splitext(fileName)[0] + '.txt'
        labelPath = os.path.join(output_labels_folder, labelFilename)
        with open(labelPath, 'w') as label_file:
            label_file.write(f"{'0'} {b_center_x} {b_center_y} {b_width} {b_height}")


if __name__ == '__main__':
    # Extragere tar file + curatare set de date
    tarFileName = "C:/Users/dariu/Downloads/flickr_logos_27_dataset.tar.gz"
    destinationFolder = "C:/Veridion/FlickrLogosDataset"
    extractTarImages(tarFileName, destinationFolder)
    txt_path  = "C:/Veridion/FlickrLogosDataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
    df = pd.read_csv(txt_path,
                     sep='\s+',
                     header=None)
    df = df.drop(columns=2)
    columns = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df.columns = columns
    df['class'] = 'logo'
    df.to_csv(txt_path, sep=' ', header=True, index=False)
    folder_path = "C:/Veridion/FlickrLogosDataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images/flickr_logos_27_dataset_images"
    annotation_file_path = "C:/Veridion/FlickrLogosDataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
    deleteBadTrainingExamples(folder_path, annotation_file_path)

    # Pregatire set de date, sa fie compatibil cu YOLO
    ImagesFolderPath = "C:/Veridion/FlickrLogosDataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images/flickr_logos_27_dataset_images"
    OutputFolderPath = "C:/Veridion/Dataset"

    output_images_folder = os.path.join(OutputFolderPath, 'images')
    output_labels_folder = os.path.join(OutputFolderPath, 'labels')
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    df = pd.read_csv(txt_path,
                     sep='\s+',
                     header=None)
    columns = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df.columns = columns
    prepareData(df, ImagesFolderPath, output_images_folder, output_labels_folder)
    input_folder = "C:/Veridion/Dataset"
    output = "C:/Veridion/Dataset/data"

    splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .1, .1))

