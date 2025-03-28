import os
import splitfolders


def createImagesFolder(imagesPath, labelsPath, screenshotsPath):
    for no in os.listdir(labelsPath):
        base_name = os.path.splitext(no)[0]
        screenshotName = base_name + '.png'
        screenshotPath = os.path.join(screenshotsPath, screenshotName)
        imgPath = os.path.join(imagesPath, screenshotName)
        with open(screenshotPath, "rb") as f:
            img_data = f.read()
        with open(imgPath, "wb") as f:
            f.write(img_data)


if __name__ == '__main__':
    labelsPath = "C:/Veridion/Cod/PrepareData2.0/labels"
    imagesPath = "C:/Veridion/Cod/PrepareData2.0/img"
    screenshotsPath = "C:/Veridion/Cod/Screenshots"
    # createImagesFolder(imagesPath, labelsPath, screenshotsPath)
    inputFolder = "C:/Veridion/Dataset2"
    output = "C:/Veridion/Dataset2/data"
    dataPath = splitfolders.ratio(inputFolder, output=output, seed=42, ratio=(.8, .1, .1))

