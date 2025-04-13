import cv2
import pandas as pd
import requests
import urllib3
import os
import imagehash
import matplotlib.pyplot as plt
import matplotlib
import time
import skimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from PIL._imaging import display
from selenium import webdriver
from ultralytics import YOLO
from selenium.webdriver.chrome.options import Options
from skimage import io,color


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
matplotlib.use('TkAgg')


def readParquet(filepath):
    df = pd.read_parquet(filepath)
    return df


def checkLinkAvailable(domain):
    url = f"https://{domain}"
    workingDF = pd.DataFrame(columns=['domain'])
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        r = requests.get(url, timeout=5, headers=headers, verify=False)
        # requests.codes.ok <=> 200: ok, okay, all_ok, all_okay, all_good, \o/, ✓
        if r.status_code == requests.codes.ok:
            return domain
        else:
            print(f"{url} returned status code: {r.status_code}")

    except requests.exceptions.Timeout:
        print(f"{url}: time too long")

    except requests.exceptions.RequestException:
        print(f"Error accessing {url}: {requests.exceptions.RequestException}")

    return None


def createGoodParquetFolder(working_df, folderPath):
    parquetDir = os.path.join(folderPath, "WorkingParquet")
    os.mkdir(parquetDir)
    filePath = os.path.join(parquetDir, "data.parquet")
    working_df.to_parquet(filePath, index=False)


def captureScreenshot(domain, imgPath, headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        url = f"https://{domain}"
        driver.get(url)
        time.sleep(3)
        driver.save_screenshot(imgPath)
    except Exception as e:
        print(f"[!] Failed to capture {domain}: {e}")
    finally:
        driver.quit()


def takeImagesFromDomain(workingDf, folderPath, headless=True, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for counter, row in enumerate(workingDf.itertuples()):
            domain = row[1]
            imgPath = os.path.join(folderPath, f"{counter}.png")
            futures.append(executor.submit(captureScreenshot, domain, imgPath, headless))


def takeLogoFromImages(folderPath, screenshotPath, model):

    for img in os.listdir(screenshotPath):
        logoPath = os.path.join(folderPath, f"{os.path.splitext(img)[0]}.png")
        img_rgb = io.imread(os.path.join(screenshotPath, img))
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        result_predict = model.predict(
            source=img_rgb,
            conf=0.1,
            iou=0.45,
            verbose=False
        )
        result = result_predict[0]
        max_conf = -1
        for box in result.boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if conf > max_conf:
                max_conf = conf
                best_crop = img_rgb[y1:y2, x1:x2]

        if len(result.boxes.conf) > 0:
            cv2.imwrite(logoPath, best_crop)


def createPerceptualHash(logosPath, df):
    hashFileNameToHash = {}
    hashFileNameToNames = {}

    for file_name in os.listdir(logosPath):
        img_path = os.path.join(logosPath, file_name)
        image = Image.open(img_path)
        phash = imagehash.phash(image)
        hashFileNameToHash[file_name] = phash

    for f1, h1 in hashFileNameToHash.items():
        if f1 not in hashFileNameToNames.values():
            number = os.path.splitext(f1)[0]
            hashFileNameToNames[f1] = [df.iloc[int(number), 0]]
        else:
            continue
        for f2, h2 in hashFileNameToHash.items():
            if f1 < f2:  # just to avoid double comparisons
                distance = h1 - h2  # this calculates the Hamming distance
                if distance <= 5:  # threshold depends on your data
                    # print(f"{f1} and {f2} are probably the same or very similar (distance={distance})")
                    hashFileNameToNames[f1].append(df.iloc[int(os.path.splitext(f2)[0]), 0])

    return hashFileNameToNames


if __name__ == "__main__":

    df = readParquet('C:/Users/dariu/Downloads/logos.snappy.parquet')

    working_domains = []
    # Am paralelizat procesul de selectare a domeniilor functionale pentru un timp de executie mai bun
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(checkLinkAvailable, row.domain) for row in df.itertuples()]
        for future in as_completed(futures):
            result = future.result()
            if result:  # Only append if result is not None
                working_domains.append(result)

    working_df = pd.DataFrame(working_domains, columns=['domain'])
    folderPath = "C:/Veridion"
    createGoodParquetFolder(working_df, folderPath)
    workingDF = readParquet("C:/Veridion/WorkingParquet/data.parquet")
    takeImagesFromDomain(workingDF, "C:/Veridion/Cod/Screenshots")
    model = YOLO('C:/Veridion/Cod/Veridion_Cod/Model/best.pt')
    logosPath = "C:/Veridion/Cod/Logos"
    takeLogoFromImages("C:/Veridion/Cod/Logos", "C:/Veridion/Cod/Screenshots", model)
    #In aceasta functie am calculat atat PerceptualHash al fiecarui logo in parte, cat si impartirea domeniilor pe baza similaritaii logourilor
    hashFileNameToNames = createPerceptualHash(logosPath, df)


