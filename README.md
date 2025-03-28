# Logo-Cluster
This project focuses on grouping visually similar logos from various web domains using perceptual hashing and computer vision techniques.
# Steps For Solving The Problem
### Step 1: Preparing the PARQUET file
I extracted URLs from the `logos.snappy.parquet` file. Using the `requests` library, I then checked the availability of each domain to ensure that the URLs were accessible and still active.
The most common issue I encountered during this process was the `403: Forbidden` error. This usually means that the server understood the request but refused to authorize it. To bypass this, I added a custom `User-Agent` header to mimic a real browser.
However, the problem sometimes persists, depending on the website's security checks and restrictions.
To speed up the checking process, I used multithreading, which allowed multiple requests to be processed concurrently and significantly reduced execution time, i used `ThreadPoolExecutor` from `concurrent.futures`
### Step 2: Taking Screenshots of Each Web Page
Using the filtered PARQUET file containing only working domains, I accessed each URL and captured a screenshot of the corresponding web page using the `selenium` library. To speed up the process, I applied the same principle as in Step 1 — using multithreading to take multiple screenshots concurrently.
### Step 3: Training a CNN for Logo Detection
This was one of the most interesting parts of the project, both from a coding and logical thinking perspective.
My initial idea was to train a model to detect various logos in general contexts, not specifically on web pages. I started with the `FlickrLogos-27` dataset, which includes over four thousand samples across multiple classes.
However, the dataset format wasn't directly compatible with the `YOLO` model, so I had to make several adjustments:
- Removed corrupted images and labels with invalid values.
- Normalized the bounding box coordinates to the [0, 1] interval.
- Changed all class labels to `0`, since I was only interested in detecting whether a logo exists, not identifying the brand.

All of this preprocessing is handled in the `PrepareData` module.
I initially trained the `yolov8m` model on this dataset, and the performance was promising. On cropped images, the model made very accurate predictions. However, when applied to full screenshots of web pages, detection accuracy dropped significantly — only a few logos were detected.
To improve this, I kept the pre-trained weights and fine-tuned the model on my dataset of webpage screenshots. Although the dataset contained only 280 images, the model performed reasonably well. I used an 80/10/10 split for training, validation, and testing. 
The data preparation for this fine-tuning process is handled by `PrepareData2.0`.
#### Challenges with Model Generalization
One major issue I encountered was dataset imbalance — most logos in my web screenshots were located near the top-left corner. As a result, the model struggled to detect logos placed in the center or in less common positions on the page.
Another challenge arose when multiple logos appeared in a single screenshot — for example, both a company’s own logo and a WhatsApp icon. In some cases, the model incorrectly selected the WhatsApp logo based on its higher confidence score.

### Step 4: Cropping Logos from Screenshots
I applied the `predict` function of my trained model to each screenshot to detect logos.  
For every detected bounding box, I cropped the corresponding region from the original image and saved it to a dedicated folder.

### Step 5: Appling a clustering method that is not DBSCAN or k-means
My first thought was to use a pre-trained CNN model and take the output from one of the last layers as the feature vector. But since the logos, from the same category,  are very similar between them, meaning that they are only slightly resized, cropped, compressed, or a little blurry, I could use perceptual hashing. 
I computed for every logo his perceptual hash (using `imagehash` library). Two logos are considered similar or identical when the Hamming distance is lower than a selected threshold(in this case I selected 5). To group the domains based on their logo similarity I took a dict, where the key was the first logo name which is not part of any values in the dict. The values of the dict are selected form the working PARQUET file. The result is saved in `hashFileNameToNames`.

# Link to Google Colab
https://colab.research.google.com/drive/1LI4M6cZ-G8LD9sw2QbaNKBJbNWsSWwaa?usp=sharing




