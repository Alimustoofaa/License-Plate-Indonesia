# LICENSE PLATE INDONESIA

## Introduction

>License Plate Indonesia uses Yolo to detect plates, and Tesseract for OCR

## Installation

> 1. Download repository 
        git clone https://github.com/Alimustoofaa/License-Plate-Indonesia.git
>2. Install darkflow and follow step by step to install.
        git clone https://github.com/thtrieu/darkflow.git
>3. Download and install Tesseract OCR.
         https://tesseract-ocr.github.io/tessdoc/Downloads.html
>4. Move tessdata in respository License-Plate-Indonesia to folder folder instalasi your tesseract/tessdata.
>5. Move folder models, file plate.py, perspective.py, tesseract.py to darakflow
>6. End install requirements.text
        pip install requirements.txt
>7. Edit tesseract.py in line 6 change to your instal tesseract.
>8. run plate.py
        plate.py --image=path-image.jpg (run image file)
        plate.py --videos=path-video.mp4 (run video file)
>9. FINISH
