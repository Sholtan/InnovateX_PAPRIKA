Overview

To run the model use two notebooks:

predict_for_files.ipynb – runs object detection on PDF files, extracts each page as an image, and saves prediction results (bounding boxes, classes, confidence).

draw_and_save.ipynb – loads prediction results and draws bounding boxes on the original PDF pages, saving visualized outputs.

The correct execution order is:

Run predict_for_files.ipynb

Then run draw_and_save.ipynb

1. Running predict_for_files.ipynb
Purpose

Reads multiple PDF files from a directory.

Converts each PDF page into an image.

Runs your YOLO model on them.

Saves predictions (usually as JSON files or text files per page).

Requirements

Python 3.8+

Ultralytics YOLO

PyMuPDF or pdf2image

OpenCV

Numpy

Steps

Open the notebook:

predict_for_files.ipynb


Configure input/output directories:

pdf_dir – path to PDF files

output_dir – where predictions will be saved

model_path – path to your trained YOLO model

Run all cells from top to bottom.

After execution, you should have:

Extracted images of PDF pages

Prediction files (bounding boxes and classes)

2. Running draw_and_save.ipynb
Purpose

Reads previously created prediction results.

Draws bounding boxes on each corresponding PDF page image.

Saves the annotated images or PDF files.

Requirements

OpenCV

JSON

Same image output directory from previous step

Steps

Open:

draw_and_save.ipynb


Configure paths:

img_dir – directory where extracted page images were saved

json_path – predictions file from predict_for_files.ipynb

save_dir – where to save visualized outputs

Run all cells from top to bottom.

The notebook will output annotated images (or PDFs) with bounding boxes for signatures, seals, codes, etc.