# Modular OMR (Optical Mark Recognition) Grader

A flexible, configuration-driven Optical Mark Recognition (OMR) pipeline built with Python and OpenCV. This project automatically detects a multiple-choice answer sheet from an image, corrects its perspective warp, slices it into distinct section columns, and processes individual bubble markings to grade student answers against an answer key.

---

## 🚀 Pipeline Workflow

The script executes a sequential computer vision engine across the following stages:
https://github.com/Jibola2004/OMR_Sheet_Scanner/blob/main/docs/images/pipeline_workflow.png

---

## 🛠️ Visual Step Breakdown

### 1. Contour Detection & Perspective Warp

The pipeline isolates the largest quadrilateral boundary matching a standard 4-corner document configuration and applies an aspect-ratio accurate perspective transform to square up skewed, rotated, or angled sheets.

### 2. Slicing & Thresholding Matrix

The warped image framework is sliced into continuous independent section columns based on grid settings inside your `config.json` profile. A binary inverse thresholding operations loop shifts the data map to absolute black-and-white to evaluate filled bubble pixel volumes cleanly.

---

## ⚙️ Configuration Properties (`config.json`)

Fine-tune your edge parameters, dimensional matrices, and bounding boxes straight inside your `config.json` without rewriting core algorithm logic:

| Parameter | Type | Default Value | Description |
| --- | --- | --- | --- |
| `MARGIN_X` | `int` | `20` | Horizontal safety padding clipped from each column edge. |
| `MARGIN_Y` | `int` | `10` | Vertical safety padding clipped from top/bottom column edges. |
| `NO_OF_ANSWERS_COLUMNS` | `int` | `4` | Number of vertical question blocks/sections printed on the sheet. |
| `CROPPED_WIDTH` | `int` | `500` | The pixel width targeted during perspective transform warp. |
| `CROPPED_HEIGHT` | `int` | `700` | The pixel height targeted during perspective transform warp. |
| `NO_OF_QUESTIONS_PER_COLUMN` | `int` | `25` | Total number of horizontal questions layered per section column. |
| `NO_OF_CHOICES_PER_QUESTION` | `int` | `4` | Total choice options mapped horizontally per question (e.g., 4 = A, B, C, D). |

---

## 📂 Project Structure

```text
├── main.py                # Script execution loop coordinating step sequences
├── utils.py               # Computational computer vision matrix functions
├── answers.csv            # Generated/Read answer sheet grading reference key
├── config.json            # Pipeline thresholding & dimensional configuration map
│
├── docs/                  # Documentation assets folder
│   └── images/            # Embedded images, screenshots, and diagrams for the README
│       ├── pipeline_workflow.png
│       ├── contour_and_warp.png
        |-- column_split_threshold.png
│       └── column_split.png
        
│
└── test_images/           # Sample input folder for scanning operations
    └── real_demo_19.jpeg  # Default benchmark bubble sheet image template
  

```

---

🏃 Getting Started
1. Environment Setup & Dependencies
This project uses a localized virtual environment (.venv) to isolate dependencies. Follow these steps to initialize your environment and install the required modules via the requirements.txt manifest:

On Windows (Command Prompt / PowerShell):

```Bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies from the requirements file
pip install -r requirements.txt
On macOS / Linux (Terminal):


# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies from the requirements file
pip install -r requirements.txt
#2. Execution Routing
Once your environment is active, you can launch the pipeline.

#To process the default benchmark test image defined inside your core module setup, run:

Bash
python main.py
#To parse an alternative custom-captured bubble sheet dynamically from your test_images/ storage #pool, pass its relative file path as a terminal argument command:

Bash
python main.py test_images/your_custom_sheet.jpeg
```
---

## 📊 Evaluation Outputs

When processed, the engine tracks choice density configurations and outputs results straight to your terminal console layout:

```text
Q1  | A:  15  B: 450  C:  22  D:  10 | max: B
Q2  | A:  12  B:  18  C:  11  D:  14 | max: ?
Q3  | A: 398  B:  24  C:   8  D:  15 | max: A

================================
DONE.

Final Student Score: 2 / 3
Pipeline finished. Configuration saved safely.

```



