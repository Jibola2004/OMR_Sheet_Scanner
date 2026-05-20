import csv
import numpy as np
import cv2
import os
import random 
import json

def reorder(points):
    """
        Description:
            Reorders 4 unordered points of a contour into a consistent
            rectangle format required for perspective transformation.

            The output order is:
                [top-left, top-right, bottom-right, bottom-left]

            This is necessary because OpenCV contour points are not
            guaranteed to be in any specific order.

        Input:
            points (numpy.ndarray):
                A 4x2 array containing x, y coordinates of a rectangle contour.
                Example shape: (4, 2)
                Example:
                    [[x1, y1],
                     [x2, y2],
                     [x3, y3],
                     [x4, y4]]

        Return:
            numpy.ndarray:
                A 4x2 array of reordered points in the following order:
                [top-left, top-right, bottom-right, bottom-left]

    """
    points = points.reshape((4, 2))
    new = np.zeros((4, 2), dtype=np.float32)

    add = points.sum(1)
    diff = np.diff(points, axis=1)

    new[0] = points[np.argmin(add)]   # top-left
    new[2] = points[np.argmax(add)]   # bottom-right
    new[1] = points[np.argmin(diff)]  # top-right
    new[3] = points[np.argmax(diff)]  # bottom-left

    return new

def crop_from_contour(image, contour,width,height):
    """
        Description:
            Extracts and straightens a rectangular region from an image
            using a contour by applying a perspective transform.

            This is commonly used for document scanning, OMR sheets,
            and extracting warped rectangles from images.

        Input:
            image (numpy.ndarray):
                The original input image (BGR format from OpenCV).

            contour (numpy.ndarray):
                A contour representing a quadrilateral shape.
                Expected shape: (4, 1, 2) or (4, 2)

            width (int, optional):
                Width of the output cropped image. Default is 600.

            height (int, optional):
                Height of the output cropped image. Default is 800.

        Return:
            numpy.ndarray:
                The warped (cropped and straightened) image.
        """

    points = contour.reshape(4, 2).astype(np.float32)
    rect = reorder(points)



    dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped

def split_boxes(img, rows, cols):
    h, w = img.shape[:2]

    h = h - (h % rows)
    w = w - (w % cols)

    img = img[:h, :w]

    rows_split = np.vsplit(img, rows)

    boxes = []



    for r in rows_split:
        cols_split = np.hsplit(r, cols)

        for box in cols_split:
            boxes.append(box)

    return boxes

def load_image(image_path):
    """Load an image from a given path and return it."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Path does not exist: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"File exists but could not be read as an image: {image_path}")

    return img

def process_images(img):
    """Accepts an image, processes it, and returns contours."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur + edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_largest_rectangle_contour(contours):
    """Finds the largest rectangle contour or returns None if it does not exist."""
    max_area = 0
    biggest_rect = None

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if contour has 4 sides (rectangle)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)

            if area > max_area:
                max_area = area
                biggest_rect = approx

    return biggest_rect

def get_warped_rectangle(img, biggest_rect, width=500, height=700):
    max_area = 0
    if biggest_rect is not None:
        # Draw contour
        cv2.drawContours(img, [biggest_rect], -1, (0, 255, 0), 3)
        

        # Reorder points
        pts = reorder(biggest_rect)

        # Perspective transform
        destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pts, destination)
        result = cv2.warpPerspective(img, matrix, (width, height))

        # Crop (your existing method)
        cropped = crop_from_contour(img, biggest_rect, width=width, height=height)
        cv2.imshow("cropped", cropped)
        return result, cropped

    return None, None

def split_into_answer_boxes(cropped,margin_x=20,margin_y=10,no_of_sections=4,offset_x=60,offset_y=0):
    """
    Divides the pre-cropped image into vertical sections (answer boxes)
    by calculating equal widths and applying horizontal and vertical margins.
    Each section is sliced from the image and stored in a list.
    """


    h, w = cropped.shape[:2]
    box_width = w // no_of_sections

    boxs = []

    for i in range(no_of_sections):
        x_start = i * box_width + margin_x +offset_x
        x_end = (i + 1) * box_width - margin_x

        y_start = margin_y + offset_y
        y_end = h - margin_y

        crop_box = cropped[y_start:y_end, x_start:x_end]

        boxs.append(crop_box)
        #cv2.imshow(f"box {i + 1}", crop_box)


    return boxs

def threshold_answer_boxes(boxes):
    """
Accepts a list of image boxes, converts each box to grayscale, and applies
binary inverse thresholding to produce a list of black-and-white images
suitable for further analysis (e.g., bubble detection in OMR processing).
Returns the list of threshold images.
"""
    threshold_boxes = []
    if len(boxes) == 0:
        return threshold_boxes


    for box in boxes:
        imageWrapGray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        imgThreshold = cv2.threshold(imageWrapGray, 147, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Threshold", imgThreshold)
        threshold_boxes.append(imgThreshold)

    return threshold_boxes

def get_pixel_counts(column_img, num_questions=5, no_of_answers=4, section_no=0):
    """
    Computes pixel counts for each answer option per question in a column image.
    Applies a threshold to determine the selected answer or mark it as uncertain ("?").
    """

    boxes = split_boxes(column_img, rows=num_questions, cols=no_of_answers)

    expected = num_questions * no_of_answers

    if len(boxes) != expected:
        print(f"Warning: expected {expected} boxes but got {len(boxes)}")
        boxes = boxes[:len(boxes)]


    max_results=[] # list of max selection question choice = {S/N: question_number,"choice":"A or B or C or D"}
    for i in range(num_questions):

        counts = {}
        max_count = 0
        max_index = -1

        for j in range(no_of_answers):

            idx = i * no_of_answers + j

            if idx >= len(boxes):
                continue

            box = boxes[idx]
            count = cv2.countNonZero(box)

            option = chr(65 + j)  # A, B, C, D...
            counts[option] = count

            # find maximum normally
            if count > max_count:
                max_count = count
                max_index = j

        # apply threshold AFTER finding max
        if max_count < 200:
            selected = "?"
            max_index = -1
        else:
            selected = chr(65 + max_index)
        
        #dict_value={"S/N":section_no * num_questions + i + 1,"choice":selected}
        max_results.append(max_index)
        print(
            f"Q{section_no * num_questions + i + 1} | "
            f"A:{counts.get('A',0):4} B:{counts.get('B',0):4} "
            f"C:{counts.get('C',0):4} D:{counts.get('D',0):4} | "
            f"max: {selected}"
        )
    return max_results

def get_all_pixel_counts(images, no_of_sections=4, num_of_questions_per_section=25, no_of_answers=4):
    """
    Processes multiple threshold image sections (columns) and computes pixel counts
    for each question across all sections. Each section is analyzed sequentially, and
    question numbering is adjusted based on its position to maintain a continuous
    global index. Validates input to ensure the expected number of sections is provided.
    Prints the detected answer counts per question and signals completion at the end.
    """
    if images is None:
        raise ValueError("No images provided")
    if len(images) !=4:
        raise ValueError(f"Expected 4 images but got {len(images)}")

    i =0
    max_results=[]
    for image in images:
        result=get_pixel_counts(image,num_questions=num_of_questions_per_section,no_of_answers=no_of_answers,section_no=i)
        i+=1
        max_results.extend(result)

    print("\n================================\n")
    print("DONE.\n")
    
    return max_results

def get_answer_choices_from_csv(file_path="answers.csv"):
    """
    Reads answer choices from a CSV file and converts them into numerical indices.

    The CSV file must contain a column named 'answer_choice' with values
    representing choices (e.g., 'A', 'B', 'C', 'D').

    Each choice is mapped as follows:
        A -> 0
        B -> 1
        C -> 2
        D -> 3
        E -> 4

    Args:
        file_path (str): Path to the CSV file. Defaults to 'answer.csv'.

    Returns:
        list[int]: A list of integer indices corresponding to the answer choices.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If 'answer_choice' column is missing or contains invalid values.
    """

    result = []
    choice_index = {"A": 0, "B": 1, "C": 2, "D": 3,"E":4}

    try:
        with open(file_path, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            
            
           
            if "S/N" not in reader.fieldnames:
                raise ValueError("CSV must contain 'S/N' column")
            if "Answer_Choice" not in reader.fieldnames:
                raise ValueError("CSV must contain 'Answer_Choice' column")

            for i, row in enumerate(reader, start=1):
                raw_choice = row.get("Answer_Choice", "").strip().upper()

                if raw_choice not in choice_index:
                    raise ValueError(
                        f"Invalid answer '{raw_choice}' at row {i}. "
                        f"Expected one of {list(choice_index.keys())}"
                    )

                result.append(choice_index[raw_choice])

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return result

def generate_random_answer_choice(number_of_choice: int, choices=['A', 'B', 'C', 'D']):
    """
       Generates a random list of multiple-choice answers and exports them to a CSV file.

       Validates inputs to ensure a valid entry count and choice list, creates a 
       sequential dataset, and writes the results to 'answers.csv' with columns 
       for serial number (S/N) and the selected choice.

       Args:
          number_of_choice (int): The total number of answers to generate.
          choices (list of str): The available options to pick from. Defaults to ['A', 'B', 'C', 'D'].

       Raises:
          ValueError: If number_of_choice is not a positive integer, or if 
          choices contains non-string elements.
    """
    if not isinstance(number_of_choice, int):
        raise ValueError("parameter expected is of type integer")
    if number_of_choice < 0:
        raise ValueError("Invalid parameter, number_of_choice must be >= 0")
    if not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"Invalid Choice, each choice expected to be a string in {choices}")
    
    result = []
    for index in range(1, number_of_choice + 1):
        # random.choice() returns a single element string instead of a list like random.choices()
        random_letter = random.choice(choices) 
        
        # Keys must exactly match the CSV fields defined below
        dict_value = {"S/N": index, "Answer_Choice": random_letter}
        result.append(dict_value)
        
    try:
        with open('answers.csv', 'w', newline='') as file: # newline='' prevents blank rows in Windows
            fieldnames = ['S/N', 'Answer_Choice']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(result) # Fixed from .rows() to .writerows()
            
        print("file created successfully.")       
            
    except Exception as ex:
        # Never pass silently! Printing the error helps you debug.
        print(f"An error occurred: {ex}") 


def save_config_file(filename: str = "config.json", **kwargs):
    """Saves the provided configuration arguments to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            
            json.dump(kwargs, file, indent=4)
        print(f"Configuration saved successfully to '{filename}'.")
    except Exception as e:
        print(f"Error saving config file: {e}")

        
def load_config_file(filename: str = "config.json"):
    """
    Loads configuration data from a JSON file. 
    If the file does not exist, creates it with default values.
    """
    print("..... loading config file ......\n")
    
    # Define your default fallback settings
    default_config = {
        "MARGIN_X": 20,
        "MARGIN_Y": 10,
        "NO_OF_ANSWERS_COLUMNS": 4,
        "CROPPED_WIDTH": 500,
        "CROPPED_HEIGHT": 700,
        "NO_OF_QUESTIONS_PER_COLUMN": 25,
        "NO_OF_CHOICES_PER_QUESTION": 4
    }
    
    # Check if the file exists using the os module
    if not os.path.exists(filename):
        print(f"'{filename}' not found. Creating a new one with defaults...")
        # Use our save function to generate the default file
        save_config_file(filename, **default_config)
        return default_config

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
            return config_data
    except Exception as e:
        print(f"Error loading config file: {e}")
        return default_config


def grade_answer_sheet(question_choices,answer_choices,percentage=False):
    score =0
    score = sum(1 for q, a in zip(question_choices, answer_choices) if q == a)
    
    if percentage  == True:
        return score/len(question_choices) if len(question_choices) >0 else 0
    return score
    


