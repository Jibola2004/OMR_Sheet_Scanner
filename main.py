import sys
import cv2


from utils import generate_random_answer_choice, get_answer_choices_from_csv, grade_answer_sheet, load_config_file, reorder, crop_from_contour, save_config_file, threshold_answer_boxes, get_pixel_counts, load_image, process_images, \
    find_largest_rectangle_contour, get_warped_rectangle, split_into_answer_boxes, get_all_pixel_counts

################## PIPELINE FLOW #######################
## Load → Detect → Warp → Split → Threshold → Analyze ##
########################################################
#################################################
############## LOAD CONFIGURATION ###############
#################################################
config = load_config_file("config.json")

# Map the JSON dictionary values to your pipeline parameters
MARGIN_X = config["MARGIN_X"]
MARGIN_Y = config["MARGIN_Y"]
NO_OF_ANSWERS_COLUMNS = config["NO_OF_ANSWERS_COLUMNS"]
CROPPED_WIDTH = config["CROPPED_WIDTH"]
CROPPED_HEIGHT = config["CROPPED_HEIGHT"]
NO_OF_QUESTIONS_PER_COLUMN = config["NO_OF_QUESTIONS_PER_COLUMN"]
NO_OF_CHOICES_PER_QUESTION = config["NO_OF_CHOICES_PER_QUESTION"]

#################################################



default_image_path ="test_image/real_demo_19.jpeg"
if len(sys.argv)>1:
    default_image_path = sys.argv[1]

img =load_image(default_image_path)
contours = process_images(img)
biggest_rectangle = find_largest_rectangle_contour(contours)
if biggest_rectangle is None:
    print("No rectangle detected")
    exit()
_,cropped= get_warped_rectangle(img, biggest_rectangle, width=CROPPED_WIDTH, height=CROPPED_HEIGHT)
if cropped is None:
    print("Warp failed")
    exit()
boxs =split_into_answer_boxes(cropped,margin_x=MARGIN_X,margin_y=MARGIN_Y,no_of_sections=NO_OF_ANSWERS_COLUMNS,offset_x=60,offset_y=0)
print("Length of Boxes:", len(boxs))
image_Thresholds = threshold_answer_boxes(boxs)
#get_pixel_counts(image_Thresholds[0],num_questions=NO_OF_QUESTIONS_PER_COLUMN,no_of_answers=NO_OF_CHOICES_PER_QUESTION)
questions_choices=get_all_pixel_counts(image_Thresholds,no_of_answers=NO_OF_CHOICES_PER_QUESTION,
                     no_of_sections=NO_OF_ANSWERS_COLUMNS,
                     num_of_questions_per_section=NO_OF_QUESTIONS_PER_COLUMN )

generate_random_answer_choice(number_of_choice=100)
answer_choices=get_answer_choices_from_csv()


grade=grade_answer_sheet(questions_choices,answer_choices)
print(f"Final Student Score: {grade} / {len(questions_choices)}")
# --- SAVE CONFIGURATION AT THE END OF THE LINE ---
# If your script modified any values inside the 'config' dictionary during runtime,
# unpacking it here with ** will save those updated values back to the disk.
save_config_file("config.json", **config)

print("Pipeline finished. Configuration saved safely.")
cv2.waitKey(0)
cv2.destroyAllWindows()