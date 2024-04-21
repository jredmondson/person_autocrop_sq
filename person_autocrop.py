import os
import cv2
import numpy as np
import tensorflow as tf
import requests
import re

# Download model if not exist
def download_model():
    url = "https://drive.google.com/uc?id=1Ml260620LIKa-OrWqdzv_z99NJKixt3W"
    output_path = "ssd_mobilenetv2_coco/saved_model.pb"

    response = requests.get(url, stream=True)

    with open(output_path, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output_file.write(chunk)

# Explicitly specify the GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize model variable
model = None

def create_folder(output_directory):
    # Check if the folder exists
    if not os.path.exists(output_directory):
        try:
            # Create the folder if it doesn't exist
            os.makedirs(output_directory)
            print(f"LOG: Folder '{output_directory}' created successfully.")
        except OSError as e:
            print(f"LOG: Error creating folder '{output_directory}': {e}")

# Function to perform object detection and crop the image
def object_detection(input_image_path, output_directory, output_format):

    # Read the input image
    image = cv2.imread(input_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0).astype(np.uint8)

    # Perform inference
    detections = model.signatures["serving_default"](tf.constant(image_expanded))

    # Extract coordinates of detected persons
    detection_classes = detections['detection_classes'][0].numpy()
    detection_boxes = detections['detection_boxes'][0]
    person_indices = tf.where(tf.equal(detection_classes, 1))[:, 0]
    person_coords = tf.gather(detection_boxes, person_indices).numpy()

    # Crop the original image based on the detected persons' coordinates
    for i, coords in enumerate(person_coords):
        ymin, xmin, ymax, xmax = coords
        
        
        # Ensure valid coordinates
        if 0 <= ymin < ymax <= image.shape[0] and 0 <= xmin < xmax <= image.shape[1]:
        
            #print(f"LOG: image.shape[0]={image.shape[0]}, image.shape[1]={image.shape[1]}")
            #print(f"LOG: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
            
            # We need a square crop
            ymin = int(ymin * image.shape[0]);
            ymax = int(ymax * image.shape[0]);
            
            xmin = int(xmin * image.shape[1]);
            xmax = int(xmax * image.shape[1]);
            
            #print(f"LOG: after*image[shape] xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

            width = xmax - xmin
            height = ymax - ymin
            
            length = int(max(width, height))
            
            #print(f"LOG: width={width}, height={height}, length={length}")
            
            if width < height:
              center = int((xmax + xmin)/2)
              growth = int(length/2)
              
              xmin = int(center - growth)
              xmax = xmin + length;
              
              print(f"LOG: center={center}, growth={growth}, new_x={xmin}-{xmax}")

              if xmin < 0:
                xmin = 0
                xmax = length
              
              if xmax > image.shape[1]:
                xmin = image.shape[1] - length - 1
                xmax = image.shape[1] - 1
                
            elif height < width:
              center = int((ymax + ymin)/2)
              growth = int(length/2)
              
              ymin = int(center - growth)
              ymax = ymin + length;
              
              #print(f"LOG: center={center}, growth={growth}, new_y={ymin}-{ymax}")

              if ymin < 0:
                ymin = 0
                ymax = length
              
              if ymax > image.shape[0]:
                ymin = image.shape[0] - length - 1
                ymax = image.shape[0] - 1
              
            #xmax = xmin + length;
            #ymax = ymin + length;
        
            #print(f"LOG: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

        
            cropped_person = image[ymin:ymax, xmin:xmax, :]

            # Check if the cropped_person array is not empty
            if not cropped_person.size == 0:
                # Save the cropped person image with correct color space conversion
                try:
                    image_name = os.path.basename(input_image_path)
                    filename, extension = os.path.splitext(image_name)
                    
                    p = re.compile('(.*)_([0-9]+)')
                    m = p.match(filename)
                    filename = m.group(1) + '_person_' + str(i) + '_' + m.group(2);
                    
                    # Save the image using cv2.imwrite()
                    if output_format == "0":
                        cv2.imwrite(f"{output_directory}/{filename}.png", cropped_person)
                        print(f"LOG: Image saved successfully to '{output_directory}/{filename}.png'.")
                    elif output_format == "1":
                        cv2.imwrite(f"{output_directory}/{filename}.jpg", cropped_person)
                        print(f"LOG: Image saved successfully to '{output_directory}/{filename}.jpg'.")
                    elif output_format == "2":
                        cv2.imwrite(f"{output_directory}/{filename}.webp", cropped_person)
                        print(f"LOG: Image saved successfully to '{output_directory}/{filename}.webp'.")
                    elif output_format == "3":
                        cv2.imwrite(f"{output_directory}/{filename}.bmp", cropped_person)
                        print(f"LOG: Image saved successfully to '{output_directory}/{filename}.bmp'.")
                    else:
                        cv2.imwrite(f"{output_directory}/{image_name}", cropped_person)
                        print(f"LOG: Image saved successfully to '{output_directory}/{image_name}'.")
                except Exception as e:
                    print(f"LOG: Error saving image: {e}")
            else:
                print(f"LOG: Warning: Cropped person {i + 1} is empty.")

if __name__ == "__main__":
    import sys
    
    if os.path.exists("ssd_mobilenetv2_coco/saved_model.pb"):
        print("LOG: SSD Mobilenet V2 COCO model found.")
    else:
        print("LOG: SSD Mobilenet V2 COCO model not found. Downloading...")
        download_model()
        print("LOG: SSD Mobilenet V2 COCO model downloaded.")

    model = tf.saved_model.load("ssd_mobilenetv2_coco")
    
    input_image_folder = input(">> Enter input image path: ")
    output_directory = input(">> Enter output image path: ")

    print("Select output image format:")
    print("[0] PNG")
    print("[1] JPG")
    print("[2] WEBP")
    print("[3] BMP")
    print("[4 or out of range] Input format will be preserved")
    output_format = input("Enter [0-4]: ")
    
    create_folder(output_directory)

    print("LOG: Running...")
    # Process all images in the input folder
    for filename in os.listdir(input_image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp") or filename.endswith(".bmp"):
            input_image_path = os.path.join(input_image_folder, filename)
            object_detection(input_image_path, output_directory, output_format)
    print("LOG: Done.")
