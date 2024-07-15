import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from PIL import Image, ImageDraw
import csv
import cv2
import os
import matplotlib.pyplot as plt
from deepgaze_pytorch import DeepGazeIII
from scipy.ndimage import gaussian_filter
import math
DEVICE = 'cpu'

def is_contour_inside(contour1, contour2):
    rect1 = cv2.boundingRect(contour1)
    rect2 = cv2.boundingRect(contour2)
    
    if (rect1[0] > rect2[0] and
        rect1[1] > rect2[1] and 
        rect1[0] + rect1[2] < rect2[0] + rect2[2] and
        rect1[1] + rect1[3] < rect2[1] + rect2[3]):
        return True
    return False

def remove_nested_contours(contours):
    to_remove = set()
    for i in range(len(contours)):
        for j in range(len(contours)):
            if i != j and is_contour_inside(contours[i], contours[j]):
                to_remove.add(i)
                break
    return [contours[i] for i in range(len(contours)) if i not in to_remove]

def merge_small_with_large_contours(contours, min_area_threshold=500000):
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_threshold]
    small_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < min_area_threshold]
    
    for small_cnt in small_contours:
        min_distance = float('inf')
        closest_large_contour = None
        sx, sy, sw, sh = cv2.boundingRect(small_cnt)
        for large_cnt in large_contours:
            lx, ly, lw, lh = cv2.boundingRect(large_cnt)
            distance = min(abs(sy - (ly + lh)), abs((sy + sh) - ly))  # vertical distance
            if distance < min_distance:
                min_distance = distance
                closest_large_contour = large_cnt
        if closest_large_contour is not None:
            merged_contour = np.concatenate((closest_large_contour, small_cnt))
            large_contours.append(merged_contour)
            large_contours.remove(closest_large_contour)

    return large_contours

def process_image(image_path, number_scanpaths):
    # Load the image
    image = np.array(image_path)
    original_image = image.copy()
    min_contour_area = 1000

    # Check image dimensions
    width,height = image.shape[:2]
    if width <= 3840 or height <= 2160:
        print("Image is not larger than 3840x2160. No sectioning will be applied.")
        mask,box=predict_scanpaths(original_image, number_scanpaths)
        return mask,box 
    else:

        # Convert the image to grayscale
        lab_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_segmented_image = lab_image_gray

        def merge_regions(contours, max_distance=400, mean_threshold=100):
            final_merged_contours = []
            final_merged_indices = set()
            
            for idx, cnt1 in enumerate(contours):
                if idx not in final_merged_indices:
                    merged_contour = cnt1
                    x1, y1, w1, h1 = cv2.boundingRect(cnt1)
                    bottom1 = y1 + h1  # Bottom y-coordinate of cnt1
                    top1 = y1  # Top y-coordinate of cnt1
                    
                    for j, cnt2 in enumerate(contours[idx + 1:]):
                        k = idx + j + 1
                        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
                        bottom2 = y2 + h2  # Bottom y-coordinate of cnt2
                        top2 = y2  # Top y-coordinate of cnt2
                        
                        # Check if contours are in the same row
                        if abs(y1 - y2) <= max_distance:  # Check proximity in the y-axis
                            # Calculate the mean value of each region
                            mask1 = np.zeros_like(gray_segmented_image)
                            mask2 = np.zeros_like(gray_segmented_image)
                            cv2.drawContours(mask1, [cnt1], 0, 255, cv2.FILLED)
                            cv2.drawContours(mask2, [cnt2], 0, 255, cv2.FILLED)
                            mean_val1 = cv2.mean(image, mask=mask1)
                            mean_val2 = cv2.mean(image, mask=mask2)
                            mean_diff = np.abs(np.array(mean_val1[:3]) - np.array(mean_val2[:3])).mean()
                            if mean_diff < mean_threshold:
                                merged_contour = np.concatenate((merged_contour, cnt2))
                                final_merged_indices.add(k)
                    final_merged_contours.append(merged_contour)
                    final_merged_indices.add(idx)
            
            return final_merged_contours

        # Apply bilateral filter to the grayscale segmented image
        blurred = cv2.bilateralFilter(gray_segmented_image, d=9, sigmaColor=75, sigmaSpace=75)

        # Perform edge detection using Canny
        edges = cv2.Canny(blurred, 100, 200)

        # Apply dilation and erosion to connect nearby contours
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=80)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Find contours in the eroded image
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Merge similar and nearby regions within the same row
        merged_contours = merge_regions(contours)


        mask_image=original_image.copy()
        box_image=original_image.copy()

        # Process each section independently
        for cnt in merged_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            section_image = original_image[y:y+h, x:x+w]  # Extract section from the original image
            section_image = cv2.cvtColor(section_image, cv2.COLOR_BGR2RGB)
            # Call predict_scanpaths to get mask and box for the section
            mask, box = predict_scanpaths(section_image, number_scanpaths)
            mask=cv2.resize(mask,(w,h))
            box=cv2.resize(box,(w,h))



            # Composite the mask and box onto the original image
            mask_image[y:y+h, x:x+w] = mask
            box_image[y:y+h, x:x+w] = box # You can change this to add transparency or overlay differently

        return mask_image, box_image


def predict_scanpaths(image_path, number_scanpaths):
    save_filepath = r''

    # Read the image
    image=Image.fromarray(image_path)
    image_or=image
    w, h = image.size
    
    # Resize the image
    image = image.resize((int(w/2.5), int(h/2.5)))
    w, h = image.size
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Load centerbias template
    # Expand ~ to the user's home directory
    centerbias_template_path = os.path.expanduser('centerbias_mit1003.npy')

# Load the .npy file
    centerbias_template = np.load(centerbias_template_path)
    #centerbias_template = np.load(r'~/area_interest_attention_heat_map(deepgaze)/deepgaze_pytorch/deepgaze_pytorch/centerbias_mit1003.npy')
    
    # Rescale centerbias to match image size
    centerbias = zoom(centerbias_template, (h / centerbias_template.shape[0], w / centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)
    
    # Initialize fixations
    fixations_x = [w // 2]
    fixations_y = [h // 2]
    fixations = 1
    
    scanpaths = []
    
    # Predict scanpaths
    while len(scanpaths) < number_scanpaths:
        # Create circular mask for inhibition of return
        radius = int(0.2 * min(w, h))
        mask = create_circular_mask(h, w, fixations_x, fixations_y, radius)
        
        # Predict next fixation
        brightest_pixel = prediction(image_np * mask.unsqueeze(2).numpy().astype('uint8'), fixations_x, fixations_y, centerbias, fixations, mask)
        
        # Add predicted fixation to the list
        fixations_x.append(brightest_pixel[3])
        fixations_y.append(brightest_pixel[2])
        
        # Increment fixations parameter
        if fixations <= 3:
            fixations += 1
        
        # Append the predicted scanpath
        scanpaths.append((brightest_pixel[3], brightest_pixel[2]))
    
    # Generate attention heatmap
    heatmap = generate_attention_heatmap(w, h, scanpaths, save_filepath)
    mask = apply_heatmap_mask(image_or, heatmap, save_filepath)
    box = draw_scanpath_with_significance_on_image(image_gray, scanpaths, save_filepath)
    
    return  np.array(org_img(mask)), np.array(org_img(box))


def prediction(image, fixations_x, fixations_y, centerbias, fixations, mask):
    # Location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
    fixation_history_x = np.array(fixations_x)
    fixation_history_y = np.array(fixations_y)

    model = DeepGazeIII(fixations, pretrained=True).to(DEVICE)

    image_tensor = torch.tensor([image[:,:,:3].transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
    x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
    y_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)

    log_density_prediction = (100 + model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)) \
                                * mask.to(DEVICE).unsqueeze(0).unsqueeze(0) - (1 - mask.to(DEVICE)) * 1000

    # Find the brightest pixel in the probability map
    brightest_pixel = (log_density_prediction==torch.max(log_density_prediction)).nonzero()[0].detach().cpu().numpy()

    return brightest_pixel

def create_circular_mask(h, w, fixations_x, fixations_y, radius):
    # Get the circular mask
    mask = torch.zeros(h, w)
    Y, X = np.ogrid[:h, :w]
    for i in range(len(fixations_x)):
        dist = np.sqrt((X - fixations_x[i])**2 + (Y - fixations_y[i])**2)
        mask = torch.maximum(mask, torch.from_numpy(dist <= radius) * (1 - 1/10 * (len(fixations_x) - i - 1)))

    return 1 - mask

def generate_attention_heatmap(width, height, scanpaths, save_filepath=None, blur_radius=120):
    # Create an empty canvas for the heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Iterate through each point in the scanpaths
    max_order = len(scanpaths)
    for order, (x, y) in enumerate(scanpaths):
        # Create a meshgrid of coordinates
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate distance from each pixel to the current point
        distance = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        
        # Adjust blur radius based on order
        adjusted_blur_radius = blur_radius * (max_order - order) / max_order
        
        # Create a mask to identify pixels within the adjusted blur radius
        mask = distance <= adjusted_blur_radius
        
        # Calculate intensity based on distance and order within the blur radius
        intensity = (1 - distance[mask] / adjusted_blur_radius) * (max_order - order) / max_order  # Linearly decreasing intensity
        
        # Add intensity to the heatmap
        heatmap[mask] += intensity

    # Normalize the heatmap
    heatmap /= np.max(heatmap)
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap = np.array(heatmap_image)
    return heatmap



def read_img(image):
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    
    w, h = image.size
        
        # Resize the image
    image = image.resize((int(w/2.5), int(h/2.5)))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def org_img(image):
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(image))
    
    w, h = image.size
    
    # Resize the image
    resized_image = image.resize((int(w*2.5), int(h*2.5)))
    resized_image=resized_image.convert('RGB')
    return resized_image


def apply_heatmap_mask(image, heatmap,save_filepath=None ,colormap=cv2.COLORMAP_JET, alpha=0.4):
    # Read the image
    image=read_img(image)

    # Apply the colormap to the heatmap

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)

    # Convert the colormap from BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Create a binary mask where the heatmap intensity is non-zero
    mask = (heatmap > 0).astype(np.uint8)

    # Apply Gaussian blur to the mask
    blurred_mask = cv2.GaussianBlur(mask.astype(float), (5, 5), 4)

    # Where the mask is zero, we want the original image to show through
    heatmap_color[blurred_mask == 0] = image[blurred_mask == 0]

    # Blend the heatmap and the original image
    blended = cv2.addWeighted(image, alpha, heatmap_color, 1 - alpha, 0)
    return blended





def draw_scanpath_with_significance_on_image(image, scanpaths, save_filepath=None):
    # Convert NumPy array to PIL image
    max_order = len(scanpaths)
    image_pil = Image.fromarray(image)

    # Convert the image to RGBA mode to allow transparency
    image_rgba = image_pil.convert('RGBA')
    draw = ImageDraw.Draw(image_rgba)
    
    for order, (x, y) in enumerate(scanpaths):
        # Calculate significance percentage based on order
        significance = (max_order - order) / max_order
        
        # Calculate box size based on significance
        box_size = int(50 * significance)  # Adjust box size based on significance
        half_box = box_size // 2
        
        # Calculate coordinates of the box
        x1 = x - half_box
        y1 = y - half_box
        x2 = x + half_box
        y2 = y + half_box
        
        # Draw rectangle with transparency
        draw.rectangle([x1, y1, x2, y2], outline='black', fill=None)
        
        # Draw text indicating significance
        draw.text((x - 50, y - 50), f'{significance:.2%}', fill='red')
    

    return image_rgba




import json
import base64
import os
import tempfile
from io import BytesIO
def handler(event, context):
    try:
        # Parse the input
        body = json.loads(event['body'])
        image_base64 = body['image']
        number_scanpaths = int(body['number_scanpaths'])

        # Decode the base64 image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Save the image to a temporary file
        #temp_image_file_path = image_file(image)
        # Call the predict_scanpaths function
        mask, box = process_image(image, number_scanpaths)

        # Encode images to base64
        mask_data = encode_image_to_base64(mask)
        box_data = encode_image_to_base64(box)

        # Remove the temporary image file
        #os.remove(temp_image_file_path)

        response = {
            'statusCode': 200,
            'body': json.dumps({
                'msg': 'success',
                'mask': mask_data,
                'box': box_data,
            }),
            'headers': {
                'Content-Type': 'application/json',
            },

        }

        return response

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
            },
        }
def encode_image_to_base64(image: Image.Image) -> str:

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert the PIL Image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Encode the image bytes to a base64 string
    encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_data

