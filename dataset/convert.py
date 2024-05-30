import json
import os
import numpy as np
from PIL import Image, ImageDraw

def segmentation_to_mask(segmentation, image_size):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    for segment in segmentation:
        draw.polygon(segment, fill=255)
    return mask

def process_images(json_path, output_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_folder, exist_ok=True)

    id_to_filename = {image['id']: image['file_name'] for image in data['images']}
    
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        segmentation = annotation['segmentation']
        
        if image_id in id_to_filename:
            image_name = id_to_filename[image_id]
            output_path = os.path.join(output_folder, image_name)
            
            # Assuming the image size is given in the JSON file or is known (e.g., 256x256)
            image_size = (256, 256)
            
            mask = segmentation_to_mask(segmentation, image_size)
            
            mask.save(output_path)

# Example usage
json_path = '/home/ipad_3d/hhc/WaterMask/data/UDW/annotations/val.json'
output_folder = '/home/ipad_3d/hhc/SimpleClick/dataset/water/masks/'

process_images(json_path, output_folder)
