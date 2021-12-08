import glob

from src.create_annotations import *

# Label ids of the dataset
category_ids = {
    'master_chef_can':1,
    'cracker_box':2,
    'sugar_box':3,
    'tomato_soup_can':4,
    'mustard_bottle':5,
    'tuna_fish_can':6,
    'pudding_box':7,
    'gelatin_box':8,
    'potted_meat_can':9,
    'banana':10,
    'pitcher_base':11,
    'bleach_cleanser':12,
    'bowl':13,
    'mug':14,
    'power_drill':15,
    'wood_block':16,
    'scissors':17,
    'large_marker':18,
    'large_clamp':19,
    'extra_large_clamp':20,
    'foam_brick':21
}

# Define which colors match which categories in the images
# category_colors = {
#     "(0, 0, 0)": 0, # Outlier
#     "(255, 0, 0)": 1, # Window
#     "(255, 255, 0)": 2, # Wall
#     "(128, 0, 255)": 3, # Balcony
#     "(255, 128, 0)": 4, # Door
#     "(0, 0, 255)": 5, # Roof
#     "(128, 255, 255)": 6, # Sky
#     "(0, 255, 0)": 7, # Shop
#     "(128, 128, 128)": 8 # Chimney
# }

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = [2, 5, 6]

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    for mask_image in glob.glob(maskpath + "*-label.png"):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        original_file_name = os.path.basename(mask_image).split("-")[0] + "-color.png"

        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image)
        w, h = mask_image_open.size
        
        # "images" info 
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for category_id, sub_mask in sub_masks.items():
            # "annotations" info
            polygons, segmentations = create_sub_mask_annotation(sub_mask)

            # Check if we have classes that are a multipolygon
            if True: #category_id in multipolygon_ids:
                # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)
                                
                annotation = create_annotation_format(multi_poly, segmentations, image_id, int(category_id), annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            # else:
            #     for i in range(len(polygons)):
            #         # Cleaner to recalculate this variable
            #         segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    
            #         annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
            #         annotations.append(annotation)
            #         annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id

if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    
    mask_path = "dataset/0000/" #YCB-Video Path
    
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    with open("output/{}.json".format('train'),"w") as outfile:
        json.dump(coco_format, outfile)
    
    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

