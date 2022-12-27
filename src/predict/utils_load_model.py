from object_detection.utils import label_map_util

ALL_MODELS = {
    "CenterNet HourGlass104 512x512": "https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1",
    "CenterNet HourGlass104 Keypoints 512x512": "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1",
    "CenterNet HourGlass104 1024x1024": "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1",
    "CenterNet HourGlass104 Keypoints 1024x1024": "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1",
    "CenterNet Resnet50 V1 FPN 512x512": "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1",
    "CenterNet Resnet50 V1 FPN Keypoints 512x512": "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1",
    "CenterNet Resnet101 V1 FPN 512x512": "https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1",
    "CenterNet Resnet50 V2 512x512": "https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1",
    "CenterNet Resnet50 V2 Keypoints 512x512": "https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1",
    "EfficientDet D0 512x512": "https://tfhub.dev/tensorflow/efficientdet/d0/1",
    "EfficientDet D1 640x640": "https://tfhub.dev/tensorflow/efficientdet/d1/1",
    "EfficientDet D2 768x768": "https://tfhub.dev/tensorflow/efficientdet/d2/1",
    "EfficientDet D3 896x896": "https://tfhub.dev/tensorflow/efficientdet/d3/1",
    "EfficientDet D4 1024x1024": "https://tfhub.dev/tensorflow/efficientdet/d4/1",
    "EfficientDet D5 1280x1280": "https://tfhub.dev/tensorflow/efficientdet/d5/1",
    "EfficientDet D6 1280x1280": "https://tfhub.dev/tensorflow/efficientdet/d6/1",
    "EfficientDet D7 1536x1536": "https://tfhub.dev/tensorflow/efficientdet/d7/1",
    "SSD MobileNet v2 320x320": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
    "SSD MobileNet V1 FPN 640x640": "https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1",
    "SSD MobileNet V2 FPNLite 320x320": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1",
    "SSD MobileNet V2 FPNLite 640x640": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1",
    "SSD ResNet50 V1 FPN 640x640 (RetinaNet50)": "https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1",
    "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)": "https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1",
    "SSD ResNet101 V1 FPN 640x640 (RetinaNet101)": "https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1",
    "SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)": "https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1",
    "SSD ResNet152 V1 FPN 640x640 (RetinaNet152)": "https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1",
    "SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)": "https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1",
    "Faster R-CNN ResNet50 V1 640x640": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
    "Faster R-CNN ResNet50 V1 1024x1024": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1",
    "Faster R-CNN ResNet50 V1 800x1333": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1",
    "Faster R-CNN ResNet101 V1 640x640": "https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1",
    "Faster R-CNN ResNet101 V1 1024x1024": "https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1",
    "Faster R-CNN ResNet101 V1 800x1333": "https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1",
    "Faster R-CNN ResNet152 V1 640x640": "https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1",
    "Faster R-CNN ResNet152 V1 1024x1024": "https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1",
    "Faster R-CNN ResNet152 V1 800x1333": "https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1",
    "Faster R-CNN Inception ResNet V2 640x640": "https://tfhub.dev/tensorflow/faster_rcnn/"
    "inception_resnet_v2_640x640/1",
    "Faster R-CNN Inception ResNet V2 1024x1024": "https://tfhub.dev/tensorflow/faster_rcnn/"
    "inception_resnet_v2_1024x1024/1",
    "Mask R-CNN Inception ResNet V2 1024x1024": "https://tfhub.dev/tensorflow/mask_rcnn/"
    "inception_resnet_v2_1024x1024/1",
}

IMAGES_FOR_TEST = {
    "Beach": "models/research/object_detection/test_images/image2.jpg",
    "Dogs": "models/research/object_detection/test_images/image1.jpg",
    # By Heiko Gorski, Source:
    # https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
    "Naxos Taverna": "https://upload.wikimedia.org/wikipedia/"
    "commons/6/60/Naxos_Taverna.jpg",
    # Source: https://commons.wikimedia.org/wiki/
    # File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
    "Beatles": "https://upload.wikimedia.org/wikipedia/commons/1/1b/"
    "The_Coleoptera_of_the_British_islands_"
    "%28Plate_125%29_%288592917784%29.jpg",
    # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/
    # File:Biblioteca_Maim%C3%B3nides,
    # _Campus_Universitario_de_Rabanales_007.jpg
    "Phones": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/"
    "Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_"
    "de_Rabanales_007.jpg/"
    "1024px-Biblioteca_Maim%C3%B3nides%2C_"
    "Campus_Universitario_de_Rabanales_007.jpg",
    # Source: https://commons.wikimedia.org/wiki/
    # File:The_smaller_British_birds_(8053836633).jpg
    "Birds": "https://upload.wikimedia.org/wikipedia/commons/0/09/"
    "The_smaller_British_birds_%288053836633%29.jpg",
}

COCO17_HUMAN_POSE_KEYPOINTS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

PATH_TO_LABELS = (
    "./models/research/object_detection/data/mscoco_label_map.pbtxt"
)
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)

model_display_name = "CenterNet HourGlass104 Keypoints 512x512"

# @param ['CenterNet HourGlass104 512x512',
# 'CenterNet HourGlass104 Keypoints 512x512',
# 'CenterNet HourGlass104 1024x1024',
# 'CenterNet HourGlass104 Keypoints 1024x1024',
# 'CenterNet Resnet50 V1 FPN 512x512',
# 'CenterNet Resnet50 V1 FPN Keypoints 512x512',
# 'CenterNet Resnet101 V1 FPN 512x512','CenterNet
# Resnet50 V2 512x512',
# 'CenterNet Resnet50 V2 Keypoints 512x512',
# 'EfficientDet D0 512x512','EfficientDet D1 640x640',
# 'EfficientDet D2 768x768','EfficientDet D3 896x896',
# 'EfficientDet D4 1024x1024','EfficientDet D5 1280x1280',
# 'EfficientDet D6 1280x1280','EfficientDet D7 1536x1536',
# 'SSD MobileNet v2 320x320','SSD MobileNet V1 FPN 640x640',
# 'SSD MobileNet V2 FPNLite 320x320',
# 'SSD MobileNet V2 FPNLite 640x640',
# 'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)',
# 'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)',
# 'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)',
# 'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)',
# 'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)',
# 'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)',
# 'Faster R-CNN ResNet50 V1 640x640',
# 'Faster R-CNN ResNet50 V1 1024x1024',
# 'Faster R-CNN ResNet50 V1 800x1333',
# 'Faster R-CNN ResNet101 V1 640x640','
# Faster R-CNN ResNet101 V1 1024x1024',
# 'Faster R-CNN ResNet101 V1 800x1333',
# 'Faster R-CNN ResNet152 V1 640x640',
# 'Faster R-CNN ResNet152 V1 1024x1024',
# 'Faster R-CNN ResNet152 V1 800x1333',
# 'Faster R-CNN Inception ResNet V2 640x640',
# 'Faster R-CNN Inception ResNet V2 1024x1024',
# 'Mask R-CNN Inception ResNet V2 1024x1024']
