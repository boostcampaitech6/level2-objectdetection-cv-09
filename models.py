import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def Faster_R_CNN_ResNet50_FPN(num_classes, pretrained=True):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model