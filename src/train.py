import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T
import argparse
from omegaconf import OmegaConf

# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

# trainer - DefaultTrainer를 상속
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok = True)
            output_folder = './output_eval'
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/train.yaml"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
        
    # config 불러오기
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(configs['TRAIN']['config']))

    # config 수정하기
    cfg.DATASETS.TRAIN = (configs['TRAIN']['name'],)
    cfg.DATASETS.TEST = (configs['TEST']['name'],)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(configs['TRAIN']['config'])

    cfg.SOLVER.IMS_PER_BATCH = configs['TRAIN']['batch_size']
    cfg.SOLVER.BASE_LR = configs['TRAIN']['lr']
    cfg.SOLVER.MAX_ITER = 15000
    cfg.SOLVER.STEPS = (configs['TRAIN']['stepone'], configs['TRAIN']['steptwo'])
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000

    cfg.OUTPUT_DIR = configs['TRAIN']['output_dir']

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.TEST.EVAL_PERIOD = 3000

    # Register Dataset
    try:
        register_coco_instances(configs['TRAIN']['name'], {}, configs['DATA']['annotation'], configs['DATA']['data_dir'])
    except AssertionError:
        pass
    try:
        register_coco_instances(configs['TEST']['name'], {}, configs['TEST']['annotation'], configs['DATA']['data_dir'])
    except AssertionError:
        pass

    MetadataCatalog.get(configs['TRAIN']['name']).thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    # train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()