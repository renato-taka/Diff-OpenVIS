from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper

from odise_video.data_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
    get_openseg_labels,
)
from odise_video.modeling.wrapper.pano_wrapper import OpenPanopticInference
from detectron2.data import MetadataCatalog

dataloader = OmegaConf.create()


#augment
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(
        dataset_names="lvvis_train", filter_empty=False,
        proposal_files=None
    ),
    mapper=L(YTVISDatasetMapper)(
        is_train=True,
        augmentations=[],
        image_format="RGB",
        use_instance_mask=True,
        sampling_frame_num=2,
        sampling_frame_range=20,
        sampling_frame_shuffle=False,
        num_classes=1203,
        # COCO LSJ aug
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    
    dataset=L(get_detection_dataset_dicts)(
        dataset_names="lvvis_val",
        filter_empty=False,
        proposal_files=None,
    ),
    mapper=L(YTVISDatasetMapper)(
        is_train=False,
        augmentations=[],
        image_format="RGB",
        use_instance_mask=True,
        sampling_frame_num=2,
        sampling_frame_range=20,
        sampling_frame_shuffle=False,
        num_classes=1203,
    ),
    num_workers=4,
)

dataloader.evaluator = L(YTVISEvaluator)(
    dataset_name="lvvis_val",
    distributed=True,
    output_dir=None,
    )

dataloader.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="lvis_1203", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name="${...test.dataset.dataset_names}"),
    semantic_on=False,
    panoptic_on=False,
)

    

