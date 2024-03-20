# CHECKPOINT_DIR = "early_stopping"
CHECKPOINT_DIR = "_fastflow_experiment_checkpoints"
CHECKPOINT_DIR = "_VisA"
CHECKPOINT_DIR = "_Scenario"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "dn_data_new",
    "dn_data_new_2",
    "dn_data_new_3",
    "dn_data3",
    "dn_data4",
    "dn_data_new_ano10"
]

DENSO_CATEGORIES = [
    "dn_data_new",
    "dn_data_new_2",
    "dn_data_new_3",
    "dn_data3",
    "dn_data4",
    "dn_data_new_ano10"
]

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum"
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"
BACKBONE_EFFIAD_SMALL = "EfficientAD_Small"
BACKBONE_EFFIAD_MEDIUM = "EfficientAD_Medium"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_EFFIAD_SMALL,
    BACKBONE_EFFIAD_MEDIUM
]

BATCH_SIZE = 32
NUM_EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10
