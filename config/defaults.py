from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.FROM_SCRATCH = True
_C.DATA_TRUNK = None

_C.OUTPUT_DIR = ''
_C.DATA_DIR = ''
_C.GLOVE_DIR = ''
_C.TENSORBOARD_DIR = ''


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.MAX_QUERY_LEN = 26
_C.INPUT.MAX_VIDEO_LEN = 200

# The input frame number (For VidSTG)
_C.INPUT.TRAIN_SAMPLE_NUM = 64
# The input frame rate (For HC_STVG, 20s per input video)
_C.INPUT.SAMPLE_FPS = 3.2

# The input video resolution
_C.INPUT.RESOLUTION = 224
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# If perform multiscale training
_C.INPUT.AUG_SCALE = True
# If perform translate augumentation training
_C.INPUT.AUG_TRANSLATE = False

# Image ColorJitter
_C.INPUT.FLIP_PROB_TRAIN = 0.5
_C.INPUT.TEMP_CROP_PROB = 0.5

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""
_C.MODEL.EMA = True
_C.MODEL.EMA_DECAY = 0.9998
_C.MODEL.QUERY_NUM = 1   # each frame a single query
_C.MODEL.DOWN_RATIO = 4

# -----------------------------------------------------------------------------
# Vision Encoder options
# -----------------------------------------------------------------------------

_C.MODEL.VISION_BACKBONE = CN()
_C.MODEL.VISION_BACKBONE.NAME = 'resnet101'  # resnet50 or resnet101
_C.MODEL.VISION_BACKBONE.POS_ENC = 'sine'  # sine, sineHW or learned
_C.MODEL.VISION_BACKBONE.DILATION = False # If true, we replace stride with dilation in the last convolutional block (DC5)
_C.MODEL.VISION_BACKBONE.FREEZE = False # If true, freeze the vision backbone parameters


# -----------------------------------------------------------------------------
# Language Encoder Config
# -----------------------------------------------------------------------------
_C.MODEL.TEXT_MODEL = CN()
_C.MODEL.TEXT_MODEL.NAME = 'roberta-base'  # "bert-base", "roberta-large"
_C.MODEL.TEXT_MODEL.FREEZE = False

# If true, use LSTM as the text encoder
_C.MODEL.USE_LSTM = False
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.NAME = 'lstm'
_C.MODEL.LSTM.HIDDEN_SIZE = 512
_C.MODEL.LSTM.BIDIRECTIONAL = True
_C.MODEL.LSTM.DROPOUT = 0
_C.MODEL.LSTM_NUM_LAYERS = 2


# -----------------------------------------------------------------------------
# CG Pipeline Config
# -----------------------------------------------------------------------------
_C.MODEL.CG = CN()
_C.MODEL.CG.HIDDEN = 256
_C.MODEL.CG.QUERY_DIM = 4  # the anchor dim
_C.MODEL.CG.ENC_LAYERS = 6
_C.MODEL.CG.DEC_LAYERS = 6
_C.MODEL.CG.FFN_DIM = 2048
_C.MODEL.CG.DROPOUT = 0.1
_C.MODEL.CG.HEADS = 8
_C.MODEL.CG.USE_LEARN_TIME_EMBED = False
_C.MODEL.CG.USE_ACTION = True  # use the actioness head by default
_C.MODEL.CG.FROM_SCRATCH = True

# For 2D-Map prediction
_C.MODEL.CG.TEMP_PRED_LAYERS = 6
_C.MODEL.CG.CONV_LAYERS = 4
_C.MODEL.CG.TEMP_HEAD = 'attn'   # attn or conv
_C.MODEL.CG.KERNAL_SIZE = 9
_C.MODEL.CG.MAX_MAP_SIZE = 128
_C.MODEL.CG.POOLING_COUNTS = [15,8,8,8]

_C.MODEL.CG.TEMP_THETA = 0.
_C.MODEL.CG.SPAT_GT_THETA = 0.
_C.MODEL.CG.SPAT_THETA = 0.

# -----------------------------------------------------------------------------
# DATASET related params
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'VidSTG'
_C.DATASET.NUM_CLIP_FRAMES = 32
# The minimum gt frames in a sampled clip
_C.DATASET.MIN_GT_FRAME = 4


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SIZE_DIVISIBILITY = 0
_C.DATALOADER.ASPECT_RATIO_GROUPING = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 30
_C.SOLVER.BATCH_SIZE = 1   # The video number per GPU, should be set 1.
_C.SOLVER.SHUFFLE = True
_C.SOLVER.BASE_LR = 2e-5
_C.SOLVER.VIS_BACKBONE_LR = 1e-5
_C.SOLVER.TEXT_LR = 2e-5
_C.SOLVER.TEMP_LR = 1e-4
_C.SOLVER.OPTIMIZER = 'adamw'
_C.SOLVER.MAX_GRAD_NORM = 0.1

# loss weight hyper-parameter
_C.SOLVER.BBOX_COEF = 5
_C.SOLVER.GIOU_COEF = 2
_C.SOLVER.TEMP_COEF = 2
_C.SOLVER.ATTN_COEF = 1
_C.SOLVER.ACTIONESS_COEF = 2
_C.SOLVER.CONF_COEF = 1

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.POWER = 0.9    # For Poly LRScheduler
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

_C.SOLVER.WARMUP_PROP = 0.01
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.SCHEDULE = CN()
_C.SOLVER.SCHEDULE.TYPE = "linear_with_warmup"
_C.SOLVER.SCHEDULE.DROP_STEP = [8,12]

# the following paramters are only used for WarmupReduceLROnPlateau
_C.SOLVER.SCHEDULE.PATIENCE = 2
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4
_C.SOLVER.SCHEDULE.COOLDOWN = 1
_C.SOLVER.SCHEDULE.FACTOR = 0.5
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 7

_C.SOLVER.PRE_VAL = False
_C.SOLVER.TO_VAL = True
_C.SOLVER.VAL_PERIOD = 3000   # every 10% training iterations completed, start a avaluation
_C.SOLVER.CHECKPOINT_PERIOD = 5000


_C.SOLVER.USE_ATTN = False  # whether to use the guided attention loss, to compare with TubeDETR
_C.SOLVER.SIGMA = 2.0  # standard deviation for the quantized gaussian law used for the kullback leibler divergence loss
_C.SOLVER.USE_AUX_LOSS = True # whether to use auxiliary decoding losses (loss at each layer)
_C.SOLVER.EOS_COEF = 0.1  # The coeff for negative sample


_C.nodes= 8
_C.tasks_per_node= 8
_C.image_tag= "in1k-16f"
_C.video_tag="k400-16x8x3"
_C.image_eval_name= "image_classification_frozen"
_C.video_eval_name= "image_classification_frozen"
_C.resume_checkpoint= False

_C.data= CN()
_C.data.root_path= "/home/ilias/PycharmProjects/pythonProject1/jepa/imagenet/"
_C.data.image_folder= "./"
_C.data.num_classes= 20
_C.data.resolution= 224
_C.data.dataset_name= "ImageNet"
_C.data.dataset_train= "/home/ilias/PycharmProjects/pythonProject1/jepa/kinetics400_train.csv"
_C.data.dataset_val= "/home/ilias/PycharmProjects/pythonProject1/jepa/kinetics400_val.csv"
_C.data.dataset_type= "VideoDataset"
_C.data.frames_per_clip= 16
_C.data.num_segments= 8
_C.data.num_views_per_segment= 3
_C.data.frame_step= 4

_C.optimization=CN()
_C.optimization.num_epochs= 20
_C.optimization.batch_size= 16
_C.optimization.weight_decay= 0.001
_C.optimization.lr= 0.001
_C.optimization.start_lr= 0.001
_C.optimization.final_lr= 0.0
_C.optimization.warmup= 0.
_C.optimization.use_bfloat16= True
_C.optimization.attend_across_segments= True
_C.optimization.resolution= 224

_C.pretrain=CN()
_C.pretrain.model_name="vit_large"
_C.pretrain.checkpoint_key= "target_encoder"
_C.pretrain.clip_duration= None
_C.pretrain.frames_per_clip= 16
_C.pretrain.tubelet_size= 2
_C.pretrain.uniform_power= True
_C.pretrain.use_sdpa= True
_C.pretrain.use_silu= False
_C.pretrain.tight_silu= False
_C.pretrain.patch_size= 16
_C.pretrain.folder= "/home/ilias/PycharmProjects/pythonProject1/jepa/checkpoints/pretrained/ViT-L"
_C.pretrain.checkpoint= "vitl16.pth.tar"  # name of pretrained model file inside folder
_C.pretrain.write_tag= "jepa"