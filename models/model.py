import sys
import torch
from models.backbone import Backbone
from models.position_encoding import PositionEmbeddingSine
from models.backbone import Joiner
from models.mdef_detr.deformable_transformer \
    import DeformableTransformer as MDefDETRTransformer
from models.mdef_detr_minus_language.deformable_transformer \
    import DeformableTransformer as MDefDETRMinusLanguageTransformer
from models.mdef_detr.mdef_detr import MDefDETR
from models.mdef_detr_minus_language.mdef_detr_minus_language import MDefDETRMinusLanguage


class Model:
    """
    This class initiates the specified model.
    """

    def __init__(self, name, checkpoints_path=None):
        # Select the correct model
        if name == "mdef_detr":
            assert checkpoints_path is not None
            model = _make_mdef_detr(checkpoints_path)
            from inference.modulated_detection import ModulatedDetection as Inference
        elif name == "mdef_detr_minus_language":
            assert checkpoints_path is not None
            model = _make_mdef_detr_minus_language(checkpoints_path)
            from inference.minus_language import MinusLanguage as Inference
        else:
            print(f"Please provide correct models to use in configuration. "
                  f"Available options are ['mdef_detr','mdef_detr_minus_language']")
            sys.exit(1)
        # Initialize the selected DataLoader
        self.model = Inference(model)

    def get_model(self):
        """
        This function returns the selected models
        """
        return self.model


def _make_backbone(backbone_name="resnet101", mask: bool = True):
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=False)

    hidden_dim = 256
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels

    return backbone_with_pos_enc


def _make_mdef_detr(checkpoints_path):
    backbone = _make_backbone()
    transformer = MDefDETRTransformer(d_model=256, return_intermediate_dec=False, num_feature_levels=4,
                                      dim_feedforward=1024, text_encoder_type="roberta-base")
    model = MDefDETR(backbone=backbone, transformer=transformer, num_classes=255, num_queries=300,
                     num_feature_levels=4)
    checkpoint = torch.load(checkpoints_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model


def _make_mdef_detr_minus_language(checkpoints_path):
    backbone = _make_backbone()
    transformer = MDefDETRMinusLanguageTransformer(d_model=256, return_intermediate_dec=True, num_feature_levels=4,
                                                   dim_feedforward=1024)
    model = MDefDETRMinusLanguage(backbone=backbone, transformer=transformer, num_classes=1, num_queries=300,
                                  num_feature_levels=4)
    checkpoint = torch.load(checkpoints_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model
