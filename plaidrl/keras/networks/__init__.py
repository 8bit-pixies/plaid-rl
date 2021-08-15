"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
# from plaidrl.torch.networks.basic import (
#     Clamp,
#     ConcatTuple,
#     Detach,
#     Flatten,
#     FlattenEach,
#     Reshape,
#     Split,
# )
# from plaidrl.torch.networks.cnn import CNN, BasicCNN, CNNPolicy, MergedCNN
# from plaidrl.torch.networks.dcnn import DCNN, TwoHeadDCNN
# from plaidrl.torch.networks.feat_point_mlp import FeatPointMlp
# from plaidrl.torch.networks.image_state import ImageStatePolicy, ImageStateQ
# from plaidrl.torch.networks.linear_transform import LinearTransform
# from plaidrl.torch.networks.mlp import (
#     ConcatMlp,
#     ConcatMultiHeadedMlp,
#     Mlp,
#     MlpPolicy,
#     MlpQf,
#     MlpQfWithObsProcessor,
#     TanhMlpPolicy,
# )
# from plaidrl.torch.networks.normalization import LayerNorm
# from plaidrl.torch.networks.pretrained_cnn import PretrainedCNN
# from plaidrl.torch.networks.two_headed_mlp import TwoHeadMlp

# __all__ = [
#     "Clamp",
#     "ConcatMlp",
#     "ConcatMultiHeadedMlp",
#     "ConcatTuple",
#     "BasicCNN",
#     "CNN",
#     "CNNPolicy",
#     "DCNN",
#     "Detach",
#     "FeatPointMlp",
#     "Flatten",
#     "FlattenEach",
#     "LayerNorm",
#     "LinearTransform",
#     "ImageStatePolicy",
#     "ImageStateQ",
#     "MergedCNN",
#     "Mlp",
#     "PretrainedCNN",
#     "Reshape",
#     "Split",
#     "TwoHeadDCNN",
#     "TwoHeadMlp",
# ]

from plaidrl.keras.networks.mlp import mlp_builder, concat_mlp_builder

__all__ = ["mlp_builder", "concat_mlp_builder"]
