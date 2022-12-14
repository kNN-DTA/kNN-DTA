import os
from typing import Any, Dict

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

from fairseq import utils
from fairseq.modules import MultiheadAttention
from fairseq import checkpoint_utils
from fairseq.models.custom_roberta.model import (
    RobertaModel,
    RobertaEncoder,
    DTIRobertaEncoder,
    RobertaClassificationHead,
    base_architecture as roberta_base_architecture
)

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import GradMultiply

DEFAULT_MAX_MOLECULE_POSITIONS = 512
DEFAULT_MAX_PROTEIN_POSITIONS = 1024

logger = logging.getLogger(__name__)

@register_model("dti_knn_from_pretrained_roberta_no_cross_attn_1")
# class RobertaDTI(RobertaModel):
class RobertaDTIKNNFromPretrainedRobertaNoCrossAttn1(BaseFairseqModel):
    def __init__(self, args, encoder_0, encoder_1):
        super().__init__()
        self.args = args
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        # We follow BERT's random weight initialization
        # self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()
        # delete this module, which is never used
        # self.cross_attn = MultiheadAttention(
        #     args.encoder_embed_dim,
        #     args.encoder_attention_heads,
        #     dropout=args.attention_dropout,
        # )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--grad-multiply", type=float, metavar="D", default=1, help="Apply different lr on backbone and classification head"
        )
        parser.add_argument(
            "--pretrained-molecule-roberta-checkpoint",
            type=str,
            metavar="STR",
            help="roberta model to use for initializing molecule encoder",
        )
        parser.add_argument(
            "--pretrained-protein-roberta-checkpoint",
            type=str,
            metavar="STR",
            help="roberta model to use for initializing protein encoder",
        )
        parser.add_argument(
            "--init-molecule-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into protein encoder",
        )
        parser.add_argument(
            "--init-protein-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into molecule encoder",
        )
        parser.add_argument(
            "--max-positions-molecule", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--max-positions-protein", type=int, help="number of positional embeddings to learn"
        )

        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )

        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        assert hasattr(args, "pretrained_molecule_roberta_checkpoint"), (
            "You must specify a path for --pretrained-molecule-roberta-checkpoint to use "
            "--arch dti_from_pretrained_roberta"
        )
        assert hasattr(args, "pretrained_protein_roberta_checkpoint"), (
            "You must specify a path for --pretrained-protein-roberta-checkpoint to use "
            "--arch dti_from_pretrained_roberta"
        )

        assert not (
            getattr(args, "init_molecule_encoder_only", False)
            and getattr(args, "init_protein_encoder_only", False)
        ), "Only one of --init-molecule-encoder-only and --init-protein-encoder-only can be set."

        # make sure all arguments are present
        base_architecture(args)

        # if not hasattr(args, "max_positions"):
        #     args.max_positions = args.tokens_per_sample
        if not hasattr(args, 'max_positions_molecule'):
            args.max_source_positions = DEFAULT_MAX_MOLECULE_POSITIONS
        if not hasattr(args, 'max_positions_protein'):
            args.max_target_positions = DEFAULT_MAX_PROTEIN_POSITIONS

        # encoder_0 = RobertaEncoder(args, task.source_dictionary_0)
        # encoder_1 = RobertaEncoder(args, task.source_dictionary_1)
        encoder_0 = cls.build_molecule_encoder(args, task.source_dictionary_0)
        encoder_1 = cls.build_protein_encoder(args, task.source_dictionary_1)

        # fix 2 encoders
        # for param in encoder_0.parameters():
        #     param.requires_grad = False
        # for param in encoder_1.parameters():
        #     param.requires_grad = False
        for i, child in enumerate(encoder_0.sentence_encoder.children()):
            if i <= 3:   
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for j, child_child in enumerate(child.children()):
                    if j <= 11:
                        for param in child_child.parameters():
                            param.requires_grad = False

        for i, child in enumerate(encoder_1.sentence_encoder.children()):
            if i <= 3:   
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for j, child_child in enumerate(child.children()):
                    if j <= 11:
                        for param in child_child.parameters():
                            param.requires_grad = False

        return cls(args, encoder_0, encoder_1)

    @classmethod
    def build_molecule_encoder(cls, args, src_dict):
        return MoleculeEncoderFromPretrainedRoberta(args, src_dict)

    @classmethod
    def build_protein_encoder(cls, args, src_dict):
        return ProteinEncoderFromPretrainedRoberta(args, src_dict)
    
    # # For training
    # def forward(
    #     self,
    #     src_tokens_0,
    #     src_tokens_1,
    #     features_only=False,
    #     return_all_hiddens=False,
    #     classification_head_name=None,
    #     **kwargs
    # ):
    #     if classification_head_name is not None:
    #         features_only = True

    #     x_0, extra_0 = self.encoder_0(src_tokens_0, features_only, return_all_hiddens, **kwargs)
    #     x_1, extra_1 = self.encoder_1(src_tokens_1, features_only, return_all_hiddens, **kwargs)
        
    #     if classification_head_name is not None:
    #         x = torch.cat((x_0[:, 0, :], x_1[:, 0, :]), 1).unsqueeze(1)
    #         if isinstance(x, Tensor):
    #             x = GradMultiply.apply(x, self.args.grad_multiply)
    #         x = self.classification_heads[classification_head_name](x)
        
    #     # return x, extra_0, extra_1
    #     return x, x_0[:, 0, :].squeeze(), x_1[:, 0, :].squeeze()

    # For test
    def forward(
        self,
        src_tokens_0,
        src_tokens_1,
        knn_cls_0=0,
        knn_cls_1=0,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        use_which_embedding='no',
        knn_embedding_weight_0=0.8,
        knn_embedding_weight_1=0.8,
        alpha=0.707,
        use_only_mlp=False,
        get_only_mol_cls=False,
        cls_0=0,
        cls_1=0,
        **kwargs
    ):
        if get_only_mol_cls:
            if classification_head_name is not None:
                features_only = True

            x_0, extra_0 = self.encoder_0(src_tokens_0, features_only, return_all_hiddens, **kwargs)
            
            return x_0[:, 0, :]

        else:
            if not use_only_mlp:
                if classification_head_name is not None:
                    features_only = True

                x_0, extra_0 = self.encoder_0(src_tokens_0, features_only, return_all_hiddens, **kwargs)
                x_1, extra_1 = self.encoder_1(src_tokens_1, features_only, return_all_hiddens, **kwargs)
                if classification_head_name is not None:              
                    if use_which_embedding == 'mol_pro':
                        x = torch.cat((alpha * (knn_embedding_weight_0 * x_0[:, 0, :] + (1 - knn_embedding_weight_0) * knn_cls_0), alpha * (knn_embedding_weight_1 * x_1[:, 0, :] + (1 - knn_embedding_weight_1) * knn_cls_1)), 1).unsqueeze(1)
                        # x = torch.cat(((knn_embedding_weight_0 * x_0[:, 0, :] + (1 - knn_embedding_weight_0) * knn_cls_0), (knn_embedding_weight_1 * x_1[:, 0, :] + (1 - knn_embedding_weight_1) * knn_cls_1)), 1).unsqueeze(1)
                    else:
                        x = torch.cat((x_0[:, 0, :], x_1[:, 0, :]), 1).unsqueeze(1)
                    if isinstance(x, Tensor):
                        x = GradMultiply.apply(x, self.args.grad_multiply)
                    x = self.classification_heads[classification_head_name](x)

                # return x, x_0[:, 0, :].squeeze(), x_1[:, 0, :].squeeze()
                return x, x_0[:, 0, :], x_1[:, 0, :]
                # return x, x_0[:, 0, :], x_1[:, 0, :], x_0[:, 0, :], x_1[:, 0, :]
            else:
                if classification_head_name is not None:              
                    if use_which_embedding == 'mol_pro':
                        x = torch.cat((alpha * (knn_embedding_weight_0 * cls_0 + (1 - knn_embedding_weight_0) * knn_cls_0), alpha * (knn_embedding_weight_1 * cls_1 + (1 - knn_embedding_weight_1) * knn_cls_1)), 1).unsqueeze(1)
                        # x = torch.cat(((knn_embedding_weight_0 * x_0[:, 0, :] + (1 - knn_embedding_weight_0) * knn_cls_0), (knn_embedding_weight_1 * x_1[:, 0, :] + (1 - knn_embedding_weight_1) * knn_cls_1)), 1).unsqueeze(1)
                    else:
                        x = torch.cat((cls_1, cls_1), 1).unsqueeze(1)
                    if isinstance(x, Tensor):
                        x = GradMultiply.apply(x, self.args.grad_multiply)
                    x = self.classification_heads[classification_head_name](x)

                return x, alpha * (knn_embedding_weight_0 * cls_0 + (1 - knn_embedding_weight_0) * knn_cls_0), alpha * (knn_embedding_weight_1 * cls_1 + (1 - knn_embedding_weight_1) * knn_cls_1)

  
    

    def my_register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=2 * self.args.encoder_embed_dim, # concat interaction tokens
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

def upgrade_state_dict_with_roberta_weights(
    state_dict: Dict[str, Any], pretrained_roberta_checkpoint: str
) -> Dict[str, Any]:
    """
    Load Roberta weights into a Roberta encoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_roberta_checkpoint: checkpoint to load roberta weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current roberta encoder
            and the pretrained_roberta_checkpoint
    """
    if not os.path.exists(pretrained_roberta_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_roberta_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_roberta_checkpoint)
    roberta_state_dict = state["model"]
    # for key in roberta_state_dict.keys():
    #     for search_key in ["embed_tokens", "embed_positions", "layers"]:
    #         if search_key in key:
    #             subkey = key[key.find(search_key) :]
    #             assert subkey in state_dict, (
    #                 "{} Transformer encoder / decoder "
    #                 "state_dict does not contain {}. Cannot "
    #                 "load {} from pretrained XLM checkpoint "
    #                 "{} into Transformer.".format(
    #                     str(state_dict.keys()), subkey, key, pretrained_roberta_checkpoint
    #                 )
    #             )

    #             state_dict[subkey] = roberta_state_dict[key]
    if 'encoder.sentence_encoder.layernorm_embedding.weight' in roberta_state_dict.keys():
        roberta_state_dict['encoder.sentence_encoder.emb_layer_norm.weight'] = roberta_state_dict.pop('encoder.sentence_encoder.layernorm_embedding.weight')
    if 'encoder.sentence_encoder.layernorm_embedding.bias' in roberta_state_dict.keys():
        roberta_state_dict['encoder.sentence_encoder.emb_layer_norm.bias'] = roberta_state_dict.pop('encoder.sentence_encoder.layernorm_embedding.bias')
    i = 0
    for key in roberta_state_dict.keys():
        if key.startswith("encoder"):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            subkey = key[len("encoder") + 1:]
            if subkey in state_dict.keys():
                state_dict[subkey] = roberta_state_dict[key]
                i += 1
    logger.info('load {} values from pretrained model'.format(i))
    return state_dict


class MoleculeEncoderFromPretrainedRoberta(DTIRobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary, args.max_positions_molecule)
        if getattr(args, "init_protein_encoder_only", False):
            # Don't load roberta weights for molecule encoder if --init-protein-encoder-only
            return

        assert hasattr(args, "pretrained_molecule_roberta_checkpoint"), (
            "--pretrained-molecule-roberta-checkpoint must be specified to load molecule "
            "encoder from pretrained roberta"
        )
        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
            state_dict=self.state_dict(),
            pretrained_roberta_checkpoint=args.pretrained_molecule_roberta_checkpoint,
        )
        # self.load_state_dict(roberta_loaded_state_dict, strict=True)
        self.load_state_dict(roberta_loaded_state_dict, strict=False)


class ProteinEncoderFromPretrainedRoberta(DTIRobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary, args.max_positions_protein)
        if getattr(args, "init_molecule_encoder_only", False):
            # Don't load roberta weights for protein encoder if --init-molecule-encoder-only
            return
        assert hasattr(args, "pretrained_protein_roberta_checkpoint"), (
            "--pretrained-protein-roberta-checkpoint must be specified to load protein "
            "encoder from pretrained roberta"
        )
        roberta_loaded_state_dict = upgrade_state_dict_with_roberta_weights(
            state_dict=self.state_dict(),
            pretrained_roberta_checkpoint=args.pretrained_protein_roberta_checkpoint,
        )
        # self.load_state_dict(roberta_loaded_state_dict, strict=True)
        self.load_state_dict(roberta_loaded_state_dict, strict=False)

@register_model_architecture(
    "dti_knn_from_pretrained_roberta_no_cross_attn_1", "dti_knn_from_pretrained_roberta_no_cross_attn_1"
)
def base_architecture(args):
    roberta_base_architecture(args)




