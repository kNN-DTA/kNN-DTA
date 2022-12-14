# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(
        self, sentence: str, *addl_sentences, no_separator=False
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = "<s> " + self.bpe.encode(sentence) + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )
        return tokens.long()

    def myencode_separate(
        self, sentence_0: str, sentence_1: str
    ) -> torch.LongTensor:

        sentence_0 = "<s> " + sentence_0 + " </s>"
        sentence_1 = "<s> " + sentence_1 + " </s>"
        tokens_0 = self.task.source_dictionary_0.encode_line(
            sentence_0, append_eos=False, add_if_not_exist=False
        )

        tokens_1 = self.task.source_dictionary_1.encode_line(
            sentence_1, append_eos=False, add_if_not_exist=False
        )

        return tokens_0.long(), tokens_1.long()
    
    def myencode_mol(
        self, sentence_0: str
    ) -> torch.LongTensor:

        sentence_0 = "<s> " + sentence_0 + " </s>"
        
        tokens_0 = self.task.source_dictionary_0.encode_line(
            sentence_0, append_eos=False, add_if_not_exist=False
        )

        return tokens_0.long()

    def myencode_pro(
        self, sentence_1: str
    ) -> torch.LongTensor:

        sentence_1 = "<s> " + sentence_1 + " </s>"

        tokens_1 = self.task.source_dictionary_1.encode_line(
            sentence_1, append_eos=False, add_if_not_exist=False
        )

        return tokens_1.long()
    
    def myencode_separate_batch(
        self, sentence_0: list, sentence_1: list
    ) -> list:
        tokens_0_list, tokens_1_list = [], []
        for s_0, s_1 in zip(sentence_0, sentence_1):
            s_0 = "<s> " + s_0 + " </s>"
            s_1 = "<s> " + s_1 + " </s>"
            tokens_0 = self.task.source_dictionary_0.encode_line(
                s_0, append_eos=False, add_if_not_exist=False
            )

            tokens_1 = self.task.source_dictionary_1.encode_line(
                s_1, append_eos=False, add_if_not_exist=False
            )
            tokens_0_list.append(tokens_0.long())
            tokens_1_list.append(tokens_1.long())
        return tokens_0_list, tokens_1_list

    def myencode_share(
        self, sentence: str, *addl_sentences, no_separator=False
    ) -> torch.LongTensor:
        bpe_sentence = "<s> " + sentence + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + s + " </s>"
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        features, extra = self.model(
            tokens.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features
    
    def myextract_features_separate(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )

        features, extra_0, extra_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states_0 = extra_0["inner_states"]
            inner_states_1 = extra_1["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states_0], [inner_state.transpose(0, 1) for inner_state in inner_states_1]
        else:
            return features  # just the last layer's features

    def myextract_features_separate_pro_hidden(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )

        features, extra_0, extra_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states_0 = extra_0["inner_states"]
            inner_states_1 = extra_1["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states_0], [inner_state.transpose(0, 1) for inner_state in inner_states_1]
        else:
            return extra_1  # just the last layer's features

    def myextract_features_separate_spec_pro(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, final_hidden_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )

        features, extra_0, extra_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            final_hidden_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states_0 = extra_0["inner_states"]
            inner_states_1 = extra_1["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states_0], [inner_state.transpose(0, 1) for inner_state in inner_states_1]
        else:
            return features  # just the last layer's features

    def myextract_features_separate_spec_pro_batch(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, final_hidden_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
    
        features, extra_0, extra_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            final_hidden_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states_0 = extra_0["inner_states"]
            inner_states_1 = extra_1["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states_0], [inner_state.transpose(0, 1) for inner_state in inner_states_1]
        else:
            return features  # just the last layer's features


    def myextract_features_separate_batch(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
    
        features, extra_0, extra_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states_0 = extra_0["inner_states"]
            inner_states_1 = extra_1["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states_0], [inner_state.transpose(0, 1) for inner_state in inner_states_1]
        else:
            return features  # just the last layer's features


    def myextract_features_separate_case_study(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )

        features, attn_0, attn_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
      
        return features, attn_0, attn_1  # just the last layer's features
    
    def myextract_features_separate_cls_attn(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )

        features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        
        return features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1  # just the last layer's features

    def myextract_features_separate_cls(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )

        features, cls_0, cls_1 = self.model(
            tokens_0.to(device=self.device),
            tokens_1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
            classification_head_name=classification_head_name,
        )
        
        return features, cls_0, cls_1 # just the last layer's features

    def myextract_features_separate_knn_cls(
        self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, knn_cls_0: torch.LongTensor = 0, knn_cls_1: torch.LongTensor = 0, return_all_hiddens: bool = False, classification_head_name: str = 'sentence_classification_head',
        use_which_embedding: str='no'
    ) -> torch.Tensor:
        if tokens_0.dim() == 1:
            tokens_0 = tokens_0.unsqueeze(0)
        if tokens_1.dim() == 1:
            tokens_1 = tokens_1.unsqueeze(0)
        if tokens_0.size(-1) > self.model.encoder_0.max_positions():
            raise ValueError(
                "tokens_0 exceeds maximum length: {} > {}".format(
                    tokens_0.size(-1), self.model.encoder_0.max_positions()
                )
            )

        if tokens_1.size(-1) > self.model.encoder_1.max_positions():
            raise ValueError(
                "tokens_1 exceeds maximum length: {} > {}".format(
                    tokens_1.size(-1), self.model.encoder_1.max_positions()
                )
            )
        
        if use_which_embedding == "mol":
            features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1 = self.model(
                tokens_0.to(device=self.device),
                tokens_1.to(device=self.device),
                knn_cls_0 = knn_cls_0.to(device=self.device),
                features_only=True,
                return_all_hiddens=return_all_hiddens,
                classification_head_name=classification_head_name,
                use_which_embedding="mol"
            )
        elif use_which_embedding == "pro":
            features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1 = self.model(
                tokens_0.to(device=self.device),
                tokens_1.to(device=self.device),
                knn_cls_1 = knn_cls_1.to(device=self.device),
                features_only=True,
                return_all_hiddens=return_all_hiddens,
                classification_head_name=classification_head_name,
                use_which_embedding="pro"
            )
        elif use_which_embedding == "mol_pro":
            features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1 = self.model(
                tokens_0.to(device=self.device),
                tokens_1.to(device=self.device),
                knn_cls_0 = knn_cls_0.to(device=self.device),
                knn_cls_1 = knn_cls_1.to(device=self.device),
                features_only=True,
                return_all_hiddens=return_all_hiddens,
                classification_head_name=classification_head_name,
                use_which_embedding="mol_pro"
            )
        else:
            features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1 = self.model(
                tokens_0.to(device=self.device),
                tokens_1.to(device=self.device),
                features_only=True,
                return_all_hiddens=return_all_hiddens,
                classification_head_name=classification_head_name,
                use_which_embedding="no"
            )
        # features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1 = self.model(
        #         tokens_0.to(device=self.device),
        #         tokens_1.to(device=self.device),
        #         knn_cls_0 = knn_cls_0.to(device=self.device),
        #         knn_cls_1 = knn_cls_1.to(device=self.device),
        #         features_only=True,
        #         return_all_hiddens=return_all_hiddens,
        #         classification_head_name=classification_head_name,
        #         use_which_embedding=use_which_embedding
        #     )
        return features, cls_0, cls_1, cls_1_attn_0, cls_0_attn_1  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def mypredict_separate(self, tokens_0: torch.LongTensor, tokens_1: torch.LongTensor, return_logits: bool = False, head: str = 'sentence_classification_head'):
        features = self.myextract_features_separate(tokens_0.to(device=self.device), tokens_1.to(device=self.device))
        return F.log_softmax(features, dim=-1)

    def extract_features_aligned_to_words(
        self, sentence: str, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
        from fairseq.models.roberta import alignment_utils
        from spacy.tokens import Doc

        nlp = alignment_utils.spacy_nlp()
        tokenizer = alignment_utils.spacy_tokenizer()

        # tokenize both with GPT-2 BPE and spaCy
        bpe_toks = self.encode(sentence)
        spacy_toks = tokenizer(sentence)
        spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
        alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)

        # extract features and align them
        features = self.extract_features(
            bpe_toks, return_all_hiddens=return_all_hiddens
        )
        features = features.squeeze(0)
        aligned_feats = alignment_utils.align_features_to_words(
            self, features, alignment
        )

        # wrap in spaCy Doc
        doc = Doc(
            nlp.vocab,
            words=["<s>"] + [x.text for x in spacy_toks] + ["</s>"],
            spaces=[True]
            + [x.endswith(" ") for x in spacy_toks_ws[:-1]]
            + [True, False],
        )
        assert len(doc) == aligned_feats.size(0)
        doc.user_token_hooks["vector"] = lambda token: aligned_feats[token.i]
        return doc

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = "<mask>"
        assert (
            masked_token in masked_input and masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token
        )

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = (
            (" {0} ".format(masked_token))
            .join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans])
            .strip()
        )
        tokens = self.task.source_dictionary.encode_line(
            "<s> " + text_spans_bpe + " </s>",
            append_eos=False,
            add_if_not_exist=False,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.model_eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(
            topk_predicted_token_bpe.split(" ")
        ):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
            if predicted_token_bpe.startswith("\u2581"):
                predicted_token = " " + predicted_token
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(
                            " {0}".format(masked_token), predicted_token
                        ),
                        values[index].item(),
                        predicted_token,
                    )
                )
            else:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(masked_token, predicted_token),
                        values[index].item(),
                        predicted_token,
                    )
                )
        return topk_filled_outputs

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(
            self.task, "disambiguate_pronoun"
        ), "roberta.disambiguate_pronoun() requires a model trained with the WSC task."
        with utils.model_eval(self.model):
            return self.task.disambiguate_pronoun(
                self.model, sentence, use_cuda=self.device.type == "cuda"
            )
