import warnings
import logging
from typing import Any, Dict, Optional, Tuple, Union

from geomloss import SamplesLoss
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader

from trainers.kgtrainer_utils import sinkhorn_loss_default
from trainers.seq2seq_trainer import Seq2SeqTrainer

from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    nested_concat,
    nested_numpify,
)

from transformers.integrations import (
    is_optuna_available,
    is_tensorboard_available,
)

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter

if is_optuna_available():
    import optuna

from evals.eval_acc_div import eval_accuracy_diversity
logger = logging.getLogger(__name__)


class KGMoESeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixtures = self.data_args.mixtures
        self.expert_prompt = self.data_args.expert_prompt
        self.mixture_embedding = self.data_args.mixture_embedding
        self.pows = self.data_args.pows
        self.kg_loss_ratio = self.data_args.kg_loss_ratio
        self.opt_loss_ratio = self.data_args.opt_loss_ratio

    def _training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer) -> torch.Tensor:

        self.B, self.L = inputs['labels'].shape  # target_ids
        self.BC, self.LC = inputs['concept_labels'].shape  # concept_labels
        assert self.B == self.BC
        self.pad_mask = (inputs['labels'] == self.config.pad_token_id).view(self.B, 1, self.L).to(self.args.device)
        self.concept_pad_mask = (inputs['concept_labels'] == self.config.pad_token_id).view(self.BC, 1, self.LC).to(
            self.args.device)

        inputs = self._prepare_inputs(inputs)  # move tensors to gpu

        mixture_tmp = torch.arange(self.mixtures, dtype=torch.long, device=inputs['input_ids'].device).view(
            self.mixtures, 1)  # [2, 1] [[0], [1]]
        kg_mixture_ids = mixture_tmp.repeat(inputs['concept_ids'].shape)  # [120, 300] [[0, 0, 0, ..., 0], [1, 1, 1, ..., 1], ..., [0, 0, 0, ..., 0], [1, 1, 1, ..., 1]]

        if self.mixture_embedding:

            lm_mixture_ids = mixture_tmp.repeat(inputs['input_ids'].shape)
            mixture_inputs = {k: self.repeat(v, self.mixtures) for k, v in inputs.items()}
            mixture_inputs['lm_mixture_ids'] = lm_mixture_ids
            mixture_inputs['kg_mixture_ids'] = kg_mixture_ids
            model.eval()

            mixture_ids = self.compute_mixture_ids(model, mixture_inputs)
            inputs['lm_mixture_ids'] = mixture_ids.expand(inputs['input_ids'].shape)
            inputs['kg_mixture_ids'] = mixture_ids.expand(inputs['concept_ids'].shape)

        else:  # using prompt as different expert

            mixture_ids_prompt = self.expert_prompt.repeat(self.B, 1).to(self.args.device) # [180, 5]
            mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1).to(self.args.device) # [180, 5]

            mixture_inputs = {k: self.repeat(v, self.mixtures) for k, v in inputs.items()}
            mixture_inputs['kg_mixture_ids'] = kg_mixture_ids
            mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
            mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)

            model.eval()
            mixture_ids = self.compute_mixture_ids(model, mixture_inputs)
            expanded_mixture_ids = mixture_ids.expand(self.B, self.data_args.prompt_nums).unsqueeze(dim=1)
            input_ids_prompt = torch.gather(mixture_ids_prompt.view(
                self.B, self.mixtures, -1), dim=1, index=expanded_mixture_ids).squeeze()
            attention_prompt = torch.full(input_ids_prompt.shape, 1).to(self.args.device)
            inputs['kg_mixture_ids'] = mixture_ids.expand(inputs['concept_ids'].shape)
            inputs['input_ids'] = torch.cat([input_ids_prompt, inputs['input_ids']], dim=1)
            inputs['attention_mask'] = torch.cat([attention_prompt, inputs['attention_mask']], dim=1)

        # do the expert training!
        model.train()
        lm_loss, kg_loss, opt_loss = self.compute_loss(model, inputs)

        if opt_loss == 0:
            loss = lm_loss + self.kg_loss_ratio * kg_loss
        else:
            loss = lm_loss + self.kg_loss_ratio * kg_loss + self.opt_loss_ratio * opt_loss

        assert torch.isnan(loss).item() == False

        # if opt_loss == 0:
        #     print("opt_loss is zero!! ")
        # print("lm_loss:", lm_loss, "kg_loss", kg_loss, "opt_loss", opt_loss)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach(), lm_loss, kg_loss, opt_loss

    def compute_loss(self, model, inputs):
        # labels=target_ids, concept_labels (if label in source subgraph is one of target concpets)
        lm_labels = inputs.pop("labels")
        # kg_labels = inputs.pop("concept_labels")
        lm_outputs, kg_logits, pooled_kg_outputs, original_kg_outputs, kg_labels, perm_idx = model(**inputs, use_cache=False)
        lm_logits = lm_outputs[0]
        lm_loss = self._compute_loss(lm_logits, lm_labels)
        kg_loss = self._compute_kg_loss(kg_logits, kg_labels)
        if original_kg_outputs is not None:
            wass_xy, opt_loss = self._compute_opt_loss(pooled_kg_outputs, original_kg_outputs, model.device)
        else:
            opt_loss = 0
        del lm_outputs, kg_logits, pooled_kg_outputs, original_kg_outputs, kg_labels
        return lm_loss, kg_loss, opt_loss

    def compute_mixture_ids(self, model, inputs):
        
        _inputs = inputs.copy()
        _lm_labels = _inputs.pop("labels")
        # _kg_labels = _inputs.pop("concept_labels")
        # _kg_labels = _inputs["concept_labels"]
        lm_outputs, kg_logits, pooled_kg_outputs, original_kg_outputs, kg_labels, perm_idx = model(**_inputs, use_cache=False)
        lm_logits = lm_outputs[0]
        mixture_ids = self._compute_mixture_loss(lm_logits, kg_logits, _lm_labels, kg_labels, pooled_kg_outputs, original_kg_outputs, perm_idx)
        del lm_outputs, kg_logits, pooled_kg_outputs, original_kg_outputs, kg_labels
        return mixture_ids

    def _compute_mixture_loss(self, lm_logits, kg_logits, lm_labels, kg_labels, pooled_kg_outputs, original_kg_outputs, perm_idx):
        # lm_logits & lm_labels: [180, 27] (decoder_dim), kg_logits & kg_labels: [180, 300]
        assert lm_logits.shape[:2] == lm_labels.shape
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')

        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1)).reshape(self.B, self.mixtures,
                                                                                                self.L)
        lm_loss = lm_loss.masked_fill(self.pad_mask, 0).sum(dim=2)
        kg_loss = self._compute_kg_loss(kg_logits, kg_labels, reduction='none').view(self.BC, self.mixtures, -1)

        concept_pad_mask = self.concept_pad_mask.squeeze(dim=1)
        new_mask = []
        for mixture in range(self.mixtures):
            _perm_idx = perm_idx[:, mixture, :]
            _concept_pad_mask = []
            for i in range(_perm_idx.shape[0]):
                _concept_pad_mask.append(concept_pad_mask[i, _perm_idx[i]])
            _concept_pad_mask = torch.stack(_concept_pad_mask, dim=0)
            new_mask.append(_concept_pad_mask)

        new_concept_pad_mask = torch.stack(new_mask, dim=1)
        kg_loss = kg_loss.masked_fill(new_concept_pad_mask, 0).sum(dim=2)
        # kg_loss = kg_loss.masked_fill(self.concept_pad_mask, 0).sum(dim=2)

        wass_xy, opt_loss = self._compute_opt_loss(pooled_kg_outputs, original_kg_outputs, lm_logits.device)
        wass_xy = wass_xy.view(self.BC, self.mixtures)

        loss = lm_loss + self.kg_loss_ratio * kg_loss + self.opt_loss_ratio * wass_xy
        mixture_ids = loss.argmin(dim=1).unsqueeze(dim=1).type(torch.int64)

        return mixture_ids

    def _compute_kg_loss(self, node_logits, node_labels, reduction='mean'):

        loss_weights = (node_labels + 1).pow(self.pows)

        if node_logits.shape != node_labels.shape:
            node_logits = node_logits[:, : node_labels.shape[-1]]

        node_loss = F.binary_cross_entropy_with_logits(
            node_logits.float(), node_labels.float(),
            weight=loss_weights, reduction='none')

        valid_mask = ~(node_labels == -1)
        labels_len = valid_mask.float().sum(dim=1)
        labels_len = labels_len.masked_fill(labels_len == 0.0, 1.0)

        if reduction == 'mean':
            _node_loss = node_loss.sum(dim=1) / labels_len
            _node_loss = _node_loss.mean()
            return _node_loss

        return node_loss

    def _compute_opt_loss(self, pooled_kg_outputs, original_kg_outputs, device):
        opt_losses = []
        total_loss = 0
        epsilon = 1.0
        opt_epochs = 10
        # node_hidden: [16, 210, 768], node_output: [16, 300, 768]
        Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        Wass_xy = Loss(pooled_kg_outputs, original_kg_outputs)
        # print("Wass_xy:", Wass_xy.shape)
        total_loss = torch.mean(Wass_xy)
        # for i in range(len(node_output)):
        #     mem = self.get_nonzero_rows(node_output[i])
        #     new_mem = self.get_nonzero_rows(node_hidden[i])
        #     if mem.shape[0] == 0: continue
        #     if new_mem.shape[0] == 0: continue
        #     loss = sinkhorn_loss_default(mem, new_mem, epsilon, niter=opt_epochs, device=device).float()
        #
        #     if loss.item() == 0:
        #         print("opt loss is 0!!\n>>> mem:", mem.shape, mem)
        #         print(">>> new_mem:", new_mem.shape, new_mem)
        #         assert False
        #     total_loss += loss
        #     opt_losses.append(loss.item())
        # if len(opt_losses) > 0:
        #     final_loss = sum(opt_losses) / len(opt_losses)
        # else:
        #     final_loss = 0.0
        # assert final_loss != 0
        # final_loss = total_loss/len(opt_losses)
        if total_loss == 0:
            print("total opt loss is zero")
        return Wass_xy, total_loss

    def get_nonzero_rows(self, M):  # M is a matrix
        # row_ind = M.sum(-1).nonzero().squeeze() #nonzero has bugs in Pytorch 1.2.0.........
        # So we use other methods to take place of it
        MM, MM_ind = M.sum(-1).sort()
        N = (M.sum(-1) > 0).sum()
        return M[MM_ind[:N]]

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.args.predict_with_generate and not self.args.prediction_loss_only:
                num_return_sequences = self.data_args.eval_beams if self.data_args.do_sample else None
                expert_prompt = self.data_args.expert_prompt if hasattr(self.data_args, 'expert_prompt') else None

                generated_tokens = model.generate(
                    # Text Input!
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    # Graph Input!
                    concept_ids=inputs["concept_ids"],
                    concept_distances=inputs["concept_distances"],
                    concept_labels=inputs["concept_labels"],
                    head_ids=inputs["head_ids"],
                    tail_ids=inputs["tail_ids"],
                    relation_ids=inputs["relation_ids"],
                    triple_labels=inputs["triple_labels"],
                    adj=inputs["adj"],
                    # Others!
                    num_beams=self.data_args.eval_beams,
                    num_return_sequences=num_return_sequences,
                    max_length=self.max_gen_length,
                    do_sample=self.data_args.do_sample,
                    top_k=self.data_args.top_k,
                    top_p=self.data_args.top_p,
                    expert_prompt=expert_prompt,
                    use_cache=True,
                )

                # in case the batch is shorter than max length, the output should be padded
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.max_gen_length)

            lm_labels = inputs.get("labels")
            lm_outputs, _, _, _, _ = model(**inputs, use_cache=False)
            # TODO: not sure why computing loss here
            # loss = self._compute_loss(lm_outputs[0], lm_labels)
            # loss = loss.mean().item()
            # if self.args.prediction_loss_only:
            #     return (loss, None, None)

            lm_logits = generated_tokens if self.args.predict_with_generate else lm_outputs[1]

        lm_labels = self.repeat(lm_labels, self.data_args.eval_beams)
        lm_labels = self._pad_tensors_to_max_len(lm_labels.detach(), self.max_gen_length)
        return lm_logits, lm_labels

    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:

        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning, )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only)

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        ''' eval all datas in the dev set '''
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            lm_logits, lm_labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if lm_logits is not None:
                preds = lm_logits if preds is None else nested_concat(preds, lm_logits, dim=0)
            if lm_labels is not None:
                label_ids = lm_labels if label_ids is None else nested_concat(label_ids, lm_labels, dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        assert preds.shape[0] == label_ids.shape[0]

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
