import torch
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


def run_batch_generation(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    input_ids, token_type_ids, lm_labels = batch
    model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels)
    loss = model_outputs[0]
    lm_logits = model_outputs[1]
    return loss, lm_logits, torch.tensor([]), torch.tensor([])


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def run_batch_generation_sample(args, model, batch, dataset):
    special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
    current_output = []

    example = batch[0]
    knowledge, history = example["knowledge"], example["history"]
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    for i in range(args.max_length):
        instance, sequence = dataset.build_input_from_segments(
            knowledge, history, current_output, with_eos=False
        )

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        model_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = model_outputs[0]

        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    logger.warning("Warning: model generating special token with probability 1! Breaking...")
                    break
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    
    return current_output, response_text, dialog_id


def run_batch_selection_train(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids,
        mc_token_ids=mc_token_ids, mc_labels=mc_labels
    )
    mc_loss = model_outputs[0]
    lm_logits, mc_logits = model_outputs[1], model_outputs[2]
    return mc_loss, lm_logits, mc_logits, mc_labels


def run_batch_selection_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, _, mc_labels = batch
    all_mc_logits = []
    for index in range(0, input_ids.size(1), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            token_type_ids=token_type_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            mc_token_ids=mc_token_ids[0, index:index+candidates_per_forward].unsqueeze(1)
        )
        mc_logits = model_outputs[1]
        all_mc_logits.append(mc_logits.detach())
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    return torch.tensor(0.0), torch.tensor([]), all_mc_logits, mc_labels


def run_batch_detection(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, lm_labels, labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids,
        mc_token_ids=mc_token_ids, labels=labels
    )
    cls_loss = model_outputs[0]
    lm_logits, cls_logits = model_outputs[1], model_outputs[2]
    return cls_loss, lm_logits, cls_logits, labels