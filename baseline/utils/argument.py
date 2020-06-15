import logging
import os

import torch


logger = logging.getLogger(__name__)


def verify_args(args, parser):
    if args.eval_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required when --eval_only is set.")
        if not args.params_file:
            logger.info("params_file is not set, using the params.json in checkpoint")
            args.params_file = os.path.join(args.checkpoint, "params.json")
        else:
            logger.info("Using params_file %s from command line", args.params_file)
    else:
        if not args.params_file:
            parser.error("--params_file is required during training")


def update_additional_params(params, args):
    if args.get("dataroot"):
        params["dataset_args"]["dataroot"] = args["dataroot"]

    if args.get("knowledge_file"):
        params["dataset_args"]["knowledge_file"] = args["knowledge_file"]
    
    if args.get("negative_sample_method", ""):
        params["dataset_args"]["negative_sample_method"] = args["negative_sample_method"]
    
    if args.get("eval_all_snippets", False):
        params["dataset_args"]["eval_all_snippets"] = args["eval_all_snippets"]
    
    for key in ["history_max_tokens", "knowledge_max_tokens"]:
        if args.get(key, -1) > -1:
            params["dataset_args"][key] = args[key]


def set_attr_if_not_exists(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def set_default_params(args):
    pass


def set_default_dataset_params(args):
    set_attr_if_not_exists(args, "n_candidates", 1)
    set_attr_if_not_exists(args, "eval_all_snippets", False)
    set_attr_if_not_exists(args, "negative_sample_method", "all")
    set_attr_if_not_exists(args, "history_max_utterances", 100000)
    set_attr_if_not_exists(args, "history_max_tokens", 128)
    set_attr_if_not_exists(args, "knowledge_max_tokens", 128)
