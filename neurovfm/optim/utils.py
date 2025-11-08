import logging
from typing import Callable, Optional, Dict, Any, Tuple, List

from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

from neurovfm.optim.cosine_schedule_warmup import get_cosine_schedule_with_warmup


def convert_epoch_to_iter(unit, steps, num_it_per_ep):
    if unit == "epoch":
        return num_it_per_ep * steps  # per epoch
    elif unit == "iter":
        return steps
    else:
        NotImplementedError("unit must be one of [epoch, iter]")


def get_optimizer_scheduler_ez(parameters,
                               opt_cf: Dict,
                               num_it_per_ep: int,
                               effective_batch_size: int,
                               num_ep_total: int,
                               schd_cf: Dict = None):
    opt_params = opt_cf["params"]

    if opt_cf.get("scale_lr", False):
        assert "lr" in opt_params

        logging.info(f"scaling learn rate, was {opt_params['lr']}")
        opt_params["lr"] = opt_params["lr"] * effective_batch_size / 256
        logging.info(f"With effective batch size: {effective_batch_size}, " +
                     f"learn rate now {opt_params['lr']}")

    opt_choices = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }

    optimizer = opt_choices[opt_cf["which"]](parameters, **opt_params)

    if not schd_cf:
        return optimizer, None

    # ==========================================================================

    sch_str = schd_cf["which"]
    sch_params = schd_cf["params"]
    required_params = {
        "step_lr": {"step_size", "step_unit", "gamma"},
        "cos_warm_restart": {"t0", "t0_unit", "t_mult", "eta_min"},
        "cos_linear_warmup": {"num_warmup_steps", "num_cycles"}
    }
    assert sch_params.keys() == required_params[sch_str]
    if sch_str == "step_lr":
        step_size = convert_epoch_to_iter(sch_params['step_unit'],
                                          sch_params['step_size'],
                                          num_it_per_ep)
        logging.info("step lr scheduler step size {}".format(step_size))
        scheduler = StepLR(optimizer,
                           step_size=step_size,
                           gamma=sch_params["gamma"])
    elif sch_str == "cos_linear_warmup":
        if sch_params['num_warmup_steps'] < 1:
            sch_params['num_warmup_steps'] = int(
                sch_params['num_warmup_steps'] * num_ep_total * num_it_per_ep)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, sch_params['num_warmup_steps'],
            num_ep_total * num_it_per_ep, sch_params['num_cycles'])
    elif sch_str == "cos_warm_restart":
        t0 = convert_epoch_to_iter(sch_params['t0_unit'], sch_params['t0'],
                                   num_it_per_ep)
        logging.info("cos_warm_restart lr scheduler t0 {}".format(t0))
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=t0,
                                                T_mult=sch_params["t_mult"],
                                                eta_min=sch_params["eta_min"])
    else:
        raise NotImplementedError(
            "Scheduler must be one of [step_lr, cos_warm_restart]")

    return optimizer, scheduler


def get_optimizer_scheduler(model,
                            opt_cf: Dict,
                            num_it_per_ep: int,
                            effective_batch_size: int,
                            num_ep_total: int,
                            schd_cf: Dict = None,
                            normbias_nowd: bool = False):
    opt_str = opt_cf["which"]
    opt_params = opt_cf["params"]

    if opt_cf.get("scale_lr", False):
        assert "lr" in opt_params

        logging.info(f"scaling learn rate, was {opt_params['lr']}")
        for k in ["start_lr", "final_lr", "lr"]:
            opt_params[k] = opt_params[k] * effective_batch_size / 256
        logging.info(f"With effective batch size: {effective_batch_size}, " +
                     f"learn rate now {opt_params['lr']}")

    if opt_str == "sgd":
        if opt_params.get("momentum", 0) == 0:
            for _ in range(99):
                logging.warning(
                    "SGD with no momentum. Are you sure this is what you want?"
                )
        opt_func = optim.SGD
    elif opt_str == "adam":
        opt_func = optim.Adam
    elif opt_str == "adamw":
        opt_func = optim.AdamW
    else:
        raise ValueError(
            "Optimizer must be one of [sgd, adam, adamw]")

    # Build parameter groups with optional norm/bias no weight decay
    if normbias_nowd:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            is_norm = any(k in n.lower() for k in ["norm", "gn", "ln", "bn", "rmsnorm"])
            if p.ndim == 1 or n.endswith(".bias") or is_norm or "pos_emb" in n or "relative_position_bias" in n:
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = opt_func(
            [{"params": decay, "weight_decay": opt_params["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0}],
            **{k:v for k,v in opt_params.items() if k not in ["start_lr", "final_lr", "weight_decay"]}
        )
    else:
        optimizer = opt_func(
            filter(lambda p: p.requires_grad, model.parameters()),
            **{k:v for k,v in opt_params.items() if k not in ["start_lr", "final_lr"]})

    if not schd_cf:
        return optimizer, None

    # ==========================================================================

    sch_str = schd_cf["which"]
    sch_params = schd_cf["params"]
    required_params = {
        "step_lr": {"step_size", "step_unit", "gamma"},
        "cos_warm_restart": {"t0", "t0_unit", "t_mult", "eta_min"},
        "cos_linear_warmup": {"num_warmup_steps", "num_cycles", "ipe_scale"}
    }
    assert sch_params.keys() == required_params[sch_str]
    if sch_str == "step_lr":
        step_size = convert_epoch_to_iter(sch_params['step_unit'],
                                          sch_params['step_size'],
                                          num_it_per_ep)
        logging.info("step lr scheduler step size {}".format(step_size))
        scheduler = StepLR(optimizer,
                           step_size=step_size,
                           gamma=sch_params["gamma"])
    elif sch_str == "cos_linear_warmup":
        if sch_params['num_warmup_steps'] < 1:
            sch_params['num_warmup_steps'] = int(
                sch_params['num_warmup_steps'] * num_ep_total * num_it_per_ep)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, sch_params['num_warmup_steps'],
            num_ep_total * num_it_per_ep, sch_params['num_cycles'], sch_params["ipe_scale"],
            opt_params["lr"], opt_params.get("start_lr", 0.), opt_params.get("final_lr", 0.))
    elif sch_str == "cos_warm_restart":
        t0 = convert_epoch_to_iter(sch_params['t0_unit'], sch_params['t0'],
                                   num_it_per_ep)
        logging.info("cos_warm_restart lr scheduler t0 {}".format(t0))
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=t0,
                                                T_mult=sch_params["t_mult"],
                                                eta_min=sch_params["eta_min"])
    else:
        raise NotImplementedError(
            "Scheduler must be one of [step_lr, cos_warm_restart]")

    return optimizer, scheduler
