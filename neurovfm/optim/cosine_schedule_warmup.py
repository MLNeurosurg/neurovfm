import math
import logging

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR


# def get_cosine_schedule_with_warmup(optimizer: optim.Optimizer,
#                                     num_warmup_steps: int,
#                                     num_training_steps: int,
#                                     num_cycles: float,
#                                     last_epoch: int = -1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         progress = float(current_step - num_warmup_steps) / float(
#             max(1, num_training_steps - num_warmup_steps))
#         return max(
#             0.0, 0.5 *
#             (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


#     logging.info(f"num_warmup_steps: {num_warmup_steps}")
#     logging.info(f"num_training_steps: {num_training_steps}")
#     logging.info(f"num_cycles: {num_cycles}")

#     return LambdaLR(optimizer, lr_lambda, last_epoch)

# adapted from https://github.com/facebookresearch/ijepa/blob/main/src/utils/schedulers.py
def get_cosine_schedule_with_warmup(optimizer: optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float,
                                    ipe_scale: float = 1.,
                                    ref_lr: float = -1.,
                                    start_lr: float = 0.,
                                    final_lr: float = 0.,
                                    last_epoch: int = -1):

    if ipe_scale > 1.:
        num_warmup_steps = int(num_warmup_steps * ipe_scale)
        num_training_steps = int(num_training_steps * ipe_scale)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    def lr_lambda_nonzero_startend(current_step):
        if current_step < num_warmup_steps:
            return (start_lr + (ref_lr - start_lr) * (float(current_step) / float(max(1, num_warmup_steps)))) / ref_lr
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return (final_lr + (ref_lr - final_lr) * max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))) / ref_lr

    logging.info(f"num_warmup_steps: {int(num_warmup_steps / ipe_scale)}")
    logging.info(f"num_training_steps: {int(num_training_steps / ipe_scale)}")
    logging.info(f"num_cycles: {num_cycles}")
    logging.info(f"ref_lr: {ref_lr}")
    logging.info(f"ipe_scale: {ipe_scale}")

    if (start_lr == 0.) and (final_lr == 0.):
        logging.info(f"Using default lr_lambda")
        return LambdaLR(optimizer, lr_lambda, last_epoch)
    else:
        logging.info(f"Using nonzero start and final lr, {start_lr} and {final_lr}")
        return LambdaLR(optimizer, lr_lambda_nonzero_startend, last_epoch)
        