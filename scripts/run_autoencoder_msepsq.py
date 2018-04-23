# -*- coding: utf-8 -*-

"""
Does training with msepsq loss.
"""
from run_autoencoder import execute_training, unpack_parsed_args

ae_loss_name = "msep_squared"
execute_training(*unpack_parsed_args(), ae_loss_name)
