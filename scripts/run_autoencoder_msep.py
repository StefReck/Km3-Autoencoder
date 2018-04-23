# -*- coding: utf-8 -*-

"""
Does training with msep loss.
"""
from run_autoencoder import execute_training, unpack_parsed_args

ae_loss_name = "mean_squared_error_poisson"
execute_training(*unpack_parsed_args(), ae_loss_name)
