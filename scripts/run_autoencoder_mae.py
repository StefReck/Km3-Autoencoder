# -*- coding: utf-8 -*-

"""
Does training with mae loss.
"""
from run_autoencoder import execute_training, unpack_parsed_args

ae_loss_name = "mae"
execute_training(*unpack_parsed_args(), ae_loss_name)
