import tensorflow as tf


def get_args(env, test_run=False):
    
    policy_graph_args = {
        'act_dim': env.action_space.n - 1,
        'clip_range': 0.2,
        'n_hidden': 3,
        'hid_size': 64,
        'conv_depth': 4,
        'learning_rate': 1e-4,
        'num_target_updates': 10,
        'num_grad_steps_per_target_update': 5,
        'actnn_layers': 4,
        'actnn_units': 64,
        'actnn_learning_rate': 1e-6,
        'actnn_nclasses': 5
    }

    adv_args = {
        'gamma': 0.99
    }

    encoder_args = {
        'obs_dim': env.observation_space.shape,
        'act_dim': 5, # number of action classes
        'n_layers_frozen': 10,
        'act_layer_extract': 30,
        'learning_rate': 1e-6,
        'actnn_layers': 2,
        'actnn_units': 64,
        'fsize': 64,
        'conv_depth': 4,
        'n_strides': 1
    }
    target_N = {
        'fsize': 32,
        'conv_depth': 4,
        'n_layers': 1,
        'kernel_init': None
    }
    pred_N = {
        'fsize': 32,
        'conv_depth': 4,
        'n_layers': 1,
        'kernel_init': None
    }
    rnd_args = {
        'learning_rate': 1e-3,
        'out_size': 512,
        'bonus_multiplier': 1,
        'proportion_to_update': .25,
        'target_args': target_N,
        'pred_args': pred_N
    }

    # Train to 1mil iterations -> other papers saw similar results 
    agent_args = {
        'rnd_train_itr': 1,
        'encoder_update_freq': 10,
        'num_random_samples': 10,
        'log_rate': 1,
        'encoder_updates': 100,
        'encoder_train_limit': 500,
        'use_encoder': None
    }

    if test_run:
        policy_graph_args['num_target_updates'] = 1
        policy_graph_args['num_grad_steps_per_target_update'] = 1
        agent_args['rnd_train_itr'] = 1
        agent_args['encoder_train_itr'] = 1
        agent_args['encoder_train_limit'] = 1

    return policy_graph_args, adv_args, encoder_args, rnd_args, agent_args
    