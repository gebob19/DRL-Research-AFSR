import tensorflow as tf


def get_args(env, test_run=False):
    
    policy_graph_args = {
        'ob_dim': env.observation_space.shape,
        'act_dim': env.action_space.n,
        'clip_range': 0.2,
        'n_hidden': 5,
        'hid_size': 64,
        'learning_rate': 5e-3,
        'num_target_updates': 1,
        'num_grad_steps_per_target_update': 2
    }

    adv_args = {
        'gamma': 0.9999999
    }

    encoder_args = {
        'act_dim': 5, # number of action classes
        'n_layers_frozen': 10,
        'act_layer_extract': 20,
        'learning_rate': 1e-4,
        'actnn_layers': 2,
        'actnn_units': 128
    }

    target_N = {
        'fsize': 32,
        'conv_depth': 4,
        'n_layers': 4,
        'kernel_init': tf.initializers.variance_scaling(scale=20)
    }
    pred_N = {
        'fsize': 32,
        'conv_depth': 4,
        'n_layers': 4,
        'kernel_init': tf.initializers.variance_scaling()
    }
    rnd_args = {
        'learning_rate': 1e-2,
        # paper used 512
        'out_size': 512,
        'bonus_multiplier': 1,
        'bonus_mean': 0,
        'bonus_var': 1,
        'target_args': target_N,
        'pred_args': pred_N
    }
    # Train to 1mil iterations -> other papers saw similar results 
    agent_args = {
        'rnd_train_itr': 1,
        'encoder_train_itr': 1,
        'num_conseq_rand_act': 10,
        'num_random_samples': 10,
        'p_rand': 0.5,                  # p(random action during rollout)
        'algorithm_rollout_rate': 1,
        'log_rate': 1,
    }

    if test_run:
        policy_graph_args['num_target_updates'] = 1
        policy_graph_args['num_grad_steps_per_target_update'] = 1
        agent_args['max_ran_samples'] = 1
        agent_args['num_random_samples'] = 1
        agent_args['rnd_train_itr'] = 1
        agent_args['encoder_train_itr'] = 1

    return policy_graph_args, adv_args, encoder_args, rnd_args, agent_args


    # action_low = 0
    # action_high = env.action_space.n
    # dynamics_graph_args = {
    #     'enc_dim': density_graph_args['z_size'],
    #     'act_dim': 1,
    #     'action_low': action_low,
    #     'action_high': action_high,
    #     'learning_rate': 1e-3,
    #     'hid_size': 64,
    #     'n_hidden': 2 
    # }
    # dynamics_rollout_args = {
    #     'horizon': 5,
    #     'num_rollouts': 50,
    # }