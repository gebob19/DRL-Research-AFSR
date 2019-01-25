def get_args(env):

    policy_graph_args = {
        'ob_dim': env.observation_space.shape,
        'act_dim': env.action_space.n,
        'clip_range': 0.2,
        'n_hidden': 5,
        'hid_size': 32,
        'learning_rate': 5e-3,
        'num_target_updates': 5,
        'num_grad_steps_per_target_update': 5
    }

    adv_args = {
        'gamma': 0.9999999
    }

    rnd_args = {
        'learning_rate': 1e-2,
        # paper used 512
        'out_size': 256,
        'n_layers': 4, 
        'n_hidden': 64,
        'bonus_multiplier': 1
    }

    # Train to 1mil iterations -> other papers saw similar results 
    agent_args = {
        'rnd_train_itr': 2,
        'dynamics_train_itr': 1,
        'num_actions_taken_conseq': 10,
        # hyperparameterized
        'exploitation_threshold': 0,
        'num_random_samples': 10,
        'algorithm_rollout_rate': 2,
        'log_rate': 5
    }

    return policy_graph_args, adv_args, rnd_args, agent_args

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