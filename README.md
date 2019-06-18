# Agent-Focused State Representations
Brennan Gebotys

## Introduction
Learning an accurate state representation quickly is an important task for all deep reinforcement learning agents.

A representation with too much information can hinder the learning process while a representation with too little information can lead to the task being unsolvable. 

For the standard actor-critic method, it can be viewed that some layer in the actor-network represents a lower level state and the later layers then transform this representation into a distribution over all actions. This is learned through backpropagation, so the policy must learn a correct action distribution for a given state and how to represent the state at the same time.

A popular method for learning a lower level state representation is to use a variational-autoencoder (VAE). The states are first encoded using the VAE and then are used to train the policy. 

This method simplifies the task for the policy by providing the policy with a low-level state representation allowing it to focus solely on how to accomplish the task. In doing so, it also limits the information available to the policy to what is encoded and adds the overhead of ensuring the autoencoder’s high accuracy throughout the training process. 

In general, the most basic video game states can be represented as the image being displayed on the screen. This usually includes the agent’s in-game character, objectives to achieve, and obstacles to avoid.

At the heart of the proposed implementation, I focus on the fact that the player which our agent controls is clearly a required factor to include in an accurate lower level state representation.

The proposed implementation aims to quickly learn an agent-focused state representation allowing the agent to accomplish the given task faster, while not limiting the information available to the policy, all with minimal additional computational cost.

## Algorithm

![alt text](https://github.com/gebob19/PredictiveExploration/blob/master/codebase/imgs/diagram.png) 

To learn an agent-focused state representation we train a shallow neural network which extends out from the first half of the policy. It is trained given two state, sand *s’* where *s’* was reached by taking action *a* at state *s*. We first pass both *s* and *s’* through the first half of our policy, to get *e(s)*, *e(s’)* (some encodings of *s* and *s’*). 

We then concatenate both encodings, which we will refer to as *e(s, s’)* and pass it through a shallow neural network which will predict *p(a | e(s, s’))*. The results are then backpropagated through this shallow neural network, and through the first half of our policy.

To predict this accurately the network must be able to extract information about the agent itself, and in doing so will hopefully learn a better state representation.

## Setup

I used a standard actor-critic algorithm setup which uses random network distillation (RND) to encourage the agent to explore. For an environment, I used Montezuma’s Revenge from OpenAI’s gym. More specific details can be viewed in the source code.

## Results

Three experiments were tested, two used the proposed implementation (AFSR+RND_1.2-1.7, AFSR+RND_1.5-1.9) and one as the standard RND setup. All experiments were trained for 100 iterations. The table below represents the average values over all rollouts.

*Note:* For the proposed implementation's naming convention (ex. AFSR+RND_1.2-1.7), the first number is the initial mean loss I trained the encoder too before training the policy (1.2), and the second number is the mean loss threshold which was maintained throughout rollouts (1.7).

![alt text](https://github.com/gebob19/PredictiveExploration/blob/master/codebase/imgs/100itr_table.png)
![alt text](https://github.com/gebob19/PredictiveExploration/blob/master/codebase/imgs/denisty_policy_loss.png)
![alt text](https://github.com/gebob19/PredictiveExploration/blob/master/codebase/imgs/encoder_rew_loss.png)

## Analysis
From the table, we see AFSR+RND achieves a better average internal reward than standard RND over 100 iterations. Which could be because using AFSR, the task is simplified for the policy, allowing it to maximize internal rewards faster.

We also see that the hyperparameters of the encoder loss thresholds must be fine-tuned for optimal results, which could be a disadvantage. 

With a loss threshold of 1.7, we see that we actually limit the policy too much, leading to a higher loss score for our policy. This could be because the policy is forced to focus too much on predicting the agent’s actions, rather than learning how to complete the task. 

With a threshold of 1.9, we relax the policies focus on the agent, allowing it to accomplish the task faster. This is seen since its actor loss is the best of the three tests. 

However since we have only tested each setup once with only 100 training iterations the results are still inconclusive, and more testing should be done.

