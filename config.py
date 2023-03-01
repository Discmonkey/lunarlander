# discounted reward constant
gamma = 0.999
# epsilon value during first episode, 1.0 means that every action chosen will be random
starting_epsilon = 1.00
# new epsilon value is set according to:
# epsilon = max(episode / epsilon_update_schedule * starting_epsilon, 0.1)
epsilon_update_schedule = 100
# the maximum number of steps in a single episode
max_steps = 1000
# clip the squared error (-1, 1)
clip_error = False
# how often we show the game during training
show_every = 10
# mini-batch size for training
mini_batch_size = 64
# learning rate
lr = 0.25
# momentum
momentum = 0.0
# training epochs
training_epochs = 100
