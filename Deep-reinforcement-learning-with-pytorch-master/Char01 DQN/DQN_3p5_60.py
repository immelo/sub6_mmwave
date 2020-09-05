import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import h5py
import scipy
from scipy.io import loadmat
from torch.autograd import Variable

# hyper-parameters
BATCH_SIZE = 512
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 50000
Q_NETWORK_ITERATION = 100

dataset = h5py.File(r'F:\deepmimo\deepmimo\DeepMIMO-codes-master\DeepMIMO-codes-master\DeepMIMO_Dataset_Generation_v1.1\Sub6-Preds-mmWave-master\Sub6-Preds-mmWave-master\DeepMIMO_dataset_yrm_nonoisy.mat','r')
# dataset = loadmat(r'F:\deepmimo\deepmimo\DeepMIMO-codes-master\DeepMIMO-codes-master\DeepMIMO_Dataset_Generation_v1.1\Sub6-Preds-mmWave-master\Sub6-Preds-mmWave-master\DeepMIMO_dataset_yrm.mat')
#状态空间是76146个位置的3p5观察值,inpTrain维度是1*1*128*76146
data = dataset['dataset']
state_channel_space = data['inpVal']
state_loc_space = data['valInpLoc']
#动作空间是64个码本码字
action_space = range(64)
#奖励空间是64个码字对应的速率
reward_space = data['codebookVal']
max_reward_space = data['maxRateVal']
# env = gym.make("CartPole-v0")
# env = env.unwrapped
NUM_ACTIONS = 63
NUM_STATES = state_channel_space.shape[0]

#ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(261, 2048)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(2048,2048)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(2048,1024)
        self.fc3.weight.data.normal_(0,0.1)
        self.out = nn.Linear(1024,64)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = x.cuda()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net = self.eval_net.cuda()
        self.target_net = self.target_net.cuda()
        # print(next(self.eval_net.parameters()).is_cuda, next(self.target_net.parameters()).is_cuda)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 261 * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss().cuda()

    def choose_action(self, state):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0)).cuda() # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action_value_max1 = torch.max(action_value,1)[0]
            action_value_max2 = torch.max(action_value_max1,1)[1].cpu().numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            action = action_value_max2[0]
        else: # random policy
            action = np.random.randint(NUM_ACTIONS)
            # action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
            action = action
        return action


    def store_transition(self, state, action, reward, next_state):
        action_reward = np.vstack((action, reward)).reshape(1,2)
        transition = np.hstack((state, action_reward, next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = Variable(torch.FloatTensor(batch_memory[:, :261])).cuda(0)
        batch_action = Variable(torch.LongTensor(batch_memory[:, 261:262].astype(int))).cuda(0)
        batch_reward = Variable(torch.FloatTensor(batch_memory[:, 262:263])).cuda(0)
        batch_next_state = Variable(torch.FloatTensor(batch_memory[:,-261:])).cuda(0)

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_eval, q_target = q_eval.cuda(), q_target.cuda()
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

def reward_func(index_user, action):
    reward = reward_space[action, int(index_user)]
    return reward

def main():
    dqn = DQN()
    episodes = 100
    print("Collecting Experience....")
    reward_list = []
    regret_list =[]

    for i in range(episodes):
        for index_user in range(32635):
            loc_zero = state_loc_space[index_user,:].reshape(1,3)
            channel_zero = state_channel_space[index_user,:].reshape(1,256)
            ob_zero = np.hstack((loc_zero, channel_zero))
            action_zero = np.random.randint(0,NUM_ACTIONS)
            reward_zero = reward_space[action_zero, index_user]
            reaction_zero = np.vstack((action_zero, reward_zero)).reshape(1,2)
            state = np.hstack((ob_zero,reaction_zero))
            # state = [ob_zero, action_zero, reward_zero]
            ep_reward = 0
            num_action = 3
            best_reward = max_reward_space[:,index_user][0]
            while num_action > 0 :
                action = dqn.choose_action(state)
                reward = reward_func(index_user, action)
                reaction_now = np.vstack((action, reward)).reshape(1,2)
                next_state =  np.hstack((ob_zero,reaction_now))

                dqn.store_transition(state, action, reward, next_state)
                ep_reward += reward
                regret = best_reward*(4-num_action) - ep_reward
                if dqn.memory_counter >= MEMORY_CAPACITY:
                    learnning_loss = dqn.learn()
                    print("episode: {} , the episode reward is {}, the regret is {}, thr loss is {}".format(i, round(ep_reward, 3), round(regret,3), learnning_loss))
                state = next_state
                num_action -= 1
            r = copy.copy(ep_reward)
            reg = copy.copy(regret)
            reward_list.append(r)
            regret_list.append(reg)
    
    # torch.save({'eval_net':self.eval_net, 'target_net':self.target_net, 'optim':self.optimizer},'/DQN_3p5_60.pkl')
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0,3263500)
    #ax.cla()
    plt.plot(reward_list, 'g-', label='total_loss')
    plt.plot(regret_list, 'r-', label='regret')
    plt.show
        

if __name__ == '__main__':
    main()