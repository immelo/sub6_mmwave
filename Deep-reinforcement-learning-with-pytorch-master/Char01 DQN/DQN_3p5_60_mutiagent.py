import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import h5py
import scipy
import os
import torch.multiprocessing as mp
import threading
from scipy.io import loadmat
from torch.autograd import Variable

# hyper-parameters
BATCH_SIZE = 512
LR = 0.001
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 100

dataset = h5py.File(r'F:\deepmimo\deepmimo\DeepMIMO-codes-master\DeepMIMO-codes-master\DeepMIMO_Dataset_Generation_v1.1\Sub6-Preds-mmWave-master\Sub6-Preds-mmWave-master\DeepMIMO_dataset_yrm_nonoisy_beamidmodify.mat','r')
# dataset = loadmat(r'F:\deepmimo\deepmimo\DeepMIMO-codes-master\DeepMIMO-codes-master\DeepMIMO_Dataset_Generation_v1.1\Sub6-Preds-mmWave-master\Sub6-Preds-mmWave-master\DeepMIMO_dataset_yrm.mat')
#状态空间是76146个位置的3p5观察值,inpTrain维度是1*1*128*76146
data = dataset['dataset']
state_channel_space = data['inpVal']
state_loc_space = data['valInpLoc']
#动作空间是64个码本码字
action_space = torch.LongTensor(np.arange(64))
action_space_onehot = torch.nn.functional.one_hot(action_space, 64)
#奖励空间是64个码字对应的速率
reward_space = data['codebookVal']
label_space = data['labelval']
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
        self.fc1 = nn.Linear(324, 2048)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(2048,2048)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(2048,2048)
        self.fc3.weight.data.normal_(0,0.1)
        self.fc4 = nn.Linear(2048,1024)
        self.fc4.weight.data.normal_(0,0.1)
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
        x = self.fc4(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net = self.eval_net.cuda()
        # self.eval_net.share_memory()
        self.target_net = self.target_net.cuda()
        # self.target_net.share_memory()
        # print(next(self.eval_net.parameters()).is_cuda, next(self.target_net.parameters()).is_cuda)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 324 * 2 + 65))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        # evel_net_optim = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # self.optimizer = torch.optim.lr_scheduler.MultiStepLR(evel_net_optim, milestones=[20000,30000,40000] , gamma= 0.5)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR, momentum=0.9)
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
        action_reward = np.hstack((action, reward)).reshape(1,65)
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
        batch_state = Variable(torch.FloatTensor(batch_memory[:, :324])).cuda(0)
        batch_action = Variable(torch.LongTensor(batch_memory[:, 324:388].astype(int))).cuda(0)
        batch_reward = Variable(torch.FloatTensor(batch_memory[:, 388:389])).cuda(0)
        batch_next_state = Variable(torch.FloatTensor(batch_memory[:,-324:])).cuda(0)

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

def reward_func(index_user, reward_space_num, action):
    reward = reward_space_num[int(index_user)][action]
    return reward

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max -min)

def store_transition_and_learn(i, dqn_num, reward_space_num, state, index_user, ob_zero, ep_reward, num_action, agent_num):
    action_index = dqn_num.choose_action(state)
    action = action_space_onehot[action_index]
    reward = reward_func(index_user, reward_space_num, action_index)
    reaction_now = np.hstack((action, reward)).reshape(1,65)
    next_state =  np.hstack((ob_zero, reaction_now))

    dqn_num.store_transition(state, action, reward, next_state)
    ep_reward += reward
    # regret = best_reward*(4-num_action) - ep_reward
    if dqn_num.memory_counter >= MEMORY_CAPACITY:
        learnning_loss = dqn_num.learn()
        print(" episodes:{}, agent: {} , ep_reward {}, loss {}".format(i, agent_num, round(ep_reward, 3), learnning_loss))
    state = next_state
    num_action -= 1
    return num_action, ep_reward, state

def process_every_network(process_num, dqn_num, state_loc_space_num, state_channel_space_num, reward_space_num, label_space_num, reward_list_num):
    # episodes = 500
    # NUM_STATES_sublist = len(state_loc_space_num)
    for i in range(500):
        for index_user in range(len(state_loc_space_num)):
            loc_zero = state_loc_space_num[index_user].reshape(1,3)
            channel_zero = state_channel_space_num[index_user].reshape(1,256)
            ob_zero = np.hstack((loc_zero, channel_zero))

            action_zero_index = np.random.randint(0,NUM_ACTIONS)
            action_zero = action_space_onehot[action_zero_index]

            best_reward_index = int(label_space_num[index_user][0])
            best_reward = reward_space_num[index_user][best_reward_index]
 
            reward_zero = reward_space_num[index_user][action_zero_index]
            reaction_zero = np.hstack((action_zero, reward_zero)).reshape(1,65)
            state = np.hstack((ob_zero, reaction_zero))
            # state = [ob_zero, action_zero, reward_zero]
            ep_reward = 0
            num_action = 3
            
            while num_action > 0:
                num_action, ep_reward, state = store_transition_and_learn(i, dqn_num, reward_space_num, state, index_user, ob_zero, ep_reward, num_action, agent_num =process_num)
                r = copy.copy(ep_reward)
                reward_list_num.append(r)      

def main():
    print("preparing data....")
    class_list_list = []
    state_loc_space_list = []
    state_channel_space_list = []
    reward_space_list = []
    label_space_list = []
    for i in range(8):
        globals()['dqn_' + str(i)] = DQN()
        globals()['reward_list_' + str(i)] = []
        globals()['regret_list_' + str(i)] = []
        globals()['class_list_' + str(i)] = np.arange(8) + 1 + i*8
        class_list_list.append(globals()['class_list_' + str(i)])
        globals()['state_loc_space_' + str(i)] = []
        state_loc_space_list.append(globals()['state_loc_space_' + str(i)])
        globals()['state_channel_space_' + str(i)] = []
        state_channel_space_list.append(globals()['state_channel_space_' + str(i)])
        globals()['reward_space_' + str(i)] = []
        reward_space_list.append(globals()['reward_space_' + str(i)])
        globals()['label_space_' + str(i)] = []
        label_space_list.append(globals()['label_space_' + str(i)])

    for index_user in range(NUM_STATES):
        class_index = label_space[:, index_user]
        for i in range(8):
            if class_index in globals()['class_list_' + str(i)]:
                globals()['state_loc_space_' + str(i)].append(state_loc_space[index_user,:])
                globals()['state_channel_space_' + str(i)].append(state_channel_space[index_user,:])
                globals()['reward_space_' + str(i)].append(reward_space[:,index_user])
                globals()['label_space_' + str(i)].append(label_space[:,index_user])
    
    # arg_list = [(0, dqn_0, state_loc_space_0, state_channel_space_0, reward_space_0, reward_list_0),
    #             (1, dqn_1, state_loc_space_1, state_channel_space_1, reward_space_1, reward_list_1),
    #             (2, dqn_2, state_loc_space_2, state_channel_space_2, reward_space_2, reward_list_2),
    #             (3, dqn_3, state_loc_space_3, state_channel_space_3, reward_space_3, reward_list_3),
    #             (4, dqn_4, state_loc_space_4, state_channel_space_4, reward_space_4, reward_list_4),
    #             (5, dqn_5, state_loc_space_5, state_channel_space_5, reward_space_5, reward_list_5),
    #             (6, dqn_6, state_loc_space_6, state_channel_space_6, reward_space_6, reward_list_6),
    #             (7, dqn_7, state_loc_space_7, state_channel_space_7, reward_space_7, reward_list_7]
    
    print("Collecting Experience....")
    for i in range(len(reward_space_6)):
        reward_space_6[i] = minmaxscaler(reward_space_6[i])

    process_every_network(6, dqn_6, state_loc_space_6, state_channel_space_6, reward_space_6, label_space_6, reward_list_6,)
  
    
    # mp.set_start_method('spawn')
    # processes = []
    # for rank in range(8):
    #     p = mp.Process(target=process_every_network, args=(arg_list[i]))
    #     p.start()
    #     processes.append(p)
    
    # print('waiting for all subprocesses done……')    

    # for p in processes:
    #     p.join()

    # threads = []
    # for rank in range(8):
    #     t = threading.Thread(target=process_every_network, args=(arg_list[i]))
    #     t.start()
    #     threads.append(t)
    
    # print('waiting for all subprocesses done……')    

    # for t in threads:
    #     t.join()

    print('All subprocesses done')

    # torch.save({'eval_net':self.eval_net, 'target_net':self.target_net, 'optim':self.optimizer},'/DQN_3p5_60.pkl')
    plt.title('Result Analysis')
    plt.plot(reward_list_0, 'g-', label='agent_0_regret')
    plt.plot(regret_list_1, 'r-', label='agent_1_regret')
    plt.plot(regret_list_2, 'k-', label='agent_2_regret')
    plt.plot(regret_list_3, 'b-', label='agent_3_regret')
    plt.plot(regret_list_4, 'y-', label='agent_4_regret')
    plt.plot(regret_list_5, 'c-', label='agent_5_regret')
    plt.plot(regret_list_6, 'm-', label='agent_6_regret')
    plt.plot(regret_list_7, 'p-', label='agent_7_regret')
    plt.legend()
    plt.savefig('result.jpg')
    plt.show()
    torch.save({'eval_net':self.eval_net, 'target_net':self.target_net, 'optim':self.optimizer},'/DQN_3p5_60.pkl')
        

if __name__ == '__main__':
    main()