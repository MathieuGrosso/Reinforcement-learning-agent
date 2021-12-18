import numpy as np
from icecream import ic
import torch 


class PPOMemory():
    def __init__(self, mem_size: int, gamma, tau, batch_size: int) -> None:
        self.buffer = []
        self.nentries = 0
        self.mem_size = mem_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def store(self, transition):
        self.buffer.append(transition)
        self.nentries += 1

    def edit_last_transition(self, **kwargs):
        self.buffer[-1]["reward"] = kwargs["reward"]
        self.buffer[-1]["new_ob"] = kwargs["new_ob"]
        self.buffer[-1]["done"] = kwargs["done"]
        self.buffer[-1]["step"] = kwargs["step"]

    def compute_gae(self):
        """
            Compute generalized advantage estimate which is basically an exponential average
            of the TD(n) advantage function
                if tau set to 1:  It is a regular TD(n) advantage function
                if tau set to 0: It is a TD(0) advantage function
        """
        self.buffer[-1]['advantage'] = torch.tensor([0])
        for step in reversed(range(len(self.buffer) - 1)):
            delta = self.buffer[step]['reward'] + self.gamma * (self.buffer[step+1]['value'] * \
                (1 - int(self.buffer[step]['done'])) - self.buffer[step]['value'])
            self.buffer[step]['advantage'] = delta + self.gamma * self.tau * \
                 self.buffer[step + 1]['advantage'] *  (1 - int(self.buffer[step]['done']))
                
    def generate_batches(self):
        obs = torch.vstack([x["ob"] for x in self.buffer]).detach()
        old_probs = torch.vstack([x["log_prob"] for x in self.buffer]).detach()
        actions = torch.vstack([x["action"] for x in self.buffer]).detach()
        advantage = torch.vstack([x["advantage"] for x in self.buffer]).detach()
        value = torch.vstack([x["value"] for x in self.buffer]).detach()
        for _ in range(len(self.buffer) // self.batch_size):
            id_batch = np.random.randint(0, len(self.buffer), self.batch_size)
            yield obs[id_batch,:], old_probs[id_batch,:], actions[id_batch,:], advantage[id_batch,:], value[id_batch,:]

    def clear(self):
        self.buffer = []
        self.nentries = 0


class MemoryPPO:
    def __init__(self):
        self.memory = []
        self.nentries = 0 

    def store(self, transition):
        self.memory.append(transition)
        self.nentries += 1 

    def clear(self):
        self.memory = []
        self.nentries = 0

   

    def get_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst,log_prob_lst, trunc_lst = [], [], [], [], [], [],[]

        for transition in self.memory:
            s, a, r, s_prime, done,oldprob, info,trunc = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            log_prob_lst.append([oldprob])
            # trunc_lst.append([info.get("TimeLimit.truncated", False)])  # True if the episode was truncated (time limit)
        self.memory = []

        # N, dim_a = len(a_lst), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float).detach(),  # (N, dim_s)
                torch.tensor(a_lst).detach(),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float).detach(),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float).detach(),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float).detach(),# (N, 1)
                torch.tensor(log_prob_lst,dtype=torch.float).detach() )
 

class SumTree:
    def __init__(self, mem_size):
        self.tree = np.zeros(2 * mem_size - 1)
        self.data = np.zeros(mem_size, dtype=object)
        self.size = mem_size
        self.ptr = 0
        self.nentities=0


    def update(self, idx, p):
        tree_idx = idx + self.size - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def store(self, p, data):
        self.data[self.ptr] = data
        self.update(self.ptr, p)
        idx=self.ptr
        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
        self.nentities+=1
        if self.nentities > self.size:
            self.nentities = self.size
        return idx

    def getNextIdx(self):
        return self.ptr

    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1

        return ptr - (self.size - 1), self.tree[ptr], self.data[ptr - (self.size - 1)]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def max_p(self):
        return np.max(self.tree[-self.size:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.size:])


class Memory:

    def __init__(self, mem_size, prior=True,p_upper=1.,epsilon=.01,alpha=1,beta=1):
        self.p_upper=p_upper
        self.epsilon=epsilon
        self.alpha=alpha
        self.beta=beta
        self.prior = prior
        self.nentities=0
        #self.dict={}
        # self.data_len = 2 * feature_size + 2
        self.mem_size = mem_size
        if prior:
            self.tree = SumTree(mem_size)
        else:

            self.mem = np.zeros(mem_size, dtype=object)
            self.mem_ptr = 0

    #def getID(self,transition):
    #    ind=-1
    #    if transition in dict:
    #        ind = dict[transition]
    #    return ind


    def store(self, transition):
        if self.prior:
            p = self.tree.max_p
            if not p:
                p = self.p_upper
            idx=self.tree.store(p, transition)
            self.nentities += 1
            if self.nentities > self.mem_size:
                self.nentities = self.mem_size
        else:
            self.mem[self.mem_ptr] = transition
            idx=self.mem_ptr
            self.mem_ptr += 1

            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
            self.nentities += 1
            if self.nentities > self.mem_size:
                self.nentities = self.mem_size
        return idx
        

    def sample(self, n):
        if self.prior:
            min_p = self.tree.min_p
            if min_p==0:
                min_p=self.epsilon**self.alpha
            seg = self.tree.total_p / n
            batch = np.zeros(n, dtype=object)
            w = np.zeros((n, 1), np.float32)
            idx = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + seg
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.sample(v)

                w[i] = (p / min_p) ** (-self.beta)
                a += seg
            return idx, w, batch
        else:
            mask = np.random.choice(range(self.nentities), n)
            return mask, 0,  self.mem[mask]

    def update(self, idx, tderr):
        if self.prior:
            tderr += self.epsilon
            tderr = np.minimum(tderr, self.p_upper)
            #print(idx,tderr)
            for i in range(len(idx)):
                self.tree.update(idx[i], tderr[i] ** self.alpha)

    def getNextIdx(self):
        if self.prior:
            ptr=self.tree.getNextIdx()
        else:
            ptr=self.mem_ptr
        return ptr

    def getData(self,idx):
        if idx >=self.nentities:
            return None
        if self.prior:
            data=self.tree.data[idx]
        else:
            data=self.mem[idx]
        return data

    def getBatch(self,batch_size):
        idx,_,batch = self.sample(batch_size)
        ob_batch = torch.vstack([x[0] for x in batch])
        action_batch =  torch.vstack([x[1] for x in batch],dtype=torch.float())
        reward_batch = torch.vstack([x[2] for x in batch],dtype=torch.float())
        new_ob_batch = torch.vstack([x[3] for x in batch],dtype=torch.float())
        done_batch   = torch.vstack([x[4] for x in batch], dtype=torch.float())
        logprob_batch = torch.vstack([x[5] for x in batch], dtype=torch.float())
        return ob_batch,action_batch,reward_batch,new_ob_batch,done_batch,logprob_batch