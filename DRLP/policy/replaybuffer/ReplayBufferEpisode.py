import random
import numpy as np
from collections import deque


class ReplayBufferEpisode(object):

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.count = 0
        self.buffer = deque()
        self.episode = []
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
            None, None, None, None, None, None

        random.seed(random_seed)

    def record(self, state, state_ori, action, reward, value, distribution, terminal=False):
        """
        Record the experience:
            Turn #  User: state         System: action                  got reward
            Turn 1  User: Cheap         System: location?               -1
            Turn 2  User: North         System: inform(cheap, north)    -1
            Turn 3  User: Bye           System: inform(XXX) --> bye     -1
            Turn 4  User: Terminal_s    System: terminal_a              20

        As:
            Experience 1: (Cheap, location?, -1, North)
            Experience 2: (North, inform(cheap, north), -1+20, Bye)
        """
        if self.s_prev is None and self.s_ori_prev is None and self.a_prev is None and self.r_prev is None and \
                self.v_prev is None and self.distribution_prev is None:
            self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
                state, state_ori, action, reward, value, distribution
            return
        else:
            if terminal:
                try:
                    # add dialogue succes reward to last added experience , -1 for goodbye turn
                    self.episode[-1][3] += reward
                    # change this experience to terminal
                    self.episode[-1][-2] = terminal
                    # add episodic experience to buffer
                    if self.count < self.buffer_size:
                        self.buffer.append(self.episode)
                        self.count += 1
                    else:
                        self.buffer.popleft()
                        self.buffer.append(self.episode)

                    self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
                        None, None, None, None, None, None
                    self.episode = []
                except IndexError:
                    self.episode = []
            else:
                self.episode.append(
                    [self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev,
                     state, state_ori, self.v_prev, terminal, self.distribution_prev])
                self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
                    state, state_ori, action, reward, value, distribution

    def size(self):
        return self.count

    def sample_batch(self):
        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        batch = np.array(batch)

        s_batch = []
        s_ori_batch = []
        a_batch = []
        r_batch = []
        s2_batch = []
        s2_ori_batch = []
        v_batch = []
        t_batch = []
        mu_batch = []

        for epi in batch:
            tmp_s, tmp_s_ori, tmp_a, tmp_r, tmp_s2, tmp_s2_ori, tmp_v, tmp_t, tmp_mu = \
                [], [], [], [], [], [], [], [], []
            for exp in epi:
                tmp_s.append(exp[0])
                tmp_s_ori.append(exp[1])
                tmp_a.append(exp[2])
                tmp_r.append(exp[3])
                tmp_s2.append(exp[4])
                tmp_s2_ori.append(exp[5])
                tmp_v.append(exp[6])
                tmp_t.append(exp[7])
                tmp_mu.append(exp[8])

            s_batch.append(tmp_s)
            s_ori_batch.append(tmp_s_ori)
            a_batch.append(tmp_a)
            r_batch.append(tmp_r)
            s2_batch.append(tmp_s2)
            s2_ori_batch.append(tmp_s2_ori)
            v_batch.append(tmp_v)
            t_batch.append(tmp_t)
            mu_batch.append(tmp_mu)

        return s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, None, v_batch, mu_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.episode = []
        self.s_prev, self.s_ori_prev, self.a_prev, self.r_prev, self.v_prev, self.distribution_prev = \
            None, None, None, None, None, None
