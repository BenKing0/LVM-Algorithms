import numpy as np
import random


class generate:
    '''A code to generate a sequence of observations from a hidden Markov model.
    Unless specified, the hidden state sample space is binary [0,1], and the observation state
    sample space is [1,6], with uniform distribution for S_i=0, or skewed distribution if S_t=1.
    The Markov chain is assumed to be first-order unless also specified under `order' parameter.'''

    def __init__(self, T, order=1, hidden_states=[0,1], pi=[0.5,0.5], A=[[0.95,0.05],[0.10,0.90]],
                 observed_states=[{1:1/6,2:1/6,3:1/6,4:1/6,5:1/6,6:1/6}, {1:1/10,2:1/10,3:1/10,4:1/10,5:1/10,6:1/2}]):
        '''Pass observed states as an array of dicts with the key being the state and the value being
        its probability for each dict in the array (one for each hidden state value).
        In contrast, pass hidden states as a list of states, and the transition matrix, A, for them in the
        same order as they appear in the list. Finally, pass pi as an array of initial prob. for each hidden state.'''

        hidden_states = np.array(hidden_states)
        A = np.array(A)
        pi = np.array(pi)

        try:
            assert A.shape[0] == len(hidden_states) ** order
        except:
            print('Error: the dimensions of the transition matrix does not fit the Markov chain order.')

        try:
            assert [np.isclose(sum(A[i]), 1.0, atol=1e-4) for i in range(A.shape[0])]
        except:
            print('Error: the transition probabilities from each vertex do not sum to 1.')

        S = self.hidden_sequence(T,hidden_states,A,pi)
        self.O = self.observe(T,S,observed_states)
        self.hidden_sequence_for_testing = S

    def hidden_sequence(self,T,hidden_states,A,pi):
        ind = random.choices(hidden_states,weights=pi)[0]
        state = hidden_states[ind]
        sequence = [state]
        for i in range(1,T):
            'Select whether to stay on state or leave to a new state, governed by the current state transition prob.'
            state = random.choices(hidden_states, weights=A[ind], k=1)[0] # replaces by default
            ind = list(hidden_states).index(state)
            sequence.append(state)
        return sequence

    def observe(self,T,S,observed_states):
        sequence = []
        for i in range(T):
            'Select the relevant dictionary based on the current hidden state'
            hidden_dict = observed_states[S[i]]
            'Transform the dictionary to a lookup table (actually an array) of states and their probability'
            observation_array = np.array(list(hidden_dict.items())).T
            ps = observation_array[1].astype(float)
            states = observation_array[0].astype(str)
            state = random.choices(states, weights=ps)[0]
            sequence.append(state)
        return sequence

    def result(self,dtype='str'):
        ''''Returns the output of the HMM, the raw observed data. Choice to specify output datatype,
        default is 'str.'''
        try:
            return np.array(self.O).astype(dtype)
        except ValueError:
            print('Cannot project to type '+dtype+', string assumed instead.')
            return np.array(self.O).astype(str)
        except:
            print('Unkown error with output type')
            return []

    def return_hidden_sequence(self):
        return self.hidden_sequence_for_testing

