import numpy as np
from numpy import random as rnd
from tqdm import trange

class evaluate:

    def __init__(self, sequence, hidden_states=[0,1], observed_states=[[1,2,3,4,5,6],[1,2,3,4,5,6]],
                 pi=[0.5,0.5], A=[[0.95,0.05],[0.1,0.9]], B=[[1/6,1/6,1/6,1/6,1/6,1/6],[1/10,1/10,1/10,1/10,1/10,1/2]]):

        '''Pass the observed sequence seen, an array of possible values a hidden state can take,
        an array of possible values the observed state can take for each hidden state (in correct order),
        the transition matrix (A), and the emission matrix (B). Defaults are S=[0,1], O=[1,2,3,4,5,6],
        pi=[0.5,0.5], A=[[0.95,0.05],[0.1,0.9]], B=[[1/6,1/6,1/6,1/6,1/6,1/6],[1/10,1/10,1/10,1/10,1/10,1/2]]'''

        self.T = len(sequence)
        A = np.array(A)

        try:
            assert len(B) == len(hidden_states) and len(observed_states) == len(hidden_states)
        except:
            raise ValueError('Error: emission probability matrix of the wrong form, or possible observed states of wrong form.')

        try:
            assert [len(observed_states[i]) == len(B[i]) for i in range(len(B))]
        except:
            raise ValueError('Error: mismatch on observed states length to emission matrix row.')

        'alphas is a matrix with each column the vector of k alpha_k(t) values, and T columns.'
        self.alphas = self.forwards_algorithm(sequence,hidden_states,observed_states,pi,A,B)
        'sum alpha_k(T) over all k for the final time step, T'
        self.likelihood = sum(self.alphas.T[-1])

    def forwards_algorithm(self,sequence,hidden_states,observed_states,pi,A,B):
        alpha_matrix = np.zeros((len(hidden_states),len(sequence)))
        'Initialise the alpha matrix'
        for k,vec in enumerate(observed_states):
            try:
                seq_index = list(vec).index(sequence[0])
                alpha_matrix.T[0][k] = B[k][seq_index] * pi[k]
            except:
                'If the observation cannot come from one of the hidden states, as that is not a correct observation given the state'
                alpha_matrix.T[0][k] = 0
        for t in range(1,self.T):
            B_vec = np.zeros((len(hidden_states),1))
            for k,vec in enumerate(observed_states):
                try:
                    seq_index = list(vec).index(sequence[t])
                    B_vec[k] = B[k][seq_index]
                except:
                    B_vec[k] = 0
            alpha_matrix.T[t] = np.multiply(np.array([A.T @ np.array(alpha_matrix.T[t-1])]).T,np.array(B_vec)).T
        return np.array(alpha_matrix)

    def result(self):
        return self.likelihood, self.alphas


class decode:

    def __init__(self, sequence, form='filtering', hidden_states=[0,1], observed_states=[[1,2,3,4,5,6],[1,2,3,4,5,6]], pi=[0.5,0.5],
                 A=[[0.95,0.5],[0.1,0.9]], B=[[1/6,1/6,1/6,1/6,1/6,1/6],[1/10,1/10,1/10,1/10,1/10,1/2]]):

        '''Pass the observed sequence seen, an array of possible values a hidden state can take,
        an array of possible values the observed state can take for each hidden state (in correct order),
        the transition matrix (A), and the emission matrix (B). Defaults are S=[0,1], O=[1,2,3,4,5,6],
        pi=[0.5,0.5], A=[[0.95,0.05],[0.1,0.9]], B=[[1/6,1/6,1/6,1/6,1/6,1/6],[1/10,1/10,1/10,1/10,1/10,1/2]].
        Also pass the form of decoding desired from: filtering (default), smoothing, viterbi.'''

        self.T = len(sequence) # Becomes unstable if T ~> 400, as norlmalisation term (sum over alphas) -> 0.0
        self.form = form
        A = np.array(A)

        if form == 'filtering':
            self.hidden_state_vector, self.hidden_probabilities = self.filtering(sequence, hidden_states, observed_states, pi, A, B)
        elif form == 'smoothing':
            self.hidden_state_vector, self.hidden_probabilities, self.betas = self.smoothing(sequence, hidden_states, observed_states, pi, A, B)
        elif form == 'viterbi':
            self.hidden_state_vector = self.viterbi(sequence, hidden_states, observed_states, pi, A, B)
        else:
            raise NameError('Sorry, '+form+' is not a supported form of decoding.')

    def filtering(self, sequence, hidden_states, observed_states, pi, A, B):
        alpha_matrix = evaluate(sequence, hidden_states, observed_states, pi, A, B).result()[1]
        filtered = np.zeros((2,self.T))
        for t in range(self.T):
            norm = sum(alpha_matrix.T[t])
            filtered.T[t] = alpha_matrix.T[t] / norm
        try:
            argmax = [hidden_states[list(vec).index(max(vec))] for vec in filtered.T]
        except:
            raise ValueError('Error: impossible observation, not able to fit to hidden state.')
        return argmax, np.array(filtered).astype(float)

    def smoothing(self, sequence, hidden_states, observed_states, pi, A, B):
        alpha_matrix = evaluate(sequence, hidden_states, observed_states, pi, A, B).result()[1]
        'Initialise beta matrix'
        beta_matrix = np.zeros((2,self.T))
        beta_matrix.T[-1] = [1, 1]
        product = np.zeros((2,self.T))
        product.T[-1] = beta_matrix.T[-1] * alpha_matrix.T[-1]
        smoothed = np.zeros((2,self.T))
        smoothed.T[-1] = product.T[-1] / sum(product.T[-1])
        for t in range(2,self.T+1):
            B_vec = np.zeros((len(hidden_states),1))
            for k,vec in enumerate(observed_states):
                try:
                    seq_index = list(vec).index(sequence[-t+1])
                    B_vec[k] = B[k][seq_index]
                except:
                    B_vec[k] = 0
            beta_matrix.T[-t] = (A @ (B_vec * np.array([beta_matrix.T[-t+1]]).T)).T
            product.T[-t] = beta_matrix.T[-t] * alpha_matrix.T[-t]
            smoothed.T[-t] = product.T[-t] / sum(product.T[-t])
        try:
            argmax = [hidden_states[list(vec).index(max(vec))] for vec in smoothed.T]
        except:
            raise ValueError('Error: impossible observation, not able to fit to hidden state.')
        assert [np.isclose(sum(smoothed.T[y]),1) for y in range(self.T)]
        return argmax, np.array(smoothed).astype(float), np.array(beta_matrix).astype(float)

    def viterbi(self, sequence, hidden_states, observed_states, pi, A, B):
        'Initialise the V_matrix with the first entry:'
        V_matrix = np.zeros((2,self.T))
        for k,vec in enumerate(observed_states):
            try:
                seq_index = list(vec).index(sequence[0])
                V_matrix.T[0][k] = B[k][seq_index] * pi[k]
            except:
                V_matrix.T[0][k] = 0
        'Fill values of V_matrix according to mathematical definition:'
        for t in range(1,self.T):
            for k in range(A.shape[0]):
                vec = observed_states[k]
                try:
                    seq_index = list(vec).index(sequence[t])
                    emiss_prob = B[k][seq_index]
                except:
                    emiss_prob = 0
                temp = A.T[k].T * V_matrix.T[t-1]
                V_matrix.T[t][k] = emiss_prob * max(temp)
        'Perform traceback to find the most probable path using V_matrix:'
        S = np.zeros((1,self.T))
        S.T[-1] = hidden_states[list(V_matrix.T[-1]).index(max(V_matrix.T[-1]))]
        for t in range(2,self.T+1):
            opt = list(hidden_states).index(S.T[-t+1])
            temp = A.T[opt].T * V_matrix.T[-t]
            S.T[-t] = hidden_states[list(temp).index(max(temp))]
        return np.array(S).astype(str)

    def result(self,dtype='str'):
        if self.form == 'filtering':
            return self.hidden_state_vector, self.hidden_probabilities
        if self.form == 'smoothing':
            return self.hidden_state_vector, self.hidden_probabilities, self.betas
        else:
            try:
                return np.array(self.hidden_state_vector).astype(dtype)
            except ValueError:
                print('Cannot project to type '+dtype+', string assumed instead.')
                return np.array(self.hidden_state_vector).astype(str)


class learn:

    def __init__(self, sequence, tolerance=1e-3, kill_switch=1e1, hidden_states=[0,1], observed_states=[[1,2,3,4,5,6],[1,2,3,4,5,6]]):

        '''Pass the observed sequence to infer the parameters from, and the desired absolute tolerance
        (if the mean probability change between iterations over all parameters is less than this, learning
        ceases). Incase of poor optimisation, pass a kill switch, the number of iterations used attempting
        to beat the tolerance until the loop is manually broken. Also pass possible hidden states as a 1D array
        and a 2D array of observed states, one row for observations given each hidden state.'''

        pi, A, B = self.initialise(sequence, hidden_states, observed_states)
        self.T = len(sequence)

        self.ticker = 0
        difference = tolerance * 1.1
        #while difference > tolerance and self.ticker < kill_switch:
        for i in trange(int(kill_switch)):
            gammas, xi_matrix = self.E_step(sequence, hidden_states, observed_states, pi, A, B)
            pi_temp, A_temp, B_temp = self.M_step(sequence, gammas, xi_matrix, A, B, observed_states)
            difference = np.mean([np.mean(list(abs(pi - pi_temp))),np.mean(list(abs(A - A_temp))),np.mean(list(abs(B - B_temp)))])
            pi = pi_temp
            A = A_temp
            B = B_temp
            self.ticker += 1
            #print(difference)

        self.pi_opt = pi
        self.A_opt = A
        self.B_opt = B

    def initialise(self, sequence, hidden_states, observed_states):
        'Random initialisation of each from a uniform distribution [0,1] and ensure necessary probabilities sum to 1:'
        pi = np.ones((len(hidden_states),1))
        pi = pi / np.sum(pi)
        A = rnd.rand(len(hidden_states),len(hidden_states))
        A = np.array([vec / np.sum(vec) for vec in A])
        B = []
        for k in range(len(hidden_states)):
            temp = rnd.rand(len(observed_states[k]))
            temp = temp / sum(temp)
            B.append(temp)
        return np.array(pi), np.array(A), np.array(B)

    def E_step(self, sequence, hidden_states, observed_states, pi, A, B):
        alphas = evaluate(sequence, hidden_states, observed_states, pi, A, B).result()[1]
        betas = decode(sequence, 'smoothing', hidden_states, observed_states, pi, A, B).result()[2]
        gammas = decode(sequence, 'smoothing', hidden_states, observed_states, pi, A, B).result()[1]
        xi_matrix = []
        for t in range(self.T-1):
            temp_xi = np.zeros(A.shape)
            B_vec = np.zeros((len(hidden_states),1))
            for k,vec in enumerate(observed_states):
                try:
                    seq_index = list(vec).index(sequence[t+1])
                    B_vec[k] = B[k][seq_index]
                except:
                    B_vec[k] = 0
            for k in range(len(hidden_states)):
                temp_xi[k] = alphas.T[t][k] * np.array(np.array([A[k]]).T * B_vec * np.array([betas.T[t+1]]).T).T
            xi_matrix.append(temp_xi)
        norm = sum(alphas.T[-1])
        'xi_matrix of shape [self.T,K,K], unlike rest of params with self.T as last axis length:'
        xi_matrix = np.array(xi_matrix) / norm
        return np.array(gammas), xi_matrix

    def M_step(self, sequence, gammas, xi_matrix, A_temp, B_temp, observed_states):
        pi = gammas.T[0]
        A = np.zeros(A_temp.shape)
        for k in range(A_temp.shape[0]):
            A[k] = np.sum(xi_matrix, axis=0)[k] / np.sum(gammas[k][0:self.T-1])
        B = []
        for k in range(len(B_temp)):
            delta_matrix = np.zeros((len(B_temp[k]),self.T))
            norm = sum(gammas[k])
            for i in range(len(B_temp[k])):
                delta_matrix[i] = np.where(np.array(sequence) == observed_states[k][i], 1, 0)
            B.append((delta_matrix @ gammas[k]) / norm)
        return np.array(pi), np.array(A), np.array(B)

    def result(self):
        return np.array(self.pi_opt).astype(float), np.array(self.A_opt).astype(float), np.array(self.B_opt).astype(float)

