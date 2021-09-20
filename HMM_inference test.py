import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from HMM_generator import generate
from HMM_inference import evaluate, decode, learn
from HMM_inference_naive import decode as naive
import numpy as np
import matplotlib.pyplot as plt

model = generate(100) # Numerical underflow after ~400 sequence length
series = model.result('float')

actual = model.return_hidden_sequence()
evaluation = evaluate(series).result()
predict_sm = decode(series,'smoothing').result()[1][1]
predict_sm_ni = naive(series,'smoothing').result()[1][1]
predict_fi = decode(series,'filtering').result()[1][1]
predict_fi_ni = naive(series,'filtering').result()[1][1]
#predict_vi = decode(series,'viterbi').result('float').T

#%%
plt.plot(np.arange(len(predict_fi)),predict_fi,'b--',label='Prob. of dice 1, filtering',alpha=0.4)
plt.plot(np.arange(len(predict_fi_ni)),predict_fi_ni,'y-',alpha=0.4)
plt.plot(np.arange(len(predict_sm)),predict_sm,'k--',label='Prob. of dice 1, smoothing',alpha=0.6)
plt.plot(np.arange(len(predict_sm_ni)),predict_sm_ni,'c-',alpha=0.4)
plt.plot(np.arange(len(actual)),actual,'r-',label='Actual die',alpha=0.4)
# plt.plot(np.arange(len(predict_vi)),predict_vi,'k-.',label='Most probable path, Viterbi',alpha=0.6)
plt.plot(np.arange(len(predict_fi)),0.5*np.ones(len(predict_fi)),'k-',alpha=0.5)
plt.legend(loc='upper left')
plt.ylim(-0.1,1.7)
plt.show()

#%%

from HMM_generator import generate
series = generate(100).result('float')
pi,A,B = learn(series, kill_switch=1000, hidden_states=[0,1], observed_states=[[1,2,3,4,5,6],[1,2,3,4,5,6]]).result()

print(pi)
print()
print(A)
print()
print(B)

#%%
print(np.array([1,2])+3)