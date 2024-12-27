import os,sys,time,pickle

os.chdir('..')
# print(os.getcwd())
sys.path.append(os.getcwd())

from bandits import *
from evaluator import evaluate
import dataset
import pickle


files = (f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090501",f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090503",f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090504")
# ,f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090505",f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090506",f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090507",f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090508",f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090509"

dataset.get_yahoo_events(files)

learn_ratio = 0.9

#随机策略
_, deploy_ctr = evaluate(Egreedy(1),learn_ratio=learn_ratio,is_training=False)
deploy_rnd_ctr = deploy_ctr[-1]


tests = [Egreedy(0.1),ThompsonSampling(),Ucb1(0.1)]

linucb_disjoint_file = '../output/instance_LinUCB_disjoint.pkl'
linucb_hybrid_file = '../output/instance_LinUCB_hybrid.pkl'
linucb_files = [linucb_disjoint_file, linucb_hybrid_file]
for file in linucb_files:
    with open(file, 'rb') as f:
        tests.append(pickle.load(f))

models_name = ['Egreedy(0.1)','TS','Ucb1(0.1)','linucb_disjoint','linucb_hybrid']

for i,test in enumerate(tests):
    l_ctr,d_ctr = evaluate(test,learn_ratio=learn_ratio,is_training=False)
    l_ctr,d_ctr = l_ctr[-1]/deploy_rnd_ctr,d_ctr[-1]/deploy_rnd_ctr
    with open('../output/log.txt','a') as f:
        f.write(f"\n{str(models_name[i])}'s learning_ctr is {l_ctr}, deploying_ctr is {d_ctr}")