import os,sys,time,pickle

os.chdir('..')
# print(os.getcwd())
sys.path.append(os.getcwd())

from evaluator import evaluate
from bandits import *
from matplotlib import pyplot as plt
import numpy as np

#文件
files = (f"../../../data/R6/ydata-fp-td-clicks-v1_0.20090501")

dataset.get_yahoo_events(files)

_, deploy_ctr = evaluate(Egreedy(1))
rnd_ctr = deploy_ctr[-1]

def plot_results(model,tests,learning_rate=0.9):
    start_time = time.time()
    learn_ctrs = []
    deploy_ctrs = []

    for test in tests:
        # is_return 为true,是希望返回每次打印的结果
        learn, deploy, outcome = evaluate(test,learn_ratio=learning_rate,is_return=True)

        with open('../output/log.txt','a') as f:
            f.write(outcome)

        learn = learn[1000:]
        if hasattr(test, 'e'):
            plt.plot(learn, label="ε={}".format(test.e),linewidth=0.5)
        else:
            plt.plot(learn, label="α={}".format(test.alpha),linewidth=0.5)
        
        learn_ctrs.append(learn[-1]/rnd_ctr)   
        deploy_ctrs.append(deploy[-1]/rnd_ctr) 

    plt.title("Learning bucket")
    plt.xlabel("times")
    plt.ylabel("Ctrs")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    plt.savefig(f'../output/learning_bucket_{model}.png',bbox_inches='tight')  # Save plot as image
    plt.close()  # Close the plot to avoid displaying in Jupyter

    
    if hasattr(test, 'e'):
        param_values = [x.e for x in tests]
        plt.xlabel("ε")
    else:
        param_values = [x.alpha for x in tests]
        plt.xlabel("α")
    plt.plot(param_values, learn_ctrs,marker='o')
    plt.title('Learning bucket')
    plt.ylabel("CTR lift")
    # plt.show()
    plt.savefig(f'../output/learning_ctr_lift_{model}.png',bbox_inches='tight')  # Save plot as image
    plt.close()  # Close the plot to avoid displaying in Jupyter

    if learning_rate!=1:
        if hasattr(test, 'e'):
            plt.xlabel("ε")
        else:
            plt.xlabel("α")
        plt.ylabel("CTR lift")
        plt.plot(param_values, deploy_ctrs,marker='o')
        plt.title('Deployment bucket')
        # plt.show()
        plt.savefig(f'../output/deployment_ctr_lift{model}.png',bbox_inches='tight')  # Save plot as image
        plt.close()  # Close the plot to avoid displaying in Jupyter


    best_idx = np.argmax(deploy_ctrs)
    with open(f'../output/instance_{model}.pkl', 'wb') as f:
        pickle.dump(tests[best_idx], f)
    # print('Best parameter:',tests[best_idx].algorithm)
    end_time = time.time()
    elapsed_time = end_time-start_time
    with open('../output/log.txt','a') as f:
        f.write('Best parameter:'+tests[best_idx].algorithm+f", Elapsed time: {elapsed_time:.4f} seconds\n")


if __name__=='__main__':
    alpha_values = np.arange(0.1,1.4,0.2)
    epsilon_values = np.arange(0.1,1,0.1)
    tests0 = [Egreedy(e) for e in epsilon_values]
    plot_results('Egreedy',tests0)
    tests1 = [Ucb1(a) for a in alpha_values]
    plot_results('UCB1',tests1)
    tests2 = [LinUCB(a, context="both") for a in alpha_values]
    plot_results('LinUCB_hybrid',tests2)
    tests3 = [LinUCB(a, context="user") for a in alpha_values]
    plot_results('LinUCB_disjoint',tests3)
