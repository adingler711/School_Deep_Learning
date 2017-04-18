import seaborn as sns
from multiprocessing import Pool, cpu_count
from simple_5x5_letter_classification_tutorial_code_v6_2017_02_02 import *


alpha_list = range(1,11)
eta_list = list(np.array(range(1,11)) / 10.0)
numHiddenNodes_list = range(2,11) 

def run_nn(args):
    
    [alpha, eta] = args
    # Parameter definitions
    numOutputNodes = 5
    numTrainingDataSets = 4
    numHiddenNodes = 6

    epsilon = 0.001
    numInputNodes = 25
    maxNumIterations = 5000
    
    results_df = pd.DataFrame(columns = ['letter', 'H0', 'H1', 'H2', 'H3', 'H4', 'H5', 
                                         'SSE', 'iteration', 'alpha', 'eta'])

    tune_nn = tune_parameters(alpha = alpha, 
                              numTrainingDataSets = numTrainingDataSets, 
                              numInputNodes=numInputNodes, 
                              numOutputNodes=numOutputNodes,
                              numHiddenNodes=numHiddenNodes, 
                              eta = eta, 
                              epsilon = epsilon, 
                              maxNumIterations = maxNumIterations)

    tune_nn['alpha'] = alpha
    tune_nn['eta'] = eta
    tune_nn['avg_SSE'] = tune_nn['SSE'].mean()

    return tune_nn

combo_list = []
for alp in alpha_list:
    for etas in eta_list:
        #for hidNode in numHiddenNodes_list: 
        combo_list.append([alp, etas])
        
use_cpu = cpu_count()
pool = Pool(use_cpu)

arg_iteration_results_df_pool = pool.map(run_nn, combo_list)
arg_iteration_results_df = pd.concat(arg_iteration_results_df_pool)
arg_iteration_results_df = arg_iteration_results_df.sort_values('avg_SSE', ascending = True)
arg_iteration_results_df.to_csv('5_letters_first_nn.csv')

arg_iteration_results_df = arg_iteration_results_df.sort_values('avg_SSE', ascending = True)
arg_iteration_results_df_trimmed =  arg_iteration_results_df[['letter','alpha', 'eta', 'SSE', 'iteration']]


alpha = arg_iteration_results_df_trimmed[arg_iteration_results_df_trimmed['alpha'] == 4]

ax = sns.pointplot(x="eta", y="SSE", hue="letter",
                    data=alpha, palette="Set2")


eta_df = arg_iteration_results_df_trimmed[arg_iteration_results_df_trimmed['eta'] == 0.6]

ax = sns.pointplot(x="alpha", y="SSE", hue="letter",
                    data=eta_df, palette="Set2")

