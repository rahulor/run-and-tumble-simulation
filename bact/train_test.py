import bact.attributes
import conc.profile
import numpy as np
import pandas as pd
import helper
from sklearn.metrics import accuracy_score
line1, line2 = '='*40, '-'*40
def write_dataframe(df, fname):
    pathcsv = 'data/' + fname + '.csv'
    df.to_csv(pathcsv, index=None)
    # pathtxt = 'data/' + fname + '.txt'
    # file = open(pathtxt, "w")
    # text = df.to_string()
    # file.write(text)
    # file.close()
def weights(clf_obj):
    W = list(clf_obj.intercept_) + list(clf_obj.coef_[0]) # first element is bias
    return(W)

def sort_weights():
    from inputs import Lambda_list, D_rot_list
    df = pd.read_csv('data/args_weights.csv', index_col=None)
    dfg = df.groupby(['Lambda', 'D_rot'])
    for i, Lambda in enumerate(Lambda_list):
        weights_mean = []
        weights_sem = []
        for D_rot in D_rot_list:
            dfgg = dfg.get_group((Lambda, D_rot)).loc[:,'bias':] # all rows,  all columns from bias to right
            weights_mean.append([Lambda, D_rot] + list(dfgg.mean(axis = 0).values))
            weights_sem.append( [Lambda, D_rot] + list( dfgg.sem(axis = 0).values))
        #
        df_weights_mean = pd.DataFrame(data=weights_mean, columns=list(df.columns))
        df_weights_sem = pd.DataFrame( data=weights_sem,  columns=list(df.columns))
        write_dataframe(df_weights_mean, 'weights_mean_' + str(i))
        write_dataframe(df_weights_sem, 'weights_sem_' + str(i))
    return

def sort_score():
    from inputs import Lambda_list, D_rot_list
    df = pd.read_csv('data/args_score.csv', index_col=None)
    dfg = df.groupby(['Lambda', 'D_rot'])
    for i, Lambda in enumerate(Lambda_list):
        score_mean = []
        score_sem = []
        for D_rot in D_rot_list:
            dfgg = dfg.get_group((Lambda, D_rot))['score']
            score_mean.append([Lambda, D_rot] + [dfgg.mean(axis = 0)])
            score_sem.append( [Lambda, D_rot] + [ dfgg.sem(axis = 0)])
        #
        df_score_mean = pd.DataFrame(data=score_mean, columns=['Lambda', 'D_rot', 'score'])
        df_score_sem = pd.DataFrame( data=score_sem,  columns=['Lambda', 'D_rot', 'score'])
        write_dataframe(df_score_mean, 'score_mean_' + str(i))
        write_dataframe(df_score_sem, 'score_sem_' + str(i))
    return    
def training():
    print('training...')
    from inputs import D_rot_list, Lambda_list
    from inputs import n_realization_traj, n_realization_signal, n_train_sample, clf, T_max, memory, grad_label
    env = conc.profile.DimensionTwo()
    dt = T_max/memory
    k1 = int(grad_label[0]) # grad_label = [k1, k2], where k is the delay. k1<k2
    k2 = int(grad_label[1])
    column_names = ['Lambda', 'D_rot'] + ['bias'] + [f'w{i}' for i in range(memory)]
    args_weights = []
    for D_rot in D_rot_list:
        Dtext = f'D_rot = {D_rot:<10}'
        for i in range(n_realization_traj):
            print(Dtext, 'traj-set :', i+1,'/',n_realization_traj, line1)
            seed_traj = i
            conc_list = bact.attributes.dataset(seed_traj, env, n_train_sample, D_rot)
            ytar = (conc_list[:, k1] - conc_list[:, k2]) > 0 # labeling
            for Lambda in Lambda_list:
                print(f'Lambda = {Lambda:.2e}')
                lambda_conc_dt = Lambda*conc_list*dt
                for j in range(n_realization_signal):
                    print(line2, 'signal :', j+1,'/', n_realization_signal)
                    seed_signal = j
                    Xtrain = bact.attributes.normalized_binding(seed_signal, lambda_conc_dt)
                    clf.fit(Xtrain, ytar)
                    row = [Lambda, D_rot] + weights(clf) # weights are combined to args
                    args_weights.append(row) # collect all args-weights combinations
    df_args_weights = pd.DataFrame(data=args_weights, columns=column_names)
    df_args_weights.to_csv('data/args_weights.csv', index=None)
    helper.pickle_to('data/clf', clf) # just soring as object. This clf will be updated during testing.
    sort_weights()
    return

def testing():
    print('testing...')
    clf = helper.unpickle_from('data/clf')
    from inputs import n_realization_traj, n_realization_signal, n_test_sample, Lambda_list, D_rot_list
    from inputs import T_max, memory, grad_label
    env = conc.profile.DimensionTwo()
    dt = T_max/memory
    k1 = int(grad_label[0]) # grad_label = [k1, k2], where k is the delay. k1<k2
    k2 = int(grad_label[1])
    # collect all weights in a single dataframe and group them by Drot in order to run faster
    file_begin = 'data/weights_mean_'
    file_ending = [str(j) for j in range(len(Lambda_list))]
    df_list = [] 
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df = pd.read_csv(f, index_col=None)
        df_list.append(df)
    df_merged = pd.concat(df_list)
    dfg = df_merged.groupby(['D_rot'])
    #
    args_score = []
    for D_rot in D_rot_list:
        dfgDrot = dfg.get_group((D_rot))
        Dtext = f'D_rot = {D_rot:<10}'
        for i in range(n_realization_traj):
            print(Dtext, 'traj-set :', i+1,'/',n_realization_traj, line1)
            seed_traj = i+33333 # different from one used in training
            conc_list = bact.attributes.dataset(seed_traj, env, n_test_sample, D_rot)
            ytrue = (conc_list[:, k1] - conc_list[:, k2]) > 0 # labeling
            for index, row in dfgDrot.iterrows():
                Lambda = row['Lambda']
                b, w = row['bias'], row.loc['w0':].values
                clf.intercept_ = b # bias
                clf.coef_[0] = w # coef
                print(f'Lambda = {Lambda:.2e}')
                lambda_conc_dt = Lambda*conc_list*dt
                for j in range(n_realization_signal):
                    print(line2, 'signal :', j+1,'/', n_realization_signal)
                    seed_signal = j+99999
                    Xtest = bact.attributes.normalized_binding(seed_signal, lambda_conc_dt)
                    ypred = clf.predict(Xtest)
                    score = accuracy_score(ytrue, ypred, normalize=True)
                    args_score.append([Lambda, D_rot, score])
    column_names = ['Lambda', 'D_rot', 'score']
    df_args_score = pd.DataFrame(data=args_score, columns=column_names)
    df_args_score.to_csv('data/args_score.csv', index=None)
    sort_score()
    return

def sample_dataset_for_plots(n_sample, D_rot, Lambda):
    from inputs import T_max, memory, grad_label
    args = pd.Series([D_rot, Lambda], index=['D_rot', 'Lambda'])
    args.to_csv('data/sample_args.csv')
    print(f'sample dataset for \n{args}')
    k1 = int(grad_label[0]) # grad_label = [k1, k2], where k is the delay. k1<k2
    k2 = int(grad_label[1])
    env = conc.profile.DimensionTwo()
    dt = T_max/memory
    seed = 0
    conc_list, df_initial, traj_list = bact.attributes.sample_dataset(seed, env, n_sample, D_rot, Lambda)
    ytar = (conc_list[:, k1] - conc_list[:, k2]) > 0 # labeling
    lambda_conc_dt = Lambda*conc_list*dt
    Xtrain = bact.attributes.normalized_binding(seed, lambda_conc_dt)
    df_signal = pd.DataFrame(Xtrain)
    df_initial.to_csv('data/initial_condition.csv', index=None)
    df_signal.to_csv('data/sample_signal.csv', index=None)
    helper.pickle_to('data/sample_trajectory', traj_list)
    helper.pickle_to('data/sample_signal_label', ytar)
if __name__ == "__main__":
    pass
    