import pandas as pd
import bact.train_test
import bact.ideal
import bact.machine_learning
import clean
import helper

def sample_dataset_for_plots(n_sample, D_rot, Lambda):
    bact.train_test.sample_dataset_for_plots(n_sample, D_rot, Lambda)
def weights_and_score():
    bact.train_test.training()
    bact.train_test.testing()
def ideal_agent():
    from inputs import D_rot_list, n_bacteria
    dir_name = 'ideal'
    clean.directory(dir_name)
    clean.directory(dir_name+'/traj')
    for fileno, D_rot in enumerate(D_rot_list):
        print(f'({fileno+1}/{len(D_rot_list)})   D_rot = {D_rot:<10}')
        args = pd.Series([dir_name, fileno, D_rot], index=['dir_name', 'fileno', 'D_rot'])
        bact.ideal.run(args)
        # collect all traj files and store as a compressed pickle
        df_list = []
        for k in range(n_bacteria):
            f = dir_name + '/traj/id_' + str(k) + '.csv'
            df = pd.read_csv(f)
            df_list.append(df)
        path = dir_name + '/D' + str(fileno)
        helper.compressed_pickle_to(path, df_list)
        clean.directory(dir_name+'/traj')

def machine_learning_agent():
    clf = helper.unpickle_from('data/clf')
    # go through the weights and use it to run ML agent
    from inputs import Lambda_list
    file_begin = 'data/weights_mean_'
    file_ending = [str(j) + '.csv' for j in range(len(Lambda_list))]
    for j, fend in enumerate(file_ending):
        fweights = file_begin + fend
        df_weights = pd.read_csv(fweights, index_col=None)
        # use a directory to store everything that comes
        dir_name = 'lambda_' + str(j) 
        clean.directory(dir_name)
        clean.directory(dir_name+'/traj')
        pd.Series(Lambda_list[j], index=['lambda']).to_csv(dir_name+'/for.csv', header=False) # just store lambda
        #
        for fileno, row in df_weights.iterrows():
            Lambda = row['Lambda']
            D_rot = row['D_rot']
            b, w = row['bias'], row.loc['w0':].values
            clf.intercept_ = b # bias
            clf.coef_[0] = w # coef
            print(f'({fileno+1}/{len(df_weights)})   D_rot = {D_rot:<10}')
            args = pd.Series([dir_name, fileno, D_rot, Lambda, clf], 
                      index=['dir_name', 'fileno', 'D_rot', 'Lambda', 'clf'])
            bact.machine_learning.run(args)
            # collect all traj files and store as a compressed pickle
            from inputs import n_bacteria
            df_list = []
            for k in range(n_bacteria):
                f = dir_name + '/traj/id_' + str(k) + '.csv'
                df = pd.read_csv(f)
                df_list.append(df)
            path = dir_name + '/D' + str(fileno)
            helper.compressed_pickle_to(path, df_list)
            clean.directory(dir_name+'/traj')

def main():
    weights_and_score()
    ideal_agent()
    machine_learning_agent()
if __name__ == '__main__':
    weights_and_score()

    
    