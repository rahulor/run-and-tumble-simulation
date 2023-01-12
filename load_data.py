import main
import helper
import clean
import pandas as pd
# for figure 1 select lambda and D_rot with below indices
lambda_index = 2
D_rot_index  = 4
#

def load_data_for_figure1(): # run only once. Run again if lambda_index, D_rot_index (defined globally) changes 
    from inputs import Lambda_list, D_rot_list
    # create training trajectory 
    D_rot, Lambda = D_rot_list[D_rot_index], Lambda_list[lambda_index]
    n_sample = 1000
    main.sample_dataset_for_plots(n_sample, D_rot, Lambda)
    print('generated training trajectory')
    # decompress simulated trajectory from pickle and write it as csv file
    dir_name = 'lambda_' + str(lambda_index) 
    path = dir_name + '/D' + str(D_rot_index)
    path_flag   = path + '_' + 'bsim_flag.csv'
    print('decompressing pickle | wait ...')
    df_list = helper.decompressed_pickle_from(path)
    print('done!')
    df_flag = pd.read_csv(path_flag)
    # succesfull agents first
    dir_traj = 'data/traj_succ'
    clean.directory(dir_traj)
    n_samples = 50
    mask = df_flag['flag_inside'] == True
    id_succ = df_flag[mask]['id'].values
    N = min(n_samples, len(id_succ))
    for i in id_succ[:N]:
        df = df_list[i]
        ending = '/agent_'+str(i)+'.csv'
        df.to_csv(dir_traj+ending, index=None)
    # then failed agents
    dir_traj = 'data/traj_fail'
    clean.directory(dir_traj)
    n_samples = 50
    mask = df_flag['flag_outside'] == True
    id_fail = df_flag[mask]['id'].values
    N = min(n_samples, len(id_fail))
    for i in id_fail[:N]:
        df = df_list[i]
        ending = '/agent_'+str(i)+'.csv'
        df.to_csv(dir_traj+ending, index=None)
    #

if __name__ == '__main__':
    load_data_for_figure1()
    