"""
for ideal agent
"""
import pandas as pd
import bact.ideal
import clean
import helper
import computation

dir_name = 'ideal'


def ideal_agent():
    from inputs import D_rot_list, n_bacteria
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

def calculations():
    from inputs import D_rot_list
    for fileno, D_rot in enumerate(D_rot_list):
        print(f'({fileno+1}/{len(D_rot_list)})   D_rot = {D_rot:<10}')
        args = pd.Series([dir_name, fileno, D_rot], index=['dir_name', 'fileno', 'D_rot'])
        computation.run(args)

if __name__ == '__main__':
    ideal_agent()
    calculations()
