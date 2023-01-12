import pandas as pd
import numpy as np
import helper
def write_dataframe(df, fname):
    pathcsv =  fname + '.csv'
    pathtxt =  fname + '.txt'
    df.to_csv(pathcsv)
    file = open(pathtxt, "w")
    text = df.to_string()
    #file.write("RESULT\n\n")
    file.write(text)
    file.close()

def bact_summary(df, path):
    print('bact_summary()')
    from inputs import n_bacteria
    inside = df['flag_inside'].values
    outside = df['flag_outside'].values
    df_stat = pd.Series(dtype = 'object')
    df_stat['total number of agents'] = n_bacteria
    df_stat['agents reached the target'] = np.sum(inside)
    df_stat['agents moved out'] = np.sum(outside)
    df_stat['agents still roaming around'] = n_bacteria - df_stat['agents reached the target'] - df_stat['agents moved out']
    write_dataframe(df_stat, path + '_bact_stat')
    print(df_stat.to_string())
    return

def time_to_target(df_list, df_flag, path):
    mask = df_flag['flag_inside'] == True
    id_succ = df_flag[mask]['id'].values
    time_to_food = []
    for i in id_succ:
        df = df_list[i]
        time_final = df["time"].iloc[-1]
        time_to_food.append(time_final)
    df_time_to_target = pd.DataFrame(id_succ, columns=['id_succ'])
    df_time_to_target['time_to_target'] = time_to_food
    df_time_to_target.to_csv(path + '_time_to_target.csv', index=False)
    
def time_to_horizon(df_list, df_flag, path):
    mask = df_flag['flag_outside'] == True
    id_horizon = df_flag[mask]['id'].values
    time_to_horz = []
    for i in id_horizon:
        df = df_list[i]
        time_final = df["time"].iloc[-1]
        time_to_horz.append(time_final)
    df_time_to_horizon = pd.DataFrame(id_horizon, columns=['id_horizon'])
    df_time_to_horizon['time_to_horizon'] = time_to_horz
    df_time_to_horizon.to_csv(path + '_time_to_horizon.csv', index=False)
    return

def ci_bins(df_list, path):
    print('ci_bins()')
    from inputs import T_max, memory, food_surface, horizon, bact_speed
    dt = T_max/memory
    r1, r2 = food_surface-0.5, horizon+0.5
    n_bins = 100
    dr = (r2 - r1) / n_bins
    bin_edge = np.linspace(start=r1, stop=r2, num=n_bins+1, endpoint=True)
    dict_empty = {i:None for i in range(n_bins)}
    dict_list = [dict_empty] # this will grow
    origin = np.array([0, 0])
    # all bin id should appear in the list if by chance someone run it with low number of agents
    # just assing None values to ci_list. 
    bin_id_list = [i for i in range(n_bins)]
    ci_list = [None for i in range(n_bins)]
    # 
    pcounter = 0
    for n, df in enumerate(df_list):
        perc_complete = (100*n/len(df_list))
        if perc_complete > 10*pcounter:
            print(f'{int(perc_complete)} % completed')
            pcounter+= 1
        pos = df[['x', 'y']].values
        
        for i in range(1, len(pos)):
            R = pos[i] - origin
            r = np.linalg.norm(R)
            e_r =  -R/r # towards center
            unit_tangent = (pos[i] - pos[i-1])/(bact_speed*dt)
            tangent_dot_e_r = np.dot(unit_tangent, e_r)
            bin_id = int((r-r1)/dr)
            bin_id_list.append(bin_id)
            ci_list.append(tangent_dot_e_r)
    df_ci_all = pd.DataFrame(bin_id_list, columns=['bin_id'])
    df_ci_all['ci'] = ci_list
    print(f'{100} % completed')
    #
    ser_ci = pd.Series([df_ci_all['ci'].mean(), df_ci_all['ci'].sem()], index=['mean', 'sem']) 
    ser_ci.to_csv(path + '_ci' + '.csv', header=None)
    # residence time
    counts = df_ci_all.groupby(['bin_id'])['ci'].count().values
    res_time = list(counts*dt) + [None] #converted to time
    #
    df_time = pd.DataFrame(bin_edge, columns=['bin_edge'])
    df_time['res_time'] = res_time
    df_time.to_csv(path + '_res_time' + '.csv')
    # ci mean for each bin
    dict_ci_bin = df_ci_all.groupby(['bin_id']).mean()['ci'].to_dict()
    dict_list.append(dict_ci_bin)
    df_ci_bin = pd.DataFrame.from_dict(dict_list, orient='columns')
    # 
    df_ci_bin = df_ci_bin.mean().values # 
    ci_for_bin = list(df_ci_bin) + [None] # bin edge in one more than number of bins
    df_ci_bins = pd.DataFrame(bin_edge, columns=['bin_edge'])
    df_ci_bins['ci'] = ci_for_bin
    df_ci_bins.to_csv(path + '_ci_bins.csv', index=False)
    return

def runtime(df_list, path):
    print('calculate_runtime()'.ljust(40, '-'))
    runtime_all = []
    for fi, df in enumerate(df_list, start=0):
        mask = df['tumble'] == True
        time_tumble = df[mask]['time'].values
        time_tumble = np.append(time_tumble, df['time'].values[-1]) # append end time
        time_tumble = np.sort(np.array(list(set(time_tumble)))) # set: to avoid repetition (of the end time) if exist
        runtime = np.diff(time_tumble)
        runtime_all.extend(list(runtime))
    df_runtime = pd.DataFrame(runtime_all, columns=['runtime'])
    df_runtime.to_csv( path + '_runtime.csv', index=False)
    #
    df_stat = pd.Series(dtype = 'object')
    df_stat['mean'] = np.mean(runtime_all)
    df_stat['std']  = np.std(runtime_all)
    df_stat['median'] = np.median(runtime_all)
    write_dataframe(df_stat, path + '_runtime_stat')
    
def false_positive(df_list, path):
    print('false_positive()'.ljust(40, '-'))
    origin = np.array([0, 0])
    radial_dist_all = []
    for df in df_list:
        df = df.iloc[1:] # drop the first row | run starts with a tumble
        mask = (df['label_pred'] == False) & (df['label_true'] == True) # unwanted tumble!
        pos = df[mask][['x', 'y']].values
        for R in pos:
            dist_vector = R - origin
            d = np.linalg.norm(dist_vector) # distance to the [R] from [food-center].
            radial_dist_all.append(d)
    n_fp = len(radial_dist_all)
    from inputs import food_surface, horizon
    r1, r2 = food_surface-0.5, horizon+0.5
    count, bins = np.histogram(radial_dist_all, bins=101, range=(r1, r2))
    count = list(count) + [None]
    df_radial_dist = pd.DataFrame(count, columns=['count'])
    df_radial_dist['bins'] = bins
    df_radial_dist.to_csv(path + '_dist_FP.csv', index=False)
    return n_fp

def false_negative(df_list, path):
    print('false_negative()'.ljust(40, '-'))
    origin = np.array([0, 0])
    radial_dist_all = []
    for df in df_list:
        df = df.iloc[1:] # drop the first row | run starts with a tumble
        mask = (df['label_pred'] == True) & (df['label_true'] == False) # missed tumble!
        pos = df[mask][['x', 'y']].values
        for R in pos:
            dist_vector = R - origin
            d = np.linalg.norm(dist_vector) # distance to the [R] from [food-center].
            radial_dist_all.append(d)
    n_fn = len(radial_dist_all)
    from inputs import food_surface, horizon
    r1, r2 = food_surface-0.5, horizon+0.5
    count, bins = np.histogram(radial_dist_all, bins=101, range=(r1, r2))
    count = list(count) + [None]
    df_radial_dist = pd.DataFrame(count, columns=['count'])
    df_radial_dist['bins'] = bins
    df_radial_dist.to_csv(path +'_dist_FN.csv', index=False)
    return n_fn

def true_positive(df_list, path):
    print('true_positive'.ljust(40, '-'))
    origin = np.array([0, 0])
    radial_dist_all = []
    for df in df_list:
        df = df.iloc[1:] # drop the first row | run starts with a tumble
        mask = (df['label_pred'] == False) & (df['label_true'] == False) # tumble - right decision
        pos = df[mask][['x', 'y']].values
        for R in pos:
            dist_vector = R - origin
            d = np.linalg.norm(dist_vector) # distance to the [R] from [food-center].
            radial_dist_all.append(d)
    n_tp = len(radial_dist_all)
    from inputs import food_surface, horizon
    r1, r2 = food_surface-0.5, horizon+0.5
    count, bins = np.histogram(radial_dist_all, bins=101, range=(r1, r2))
    count = list(count) + [None]
    df_radial_dist = pd.DataFrame(count, columns=['count'])
    df_radial_dist['bins'] = bins
    df_radial_dist.to_csv(path+ '_dist_TP.csv', index=False)
    return n_tp

def true_negative(df_list, path):
    print('true_negative'.ljust(40, '-'))
    origin = np.array([0, 0])
    radial_dist_all = []
    for df in df_list:
        df = df.iloc[1:] # drop the first row | run starts with a tumble
        mask = (df['label_pred'] == True) & (df['label_true'] == True) # no tumble - right decision
        pos = df[mask][['x', 'y']].values
        for R in pos:
            dist_vector = R - origin
            d = np.linalg.norm(dist_vector) # distance to the [R] from [food-center].
            radial_dist_all.append(d)
    n_tn = len(radial_dist_all)
    from inputs import food_surface, horizon
    r1, r2 = food_surface-0.5, horizon+0.5
    count, bins = np.histogram(radial_dist_all, bins=101, range=(r1, r2))
    count = list(count) + [None]
    df_radial_dist = pd.DataFrame(count, columns=['count'])
    df_radial_dist['bins'] = bins
    df_radial_dist.to_csv(path + '_dist_TN.csv', index=False)
    return n_tn

def run(args):
    dir_name, fileno = args['dir_name'], args['fileno']
    path = dir_name + '/D' + str(fileno)
    path_flag   = dir_name + '/D' + str(fileno) + '_' + 'bsim_flag.csv'
    df_list = helper.decompressed_pickle_from(path)
    df_flag = pd.read_csv(path_flag)
    #
    bact_summary(df_flag, path)
    time_to_target(df_list, df_flag, path)
    ci_bins(df_list, path)
    # time_to_horizon(df_list, df_flag, path)
    ci_bins(df_list, path)
    runtime(df_list, path)
    # #
    n_fp = false_positive(df_list, path)
    n_fn = false_negative(df_list, path)
    n_tp = true_positive(df_list, path)
    n_tn = true_negative(df_list, path)
    df_stat = pd.Series([n_fp, n_fn, n_tp, n_tn], index = ["FP", "FN", "TP", "TN"])
    n_f = n_fp + n_fn
    df_stat['F = FP + FN'] = n_f
    n_t = n_tp + n_tn
    df_stat['T = TP + TN'] = n_t
    df_stat['F + T'] = n_f + n_t
    df_stat['score = T/(F+T) [%]'] = np.round( 100*n_t /(n_f + n_t), 1 )
    write_dataframe(df_stat, path + '_fpfn_stat')
    
if __name__ == '__main__':
    pass
    
