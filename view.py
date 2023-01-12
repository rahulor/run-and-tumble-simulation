import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
})
fig_ext = '.eps'

def write_dataframe(df, fname):
    pathcsv = fname + '.csv'
    pathtxt = fname + '.txt'
    df.to_csv(pathcsv)
    file = open(pathtxt, "w")
    text = df.to_string()
    #file.write("RESULT\n\n")
    file.write(text)
    file.close()
    
def time_to_target(dir_name):
    from inputs import t_sim, D_rot_list
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(4,2.5))
    for fileno, D_rot in enumerate(D_rot_list):
        path = dir_name + '/D' + str(fileno)
        Dtext = str(D_rot)
        df_targ = pd.read_csv(path + '_time_to_target.csv')
        time_to_targ = df_targ['time_to_target']
        count_targ, bins = np.histogram(time_to_targ, bins=100, range=(0, t_sim+5))
        ax.hist(bins[:-1], bins, weights=count_targ, histtype='step', color=cmap(fileno), label=Dtext)
    ax.set_xlabel(r'first passage time [s]')
    ax.set_ylabel(r'count', rotation=90, size=12, labelpad=10)
    ax.set_xlim([0, t_sim+5])
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', title=r'$D_{rot}$', fontsize=8)
    # plt.savefig('fig/time_to_target'+ fig_ext, bbox_inches='tight', dpi=300)
    plt.show()

def ci_bins(dir_name):
    from inputs import food_surface, horizon, D_rot_list
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(4,2.5))
    for fileno, D_rot in enumerate(D_rot_list):
        path = dir_name + '/D' + str(fileno)
        Dtext = str(D_rot)
        df = pd.read_csv(path + '_ci_bins.csv')
        ci = df['ci'].values[:-1]
        bins = df['bin_edge'].values
        # CI = np.nan_to_num(ci) # for empty bins, nan may be converted to zero
        CI = ci.copy()
        ax.hist(bins[:-1], bins, weights=CI, histtype='step', 
                linewidth=1.0, color=cmap(fileno), label=Dtext)
        ax.set_ylabel(r'$\mathrm{CI}$', rotation=90, labelpad=-6)
        ax.set_ylim(-1,1)
        ax.minorticks_on()
        #ax[i].grid(which='major', color='white', linewidth=0.1, linestyle = '-')
        ax.grid(axis='both', which='major', color='white', linewidth=0.1, linestyle = '-')
        ax.axvspan(0, food_surface, color='magenta', zorder=2)
        ax.axhline(y=0, color='white', linestyle='--', linewidth=0.1)
    ax.legend(loc=(0.4, 0.01), title=r'$D_\mathrm{rot}$', fontsize=7.5, ncol=1, framealpha=1.0)    
    ax.set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=1)
    ax.set_xlim([0, horizon+0.5])    
    #plt.yscale('log')
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()   
    
def chemotatic_index():
    from inputs import D_rot_list, Lambda
    CI_list = []
    for fileno, D_rot in enumerate(D_rot_list):
        filepath = 'data/' + 'D' + str(fileno) + '_'
        df_CI = pd.read_csv(filepath + 'CI.csv')
        CI = df_CI['CI'].values
        CI_mean = np.mean(CI)
        CI_sem = scipy.stats.sem(CI)
        CI_list.append([D_rot, CI_mean, CI_sem])
    df = pd.DataFrame(data=CI_list, columns=['D_rot', 'mean', 'sem'])
    print(df)
    write_dataframe(df, 'data/CIvsDrot')
    #
    color_gray = '#c6bdba'
    Ltext = r'$\lambda = \, $' + str(Lambda)
    fig, ax = plt.subplots(1, figsize=(3.5,2.5))
    ax.fill_between(df['D_rot'], df['mean'] - df['sem']/2 , df['mean'] + df['sem']/2, color=color_gray)
    ax.plot(df['D_rot'], df['mean'], color='tab:blue', label = Ltext, lw=1.0)
    ax.scatter(df['D_rot'], df['mean'], color='tab:blue', s=1.5)
    ax.set_ylabel(r'chemotatic index')
    ax.set_xlabel(r'$D_{rot}$ [rad$^2/s$]', size=11,  rotation=0, labelpad=5)
    ax.legend(loc='lower left', fontsize=11)
    #ax.set_yscale('log')
    ax.set_ylim([0, None])
    #plt.savefig('fig/CI'+ fig_ext, bbox_inches='tight', dpi=300)
    plt.show()
        
if __name__ == '__main__':
    pass
    dir_name = 'lambda_0' # 'lambda_1' # 'lambda_1'
    #dir_name = 'ideal'
    time_to_target(dir_name)
    ci_bins(dir_name)
    time_to_target(dir_name)