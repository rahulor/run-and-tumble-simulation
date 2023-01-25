import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import conc.profile
import helper
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    'axes.labelsize': 10,
    'xtick.labelsize':8,
    'ytick.labelsize':8
})
from scipy.optimize import curve_fit
fig_ext = '.eps'
mm = 1/25.4 # inches

def conc_profile():
    from inputs import horizon, food_surface, bact_speed, conc_horizon
    pos_x = np.linspace(0.0, horizon+bact_speed/10, num=100, endpoint=True)
    pos_xandy = lambda arr: np.vstack((arr, np.zeros(len(arr)))).T # create two columns x=arr, y=0
    env = conc.profile.DimensionTwo()
    conc_and_flag = map(env.concentration_at, pos_xandy(pos_x))
    concentration = [item[0] for item in conc_and_flag]
    fig, ax = plt.subplots(figsize=(80*mm, 60*mm))
    ax.set_title(env.grad_fun)
    ax.plot(pos_x, concentration, color='tab:blue')
    ax.set_xlabel(r'radial distance $\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]',  rotation=0, labelpad=5)
    ax.set_ylabel(r'concentration $ \, [\mathrm{\mu M}]$')
    ax.axvspan(pos_x.min(), food_surface, color='magenta')
    ax.axvspan(horizon, pos_x.max(), color='gray')
    ax.set_xlim([0,None])
    ax.set_ylim([conc_horizon,None])
    if env.grad_fun=='exp':
        plt.yscale('log')
    print(env.conc_txt)
    ax.grid(True, linestyle = ':', linewidth = 0.5)
    figpath = 'fig/conc_profile' + fig_ext
    txtpath = 'data/conc_profile.txt'
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    file = open(txtpath, "w")
    file.write(env.conc_txt)
    file.close()
    plt.show()
def write_dataframe(df, fname):
    pathcsv = 'data/' + fname + '.csv'
    pathtxt = 'data/' + fname + '.txt'
    df.to_csv(pathcsv, index=None)
    file = open(pathtxt, "w")
    text = df.to_string()
    file.write(text)
    file.close()

def initial_condition():
    print('plot initial_condition() '.ljust(40, '-'))
    from inputs import food_surface, horizon
    df = pd.read_csv('data/initial_condition.csv')
    x, y = df['x'].values, df['y'].values
    fig, ax = plt.subplots(figsize=(70*mm, 70*mm))
    ax.set_title("initial position")
    ax.scatter(x, y, marker='o', color='tab:blue', s=0.1)
    food = plt.Circle((0,0), food_surface, color='magenta')
    horz = plt.Circle((0,0), horizon, color='gray', lw=0.5, fill=False)
    ax.add_patch(food)
    ax.add_patch(horz)
    ax.set_xlim([-1.01*horizon, 1.01*horizon])
    ax.set_ylim([-1.01*horizon, 1.01*horizon])
    ax.set_xlabel(r'$x \, [\mu \mathrm{m}]$', rotation=0, labelpad=5)
    ax.set_ylabel(r'$y \, [\mu \mathrm{m}]$', rotation=90, labelpad=5)
    ax.grid(True, linestyle = ':', linewidth = 0.5)
    ax.set_aspect('equal')
    plt.savefig('fig/init_position'+ fig_ext, bbox_inches='tight', dpi=300)
    plt.show()

def trajectory():
    print('plot trajectory() '.ljust(40, '-'))
    n_samples = 300
    traj_all = helper.unpickle_from('data/sample_trajectory')
    ytar = helper.unpickle_from('data/sample_signal_label')
    N = min(n_samples, len(traj_all))
    traj_list = traj_all[:N]
    from inputs import food_surface, horizon
    fig, ax = plt.subplots(figsize=(70*mm, 70*mm))
    ax.set_title(f'trajectory [{N} samples]')
    for i, R in enumerate(traj_list):
        Rx = R[:,0]
        Ry = R[:,1]
        if ytar[i]==True:
            ax.plot(Rx, Ry, color='tab:green', lw=0.5)
        else:
            ax.plot(Rx, Ry, color='tomato', lw=0.5)
    food = plt.Circle((0,0), food_surface, color='magenta')
    horz = plt.Circle((0,0), horizon, color='gray', lw=0.5, fill=False)
    ax.add_patch(food)
    ax.add_patch(horz)
    ax.set_xlim([-1.01*horizon, 1.01*horizon])
    ax.set_ylim([-1.01*horizon, 1.01*horizon])
    ax.set_xlabel(r'$x \, [\mu \mathrm{m}]$', rotation=0, labelpad=5)
    ax.set_ylabel(r'$y \, [\mu \mathrm{m}]$', rotation=90, labelpad=5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle = ':', linewidth = 0.5)
    args = pd.read_csv('data/sample_args.csv', index_col=0, header=None).squeeze("columns")
    print(args)
    Dtext = r'$D_{\mathrm{rot}} =$' + str(args['D_rot']) + r'$\, [\mathrm{rad}^2/\mathrm{s}]$'
    ax.text(0.1, 0.1, Dtext, color='black', size=8, ha='left', va='bottom',
                            transform=ax.transAxes,
                            bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round'))
    plt.savefig('fig/trajectory'+ fig_ext , bbox_inches='tight', dpi=300)
    plt.show()

def signal():
    from inputs import T_max, memory
    dt = T_max/memory
    df_signal = pd.read_csv('data/sample_signal.csv', index_col=None)
    signal_all = df_signal.values
    nfig = 10
    signal = signal_all[0:nfig,:]
    ytar = helper.unpickle_from('data/sample_signal_label')
    time = np.arange(len(signal[0]))*dt
    args = pd.read_csv('data/sample_args.csv', index_col=0, header=None).squeeze("columns")
    Lambda, D_rot = args['Lambda'], args['D_rot']
    Dtext =  f'{D_rot}'
    title = r'$D_{\mathrm{rot}}= $' + Dtext + '$\ \ \ \ \lambda=$' + format_latex(Lambda)
    fig, ax = plt.subplots( len(signal), figsize=(120*mm, 150*mm), sharex=True)
    ax[0].set_title(title, y=0.94)
    for i, u in enumerate(signal):
        if ytar[i]==True:
            ax[i].plot(time, u, lw=0.5, color='tab:green', label=r'+ve')
        else:
            ax[i].plot(time, u, lw=0.5, color='tomato', label=r'-ve')
        ax[i].set_yticks([0,2])
        ax[i].set_ylim([0, None])
    ax[i].set_xlabel(r'time [s]', rotation=0, labelpad=1)
    fig.subplots_adjust(hspace=0.5)
    figpath = 'fig/signal' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    plt.show()

def weights_for_grid():
    from inputs import Lambda_list, D_rot_list, T_max, memory
    dt = T_max/memory
    tau = np.arange(memory)*dt
    nrow, ncol = len(D_rot_list), len(Lambda_list)
    Lambda_text = [r'$\lambda = $' + format_latex(L) for L in np.array(Lambda_list)]
    color_gray = '#c6bdba'
    fig, axs = plt.subplots(nrow, ncol, figsize=(180*mm, 220*mm), sharex=True)
    for j in range(ncol):
        df = pd.read_csv('data/weights_mean_' + str(j) + '.csv', index_col=None)
        dfsem = pd.read_csv('data/weights_sem_' + str(j) + '.csv', index_col=None)
        for i in range(nrow):
            item = df.iloc[i]
            wsem = dfsem.iloc[i]
            b, w = item['bias'], item.loc['w0':].values
            err = wsem.loc['w0':].values
            ax = axs[i,j]
            ax.fill_between(tau, w - err/2, w + err/2, color=color_gray)
            ax.plot(tau, w, color='tab:blue', label = r'$w$', lw=0.5)
            ax.vlines(tau[-1], ymin=0, ymax=b, colors='tab:green', linestyles='solid', label=r'$b$')
            ax.axhline(y=0.0, color="black", linestyle="--", lw=0.1)
    for ax in axs[-1,:]:
        ax.legend(loc='upper center', fontsize=5)
        ax.set_xlabel(r'$\tau$ [s]', rotation=0, labelpad=0)
    for i, ax in enumerate(axs[:,0]):
        Dtext = r'$D_{\mathrm{rot}} = $' + str(D_rot_list[i])
        ax.set_ylabel(r'$w(\tau)$', rotation=90, labelpad=1)
        ax.text(-0.4, 0.5, Dtext, color='black', size=8, ha='center', va='center', rotation = 90,
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
    for j, ax in enumerate(axs[0,:]):
        ax.text(0.5, 1.1, Lambda_text[j], color='black', size=8, ha='center', va='center',
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    fig.align_ylabels(axs[:, 1])
    plt.savefig('fig/weights_grid'+ '.pdf', bbox_inches='tight', dpi=300,  pad_inches=0.1)
    plt.show()

def format_latex(number):
    from decimal import Decimal
    x = Decimal(number)
    prec = 1
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    digit_first = digits[0]
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = x.adjusted()
    if (digit_first == 1) and (digits[1:][0] == 0):
        number_latex = f'{sign}$ 10^{exp}$'
    else:
        number_latex = f'{sign}{digit_first}.{dec}$\\times 10^{exp}$'
    return(number_latex)
def score_vs_Drot():
    from inputs import Lambda_list
    file_begin = 'data/score_mean_'
    selected_lambda_idx = list(np.arange(len(Lambda_list))) #required for the plot.
    selected_Lambda = np.array(Lambda_list)[selected_lambda_idx]
    file_ending = [str(j) for j in selected_lambda_idx]
    #
    Lambda_text = [format_latex(L) for L in selected_Lambda]
    colors = plt.cm.Reds(np.linspace(0.35, 1, len(Lambda_text)))
 
    fig, ax = plt.subplots(1, figsize=(80*mm, 40*mm))
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df = pd.read_csv(f, index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        score = df['score'].values # entire column
        ax.scatter(D_rot, score*100, color=colors[j], s=1)
        ax.plot(D_rot, score*100, label = Ltext, color=colors[j], linewidth=0.5)
    hpos = 0.5*D_rot[-1]
    ax.annotate(r'', xy=(hpos, 95), xytext=(hpos, 75), color='black',
                arrowprops={'arrowstyle': '->', 'lw': 0.3, 'color': 'black'},
                va='top', ha='center', size=10)
    ax.text(hpos*1.1, 85, r'$\lambda$', color='black', size=8, ha='left', va='center')
    ax.set_ylabel(r'score [$\%$]', rotation=90, labelpad=-2)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/\mathrm{s}]$', rotation=0, labelpad=2)
    ax.legend(loc=(0.1,0.1), title=r'$\lambda\, [1/(\mu\mathrm{M}\, \mathrm{s})]$', title_fontsize=8,
              fontsize=7, facecolor='white', framealpha=1)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_ylim([70, 101])
    ax.axhline(y=100, color='cadetblue', linestyle="--", lw=1.0)

    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/score' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()
    
def time_to_target():
    from ideal.inputs import sim_time, D_rot_list, n_bacteria
    D_rot_list = D_rot_list[:-1]
    print(f'D_rot_list  {D_rot_list}')
    colors = plt.cm.Blues(np.linspace(1, 0.5, len(D_rot_list)))
    dirlist = ['lambda10', 'ideal']
    Ltextlist = [r'$\lambda = \, $' + str(10), 'ideal']
    fig, ax = plt.subplots(2, figsize=(54*mm, 58*mm), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    for i, dirname in enumerate(dirlist):
        Ltext = Ltextlist[i]
        ax[i].text(60, 0.0195, Ltext, color='black', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'))
        rate_list = []
        prob_list = []
        for fileno, D_rot in enumerate(D_rot_list):
            filepath = dirname + '/data/' + 'D' + str(fileno) + '_'
            Dtext = str(D_rot)
            df_targ = pd.read_csv(filepath + 'time_to_target.csv')
            time_to_targ = df_targ['time_to_target']
            count_targ, bins = np.histogram(time_to_targ, bins=100, range=(0, sim_time+5))
            binsize = (bins[1]-bins[0])
            count_targ = count_targ/(n_bacteria*binsize)
            ax[i].hist(bins[:-1], bins, weights=count_targ, histtype='step', color=colors[fileno],
                       label=Dtext, linewidth=0.5)
            #stats / percentage
            df_stat = pd.read_csv(filepath + 'bact_stat.csv', header=None, index_col=0).squeeze("columns")
            n_bacteria = df_stat['total number of agents']
            n_targ = df_stat['agents reached the target']
            rate = 100*n_targ/(n_bacteria)
            prob = np.sum(count_targ*binsize)
            rate_list.append(rate)
            prob_list.append(prob)
        ax[i].set_ylabel(r'$p (t)$ [1/s]', rotation=90, labelpad=3)
        ax[i].set_ylim(0,0.024)
        ax[i].set_xlim(0, 205)
        ax[i].minorticks_on()
        # ax[i].grid(which='major', color='gray', linewidth=0.1, linestyle = '-')
        # ax[i].grid(which='minor', color='gray', linewidth=0.1, linestyle = '-')
        ax[i].legend(loc=(0.55,0.25), title=r'$D_\mathrm{rot}$', fontsize=5.3)
        axins = ax[i].inset_axes([0.8, 0.27, 0.35, 0.5])
        axins.set(facecolor ='#cdcdcd')
        axins.minorticks_on()
        axins.grid(axis='x', which='minor', color='white', linewidth=0.1, linestyle = '-')
        axins.grid(axis='x', which='major', color='white', linewidth=0.1, linestyle = '-')
        axins.barh(np.arange(len(D_rot_list))*1.5, prob_list, tick_label='',
                    color=colors, zorder=2)
        axins.invert_yaxis()  # labels read top-to-bottom
        axins.invert_xaxis()
        axins.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                      bottom=True, top=False, left=False, right=False, pad=2)
        axins.yaxis.set_tick_params(which='minor', bottom=False)

        axins.set_xlim([0.6, 1.0])
        ax[i].text(180,0.0205, r'$\int p(t) dt$', color='black', size=6,
                bbox=dict(facecolor='#cdcdcd', edgecolor='None', boxstyle='round'))

    ax[i].set_xlabel(r'first passage time [s]', labelpad=0)
#    plt.savefig('fig/time_to_target'+ fig_ext, bbox_inches='tight', dpi=300)
    #fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('tikz/time_to_target'+ fig_ext, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()

def runtime_grid():
    dir_list = ['lambda_0', 'lambda_1', 'lambda_2', 'ideal']
    from inputs import Lambda_list, D_rot_list, T_max
    
    Lambda_text = [r'$\lambda = $' + format_latex(L) for L in np.array(Lambda_list)]
    Lambda_text.append(r'ideal')
    nrow, ncol = len(D_rot_list), len(Lambda_text)
    color_gray = '#c6bdba'
    fig, axs = plt.subplots(nrow, ncol, figsize=(180*mm, 250*mm), sharex=True)
    for j in range(ncol):
        dir_name = dir_list[j]
        for fileno in range(nrow):
            path = dir_name + '/D' + str(fileno) + '_'
            df = pd.read_csv(path + 'runtime.csv', index_col=None)
            #print(df)
            runtime_all = df['runtime'].values
            runtime_above_Tmax = runtime_all[runtime_all>=T_max]
            ax = axs[fileno, j]
            count, bins, patches = ax.hist(runtime_above_Tmax, bins=100, range=(0,np.max(runtime_above_Tmax)))
            ax.set_xlim([0, 30])
            ax.set_ylim([0, 5000])
            #
            axins = inset_axes(ax, width="60%", height="65%", loc=1)
            time_bins = (bins[:-1] + bins[1:]) / 2
            axins.plot(time_bins, count, linestyle='solid', linewidth=0.5, c='tab:blue')
            axins.set_yscale("log")
            axins.set_xlim([0, 30])
            axins.set_xticks([0, 30])
            axins.minorticks_on()
            axins.grid(axis='y', which='minor', color='white', linewidth=0.1, linestyle = '-')
            axins.grid(axis='y', which='major', color='white', linewidth=0.1, linestyle = '-')
            ax.set_yticks([])
    for ax in axs[-1,:]:
        ax.set_xlabel(r'run time $t$[s]', rotation=0, labelpad=0)
    for fileno, ax in enumerate(axs[:,0]):
        D_rot = D_rot_list[fileno]
        Dtext = r'$D_{\mathrm{rot}} = $' + str(D_rot)
        ax.text(-0.45, 0.5, Dtext, color='black', size=8, ha='center', va='center', zorder = 2, rotation = 90,
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
        ax.set_ylabel(r'count', rotation=90, labelpad=0)
        ax.set_yticks([2000, 5000])
    for j, ax in enumerate(axs[0,:]):
        ax.text(0.5, 1.1, Lambda_text[j], color='black', size=8, ha='center', va='center', zorder = 2,
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    fig.align_ylabels(axs[:, 1])
    plt.savefig('fig/runtime_grid'+ '.pdf', bbox_inches='tight', dpi=300,  pad_inches=0.1)
    plt.show()

def reach_vs_Drot():
    from inputs import D_rot_list, Lambda_list
    dir_list = ['lambda_'+str(j) for j in range(len(Lambda_list))] + ['ideal']
    Lambda_text = [format_latex(L) for L in Lambda_list] + [r'ideal']
    colors = plt.cm.Reds(np.linspace(0.35, 1, len(Lambda_text)))
    
    fig, ax = plt.subplots(1, figsize=(80*mm, 50*mm))
    for j, dir_name in enumerate(dir_list):
        reach_list = []
        for fileno, D_rot in enumerate(D_rot_list):
            path = dir_name + '/D' + str(fileno) + '_'
            df = pd.read_csv(path + 'bact_stat.csv', header=None, index_col=0).squeeze("columns")
            n_bacteria = df['total number of agents']
            n_targ = df['agents reached the target']
            reach = 100*n_targ/(n_bacteria)
            reach_list.append(reach)
        Ltext = Lambda_text[j]
        ax.scatter(D_rot_list, reach_list, color=colors[j], s=1)
        ax.plot(D_rot_list, reach_list, color=colors[j], label = Ltext, linewidth=0.5)
    ax.set_ylabel(r'reach [$\%$]', rotation=90, labelpad=-2)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
    ax.legend(loc='lower left', title=r'$\lambda$', title_fontsize=5,
              fontsize=5, facecolor='white', framealpha=0.5)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_ylim([75, 101])
    ax.axhline(y=100, color='cadetblue', linestyle="--", lw=1.0)

    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/reach' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()    
def lost_vs_Drot():
    from inputs import D_rot_list, Lambda_list
    dir_list = ['lambda_'+str(j) for j in range(len(Lambda_list))] + ['ideal']
    Lambda_text = [format_latex(L) for L in Lambda_list] + [r'ideal']
    colors_blue = plt.cm.Blues(np.linspace(0.35, 1, len(Lambda_text)))   
    fig, ax = plt.subplots(1, figsize=(80*mm, 50*mm))
    for j, dir_name in enumerate(dir_list):
        lost_list = []
        for fileno, D_rot in enumerate(D_rot_list):
            path = dir_name + '/D' + str(fileno) + '_'
            df = pd.read_csv(path + 'bact_stat.csv', header=None, index_col=0).squeeze("columns")
            n_bacteria = df['total number of agents']
            n_horz = df['agents moved out']
            lost = 100*n_horz/(n_bacteria)
            lost_list.append(lost)
        Ltext = Lambda_text[j]
        ax.scatter(D_rot_list, lost_list, color=colors_blue[j], s=1)
        ax.plot(D_rot_list, lost_list, color=colors_blue[j], label = Ltext, linewidth=0.5)
    ax.set_ylabel(r'lost [$\%$]', rotation=90, labelpad=2)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
    ax.legend(loc='upper left', title=r'$\lambda$', title_fontsize=5,
              fontsize=5, facecolor='white', framealpha=0.5)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_ylim([0, 30])
    #ax.axhline(y=100, color='cadetblue', linestyle="--", lw=1.0)

    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/lost' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()    

def active_vs_Drot():
    from inputs import D_rot_list, Lambda_list
    dir_list = ['lambda_'+str(j) for j in range(len(Lambda_list))] + ['ideal']
    Lambda_text = [format_latex(L) for L in Lambda_list] + [r'ideal']
    colors_green = plt.cm.Greens(np.linspace(0.35, 1, len(Lambda_text)))
    df_act = pd.DataFrame(D_rot_list, columns=['D_rot'])
    fig, ax = plt.subplots(1, figsize=(80*mm, 50*mm))
    for j, dir_name in enumerate(dir_list):
        active_list = []
        for fileno, D_rot in enumerate(D_rot_list):
            path = dir_name + '/D' + str(fileno) + '_'
            df = pd.read_csv(path + 'bact_stat.csv', header=None, index_col=0).squeeze("columns")
 #           n_bacteria = df['total number of agents']
            #
            n_still_active = df['agents still roaming around']
            active = n_still_active*1.0
            active_list.append(active)
        Ltext = Lambda_text[j]
        ax.scatter(D_rot_list, active_list, color=colors_green[j], s=1)
        ax.plot(D_rot_list, active_list, color=colors_green[j], label = Ltext, linewidth=0.5)
        df_act[Ltext] = active_list
    ax.set_ylabel(r'still running [count]', rotation=90, labelpad=2)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
    ax.legend(loc='lower left', title=r'$\lambda$', title_fontsize=5,
              fontsize=5, facecolor='white', framealpha=0.5)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_yscale('symlog', linthresh=0.001)
    #ax.set_ylim([0, 6])
    #ax.axhline(y=100, color='cadetblue', linestyle="--", lw=1.0)

    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/active' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()
    print(df_act)
    write_dataframe(df_act, 'still_active')
    
if __name__ == '__main__':
    pass
