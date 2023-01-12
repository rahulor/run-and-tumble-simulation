import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# import scipy.stats
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    'axes.labelsize': 10,
    'xtick.labelsize':8,
    'ytick.labelsize':8
})
fig_ext = '.eps'
mm = 1/25.4  # mm in inches
Lambdaunit = r'$\, [1/(\mu\mathrm{M}\, \mathrm{s})]$'
Drotunit = r'$\, [1/s]$'
def write_dataframe(df, fname):
    pathcsv = fname + '.csv'
    pathtxt = fname + '.txt'
    df.to_csv(pathcsv)
    file = open(pathtxt, "w")
    text = df.to_string()
    #file.write("RESULT\n\n")
    file.write(text)
    file.close()

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

def time_to_target(lambda_idx, Drot_idx):
    dir_list = ['lambda_'+str(lambda_idx), 'ideal']
    from inputs import t_sim, D_rot_list, n_bacteria, Lambda_list
    #Lambda = Lambda_list[lambda_idx]
    print(f'D_rot_list  {D_rot_list}')
    colors = plt.cm.Blues(np.linspace(1, 0.5, len(Drot_idx)))
    
    #Lambdatext = [r'$\lambda = \, $' + format_latex(Lambda) + Lambdaunit, 'ideal'] 
    Lambdatext = [r'ML', 'ideal'] 
    fig, ax = plt.subplots(2, figsize=(54*mm, 60*mm), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    for i, dir_name in enumerate(dir_list):
        Ltext = Lambdatext[i]
        ax[i].text(0.45, 0.8, Ltext, color='black', fontsize=8,  transform=ax[i].transAxes, ha='right',
                bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'))
        rate_list = []
        prob_list = []
        for clr, fileno in enumerate(Drot_idx):
            path = dir_name + '/D' + str(fileno)
            D_rot = D_rot_list[fileno]
            Dtext = str(D_rot)
            df_targ = pd.read_csv(path + '_time_to_target.csv')
            time_to_targ = df_targ['time_to_target']
            count_targ, bins = np.histogram(time_to_targ, bins=100, range=(0, t_sim+5))
            binsize = (bins[1]-bins[0])
            count_targ = count_targ/(n_bacteria*binsize)
            ax[i].hist(bins[:-1], bins, weights=count_targ, histtype='step', color=colors[clr], 
                       label=Dtext, linewidth=0.5)
            #stats / percentage
            df_stat = pd.read_csv(path + '_bact_stat.csv', header=None, index_col=0).squeeze("columns")
            n_bacteria = df_stat['total number of agents']
            n_targ = df_stat['agents reached the target']
            rate = 100*n_targ/(n_bacteria)
            prob = np.sum(count_targ*binsize)
            rate_list.append(rate)
            prob_list.append(prob)
        ax[i].set_ylabel(r'$p (t)$ [1/s]', rotation=90, labelpad=3)
        ax[i].set_ylim(0,0.025)
        ax[i].set_xlim(0, 205)
        ax[i].minorticks_on()
        # ax[i].grid(which='major', color='gray', linewidth=0.1, linestyle = '-')
        # ax[i].grid(which='minor', color='gray', linewidth=0.1, linestyle = '-')
        
        axins = ax[i].inset_axes([0.8, 0.27, 0.35, 0.5])
        # axins.set(facecolor ='#cdcdcd')
        #axins.set(facecolor ='#cdcdcd')
        axins.minorticks_on()
        axins.grid(axis='x', which='minor', color='white', linewidth=0.1, linestyle = '-')
        axins.grid(axis='x', which='major', color='white', linewidth=0.1, linestyle = '-')
        axins.barh(np.arange(len(Drot_idx))*1.5, prob_list, tick_label='',
                    color=colors, zorder=2)
        axins.invert_yaxis()  # labels read top-to-bottom
        axins.invert_xaxis()
        axins.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                      bottom=True, top=False, left=False, right=False, pad=2)
        axins.yaxis.set_tick_params(which='minor', bottom=False)
        
        axins.set_xlim([0.65, 1.0])
        ax[i].text(0.99, 0.95, r'$\int p(t) dt$', color='black', size=7, transform=ax[i].transAxes,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'),
                ha='center', va='top') 
    ax[i].legend(loc=(0.49,0.25), title=r'$D_\mathrm{rot}$' , fontsize=7, title_fontsize=9)
    ax[i].set_xlabel(r'first passage time, t [s]', labelpad=0)
#    plt.savefig('fig/time_to_target'+ fig_ext, bbox_inches='tight', dpi=300)
    #fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('fig/time_to_target'+ fig_ext, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
def CIbins(lambda_idx, Drot_idx):
    dir_list = ['lambda_'+str(lambda_idx), 'ideal']
    from inputs import D_rot_list, horizon
    #Lambda = Lambda_list[lambda_idx]
    colors = plt.cm.Blues(np.linspace(1, 0.5, len(Drot_idx)))
    # Lambdatext = [r'$\lambda = \, $' + format_latex(Lambda), 'ideal'] 
    Lambdatext = [r'ML', 'ideal'] 
    fig, ax = plt.subplots(2, figsize=(30*mm, 35*mm), sharex=True)
    fig.subplots_adjust(hspace=0.0)
    #fig.suptitle(r'chemotactic index', size=10, y=0.99)
    for i, dir_name in enumerate(dir_list):
        Ltext = Lambdatext[i]
        ax[i].text(40,-0.7, Ltext, color='black', size=8, horizontalalignment='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'))
        for clr, fileno in enumerate(Drot_idx):
            path = dir_name + '/D' + str(fileno)
            D_rot = D_rot_list[fileno]
            Dtext = str(D_rot)
            df = pd.read_csv(path + '_ci_bins.csv')
            ci = df['ci'].values[:-1]
            bins = df['bin_edge'].values
            CI = ci.copy()
            ax[i].hist(bins[:-1], bins, weights=CI, color=colors[clr], histtype='step', 
                       label=Dtext, linewidth=0.5)
        ax[i].set_ylabel(r'$\mathrm{CI}$', rotation=90, labelpad=-6)
        ax[i].set_ylim(-1,1)
        ax[i].minorticks_on()
        #ax[i].grid(which='major', color='white', linewidth=0.1, linestyle = '-')
        ax[i].grid(axis='both', which='major', color='white', linewidth=0.1, linestyle = '-')
        ax.grid(axis='y', which='major', color='tan', linestyle = '-', linewidth=0.7, zorder=-1)
        #ax[i].axvspan(food_surface+bact_speed, horizon-bact_speed, color='#d9d9d9', zorder=0)
        #ax[i].axvspan(horizon, horizon+bact_speed, color='gray', zorder=2)
        #ax[i].axvspan(0, food_surface, color='magenta', zorder=2)
        #ax[i].axhline(y=0, color='white', linestyle='--', linewidth=0.1)
    #ax[i].legend(loc=(0.4, 0.01), title=r'$D_\mathrm{rot}$', fontsize=4.5, ncol=1, framealpha=1.0)    
    ax[i].set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=1)
    ax[i].set_xlim([0, horizon+0.5])    
    #plt.yscale('log')
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('fig/CIbins'+ fig_ext, bbox_inches='tight', dpi=400,  pad_inches=0.05)
    plt.show()   

def combined_hist_grid_with_tres(lambda_idx, Drot_idx):
    dir_name = 'lambda_'+str(lambda_idx) 
    path = dir_name + '/D' + str(Drot_idx) + '_'
    from inputs import horizon, Lambda_list, D_rot_list
    Lambdatext = r'$\lambda = \, $' + format_latex(Lambda_list[lambda_idx])
    Drottext = r'$D_{\mathrm{rot}} =\, $' + str(D_rot_list[Drot_idx])
    fig, axs = plt.subplots(2,2, figsize=(75*mm, 55*mm), sharex=True, gridspec_kw={'height_ratios': [1,1]})
    #
    fig.suptitle(Lambdatext + ';' + Drottext, size = 7)
    #
    ax = axs[0][0]
    df = pd.read_csv(path + 'dist_FN.csv')
    count = df['count'].values[:-1]
    bins = df['bins'].values
    binsize = bins[1]-bins[0]
    count = count/binsize
    ax.hist(bins[:-1], bins, weights=count, color='cadetblue', histtype='stepfilled',  label=r'FN [$1/\mu\mathrm{m}$]')
    ax.set_yscale('log')
    #
    df = pd.read_csv(path + 'dist_FP.csv')
    count = df['count'].values[:-1]
    bins = df['bins'].values
    binsize = bins[1]-bins[0]
    count = count/binsize
    ax.hist(bins[:-1], bins, weights=count, color='darkseagreen', histtype='stepfilled', label=r'FP [$1/\mu\mathrm{m}$]')
    ax.set_yscale('log')
    #
    ax.legend(loc='center left', fontsize=5)
    ax.minorticks_on()
    # ax.grid(axis='x', which='minor', color='gray', linestyle = '-', linewidth=1.0)
    # ax.grid(axis='y', which='major', color='gray', linestyle = '-', linewidth=1.0)
    #ax.set_ylim([10**1, 10**5])
    #
    ax = axs[1][0]
    df = pd.read_csv(path + 'res_time.csv')
    res_time = df['res_time'].values[:-1]
    bins = df['bin_edge'].values
    binsize = bins[1]-bins[0]
    res_time = res_time/binsize
    ax.hist(bins[:-1], bins, weights=res_time, color='rosybrown', histtype='stepfilled', label=r'$t_{\mathrm{res}}$ $[\mathrm{s}/\mu\mathrm{m}$]')
    ax.set_yscale('log')
    ax.legend(loc='center left', fontsize=6)
    ax.minorticks_on()
    ax.set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=-7)
    #
    df = pd.read_csv(path + 'dist_FP.csv')
    FP = df['count'].values[:-1]
    df = pd.read_csv(path + 'dist_FN.csv')
    FN = df['count'].values[:-1]
    df = pd.read_csv(path + 'dist_TP.csv')
    TP = df['count'].values[:-1]
    df = pd.read_csv(path + 'dist_TN.csv')
    TN = df['count'].values[:-1]
    bins = df['bins'].values
    rho1 = FP/(FP+TP)
    rho2 = FN/(FN+TN)
    #
    ax = axs[0][1]
    ax.hist(bins[:-1], bins, weights=rho1, label=r'FP/(FP+TP)', color='darkseagreen', histtype='stepfilled')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=5)
    ax.minorticks_on()
    ax.grid(axis='y', which='major', color='tan', linestyle = '-', linewidth=0.7, zorder=-1)
    #
    ax = axs[1][1]
    ax.hist(bins[:-1], bins, weights=rho2, label=r'FN/(FN+TN)', color='cadetblue', histtype='stepfilled')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=5)
    ax.minorticks_on()
    ax.grid(axis='y', which='major', color='tan', linestyle = '-', linewidth=0.7, zorder=-1)
    #
    ax.set_xlim([0, horizon+0.5])
    ax.set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=-7)
    fig.subplots_adjust(hspace=0.14, wspace=0.35)
    plt.savefig('fig/fpfn'+ fig_ext, bbox_inches='tight', dpi=400,  pad_inches=0.0)
    plt.show()

def combined_hist_grid(lambda_idx, Drot_idx):
    colors = plt.cm.Blues(np.linspace(1, 0.5, len(Drot_idx)))
    dir_name = 'lambda_'+str(lambda_idx) 
    from inputs import horizon, Lambda_list, D_rot_list
    Lambdatext = r'ML-agent, for ' +  r'$\lambda = \, $' + format_latex(Lambda_list[lambda_idx]) + Lambdaunit
    #
    tlsize = 6
    fig, axs = plt.subplots(2,2, figsize=(70*mm, 58*mm), sharex=True, gridspec_kw={'height_ratios': [1,1]})
    fig.text(0.5, 0.97, Lambdatext, color='black', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'), ha='center', va='top')
    ax = axs[0][0]
    title = r'FP [$1/\mu\mathrm{m}$]'
    ax.text(0.5, 0.9, title, color='black', size=tlsize, ha='center', va='center', transform=ax.transAxes)
    for clr, fileno in enumerate(Drot_idx):
        path = dir_name + '/D' + str(fileno) + '_'
        D_rot = D_rot_list[fileno]
        Dtext = str(D_rot)
        df = pd.read_csv(path + 'dist_FP.csv')
        count = df['count'].values[:-1]
        bins = df['bins'].values
        binsize = bins[1]-bins[0]
        count = count/binsize
        ax.hist(bins[:-1], bins, weights=count, color=colors[clr], histtype='step', label=Dtext, linewidth=0.5)
    ax.set_yscale('log')
    #ax.legend(loc='center left', fontsize=5)
    ax.minorticks_on()
    ax.set_ylim([10**1, 10**5])
    #
    ax = axs[1][0]
    title = r'FN [$1/\mu\mathrm{m}$]'
    ax.text(0.5, 0.9, title, color='black', size=tlsize, ha='center', va='center', transform=ax.transAxes)
    for clr, fileno in enumerate(Drot_idx):
        path = dir_name + '/D' + str(fileno) + '_'
        D_rot = D_rot_list[fileno]
        Dtext = str(D_rot)
        df = pd.read_csv(path + 'dist_FN.csv')
        count = df['count'].values[:-1]
        bins = df['bins'].values
        binsize = bins[1]-bins[0]
        count = count/binsize
        ax.hist(bins[:-1], bins, weights=count, color=colors[clr], histtype='step', label=Dtext, linewidth=0.5)
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.set_ylim([10**1, 10**5])
    ax.set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=-7)
    #
    ax = axs[0][1]
    title = r'FP/(FP+TP)'
    ax.text(0.5, 0.9, title, color='black', size=tlsize, ha='center', va='center', transform=ax.transAxes)
    for clr, fileno in enumerate(Drot_idx):
        path = dir_name + '/D' + str(fileno) + '_'
        D_rot = D_rot_list[fileno]
        Dtext = str(D_rot)
        df = pd.read_csv(path + 'dist_FP.csv')
        FP = df['count'].values[:-1]
        df = pd.read_csv(path + 'dist_TP.csv')
        TP = df['count'].values[:-1]
        bins = df['bins'].values
        rho1 = FP/(FP+TP)
        binsize = bins[1]-bins[0]
        count = count/binsize
        ax.hist(bins[:-1], bins, weights=rho1, color=colors[clr], histtype='step', label=Dtext, linewidth=0.5)
    ax.set_ylim([0, 1])
    #ax.legend(loc='center left', fontsize=5)
    ax.minorticks_on()
    ax.grid(axis='y', which='major', color='tan', linestyle = '-', linewidth=0.7, zorder=-1)
    #
    ax = axs[1][1]
    title = r'FN/(FN+TN)'
    ax.text(0.5, 0.9, title, color='black', size=tlsize, ha='center', va='center', transform=ax.transAxes)
    for clr, fileno in enumerate(Drot_idx):
        path = dir_name + '/D' + str(fileno) + '_'
        D_rot = D_rot_list[fileno]
        Dtext = str(D_rot)
        df = pd.read_csv(path + 'dist_FN.csv')
        FN = df['count'].values[:-1]
        df = pd.read_csv(path + 'dist_TN.csv')
        TN = df['count'].values[:-1]
        bins = df['bins'].values
        rho2 = FN/(FN+TN)
        binsize = bins[1]-bins[0]
        count = count/binsize
        ax.hist(bins[:-1], bins, weights=rho2, color=colors[clr], histtype='step', label=Dtext, linewidth=0.5)
    ax.set_ylim([0, 1])
    #ax.legend(loc='center left', title=r'$D_\mathrm{rot}$', fontsize=5)
    ax.minorticks_on()
    ax.grid(axis='y', which='major', color='tan', linestyle = '-', linewidth=0.7, zorder=-1)
    #
    ax.set_xlim([0, horizon+0.5])
    ax.set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=-7)
    fig.subplots_adjust(hspace=0.14, wspace=0.35)
    plt.savefig('fig/fpfn'+ fig_ext, bbox_inches='tight', dpi=400,  pad_inches=0.05)
    plt.show()
def ci_and_tres(lambda_idx, Drot_idx):
    dir_list = ['lambda_'+str(lambda_idx), 'ideal']
    from inputs import horizon, D_rot_list
    #
    Lambdatext = [r'ML', 'ideal'] + ['empty'] + [r'ML', 'ideal']
    colors = plt.cm.Blues(np.linspace(1, 0.5, len(Drot_idx)))
    gridspec = dict(hspace=0.28, height_ratios=[1, 1, 0.07, 1,1])
    fig, ax = plt.subplots(5, figsize=(30*mm, 60*mm), sharex=True, 
                        gridspec_kw=gridspec)
    ax[2].set_visible(False)

    # for ci
    for i, dir_name in enumerate(dir_list):
        Ltext = Lambdatext[i]
        ax[i].text(0.07, 0.1, Ltext, color='black', fontsize=6, transform=ax[i].transAxes,
                bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'), ha='left', va='bottom')
        for clr, fileno in enumerate(Drot_idx):
            path = dir_name + '/D' + str(fileno)
            D_rot = D_rot_list[fileno]
            Dtext = str(D_rot)
            df = pd.read_csv(path + '_ci_bins.csv')
            ci = df['ci'].values[:-1]
            bins = df['bin_edge'].values
            CI = ci.copy()
            ax[i].hist(bins[:-1], bins, weights=CI, color=colors[clr], histtype='step', 
                       label=Dtext, linewidth=0.5)
        ax[i].set_ylim(-1,1)
        ax[i].grid(axis='y', which='major', color='tan', linestyle = '-', linewidth=0.4, zorder=-1)
    fig.text(-0.24, 0.7, r'CI', va='center', rotation='vertical')
    # for residence time
    for i, dir_name in enumerate(dir_list, start=3):
        Ltext = Lambdatext[i]
        ax[i].text(0.07, 0.1, Ltext, color='black', fontsize=6, transform=ax[i].transAxes,
                bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'), ha='left', va='bottom')
        for clr, fileno in enumerate(Drot_idx):
            path = dir_name + '/D' + str(fileno) + '_'
            D_rot = D_rot_list[fileno]
            Dtext = str(D_rot)
            df = pd.read_csv(path + 'res_time.csv')
            res_time = df['res_time'].values[:-1]
            bins = df['bin_edge'].values
            binsize = bins[1]-bins[0]
            res_time = res_time/binsize
            ax[i].hist(bins[:-1], bins, weights=res_time, color=colors[clr], histtype='step', 
                    label=Dtext, linewidth=0.5)
        ax[i].set_yscale('log')
        ax[i].minorticks_on()
        ax[i].grid(axis='both', which='major', color='white', linewidth=0.1, linestyle = '-')
        ax[i].set_ylim(None, 1e4)
    fig.text(-0.2, 0.2, r'$t_{\mathrm{res}} [\mathrm{s}/\mu \mathrm{m}]$', ha='center', rotation='vertical')
    #ax[2].set_ylabel(r'$t_{\mathrm{res}} [\mathrm{s}/\mu \mathrm{m}]$', rotation=90, labelpad=2)
    ax[-1].set_xlabel(r'$\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]', labelpad=-8)
    ax[-1].set_xlim([0, horizon+0.5])
    plt.savefig('fig/ci_and_tres'+ fig_ext, bbox_inches='tight', dpi=400,  pad_inches=0.05)
    plt.show()
    #
def draw_all():
    lambda_idx = 0
    Drot_idx = [0,3,5]
    time_to_target(lambda_idx, Drot_idx)
    ci_and_tres(lambda_idx, Drot_idx)
    combined_hist_grid(lambda_idx, Drot_idx)
    
    if os.system("latex figure2.tex"):
        print('Error; no dvi output')
    else:
        print('Done!')
        os.system("dvips figure2.dvi -o figure2.eps")
if __name__ == '__main__':
    draw_all()
    