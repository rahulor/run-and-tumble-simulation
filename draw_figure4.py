import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def score_CI_vs_Drot():
    from inputs import Lambda_list
    dir_list = ['lambda_'+str(i) for i in range(len(Lambda_list))] + ['ideal']
    Lambda_text = [format_latex(L) for L in Lambda_list] + [r'ideal']
    colors = plt.cm.Reds(np.linspace(0.35, 1, len(Lambda_text)))
 
    fig, axs = plt.subplots(2, figsize=(80*mm, 80*mm), sharex=True)
    ax = axs[0]
    for j, dir_name in enumerate(dir_list):
        df = pd.read_csv(dir_name+'/ci_vs_Drot.csv', index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        ci = df['ci'].values # entire column
        #ci_sem = df['ci_sem'].values # entire column
        ax.scatter(D_rot, ci, color=colors[j], s=1)
        ax.plot(D_rot, ci, label = Ltext, color=colors[j], linewidth=0.5)
        #ax.fill_between(D_rot, ci - ci_sem/2, ci + ci_sem/2, color='gray')
    ax.set_ylabel(r'CI', rotation=0, labelpad=8)
    ax.set_xscale('symlog', linthresh=0.001)
    
    ax = axs[1]
    file_begin = 'data/score_mean_'
    for j, Lambda in enumerate(Lambda_list):
        f = file_begin + str(j) + '.csv'
        df = pd.read_csv(f, index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        score = df['score'].values # entire column
        ax.scatter(D_rot, score*100, color=colors[j], s=1)
        ax.plot(D_rot, score*100, label = Ltext, color=colors[j], linewidth=0.5)
    ideal_score = np.ones(len(D_rot))
    ax.scatter(D_rot, ideal_score*100, color=colors[-1], s=1)
    ax.plot(D_rot, ideal_score*100, label = Lambda_text[-1], color=colors[-1], linewidth=0.5)
    hpos = 0.5*D_rot[-1]
    ax.annotate(r'', xy=(hpos, 95), xytext=(hpos, 60), color='black',
                arrowprops={'arrowstyle': '->', 'lw': 0.3, 'color': 'black'},
                va='top', ha='center', size=10)
    ax.text(hpos*1.1, 70, r'$\lambda$', color='black', size=8, ha='left', va='center')
    ax.set_ylabel(r'score [$\%$]', rotation=90, labelpad=0)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
   
    ax.legend(loc='lower left', title=r'$\lambda\, [1/(\mu\mathrm{M}\, \mathrm{s})]$', title_fontsize=7,
              fontsize=7, facecolor='white', framealpha=0.5)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_ylim([50, 102])

    fig.subplots_adjust(hspace=0.05)
    figpath = 'figure4' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()

def CI_vs_Drot():
    from inputs import Lambda_list
    dir_list = ['lambda_'+str(i) for i in range(len(Lambda_list))] + ['ideal']
    Lambda_text = [format_latex(L) for L in Lambda_list] + [r'ideal']
    colors = plt.cm.Reds(np.linspace(0.35, 1, len(Lambda_text)))
 
    fig, ax = plt.subplots(1, figsize=(80*mm, 40*mm))
    for j, dir_name in enumerate(dir_list):
        df = pd.read_csv(dir_name+'/ci_vs_Drot.csv', index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        ci = df['ci'].values # entire column
        #ci_sem = df['ci_sem'].values # entire column
        ax.scatter(D_rot, ci, color=colors[j], s=1)
        ax.plot(D_rot, ci, label = Ltext, color=colors[j], linewidth=0.5)
        #ax.fill_between(D_rot, ci - ci_sem/2, ci + ci_sem/2, color='gray')
    ax.set_ylabel(r'CI', rotation=0, labelpad=8)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
    ax.set_xscale('symlog', linthresh=0.001)
    
    # hpos = 0.5*D_rot[-1]
    # ax.annotate(r'', xy=(hpos, 0.4), xytext=(hpos, 60), color='black',
    #             arrowprops={'arrowstyle': '->', 'lw': 0.3, 'color': 'black'},
    #             va='top', ha='center', size=10)
    # ax.text(hpos*1.1, 0.2, r'$\lambda$', color='black', size=8, ha='left', va='center')
    
   
    ax.legend(loc='lower center', title=r'$\lambda\, [1/(\mu\mathrm{M}\, \mathrm{s})]$', title_fontsize=7,
              fontsize=7, facecolor='white', framealpha=0.5)
    #fig.subplots_adjust(hspace=0.05)
    figpath = 'figure4' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()

def collect_ci():
    from inputs import D_rot_list, Lambda_list
    dir_list = ['ideal']  + ['lambda_'+str(i) for i in range(len(Lambda_list))]
    for dir_name in dir_list:
        ci_list = []
        for fileno, D_rot in enumerate(D_rot_list):
            path = dir_name + '/D' + str(fileno) + '_'
            df = pd.read_csv(path + 'ci.csv', header=None, index_col=0).squeeze("columns")
            ci_list.append([D_rot, df.loc['mean'], df.loc['sem']])
        df_ci = pd.DataFrame(ci_list, columns=['D_rot', 'ci', 'ci_sem'])
        df_ci.to_csv(dir_name+'/ci_vs_Drot.csv', index=None)
def draw_all():
    collect_ci()
    CI_vs_Drot()
if __name__ == '__main__':
    draw_all()
    