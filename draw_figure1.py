import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=8)
import os
import helper
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    'axes.labelsize': 10,
    'axes.titlesize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8
})

fig_ext = '.eps'
mm = 1/25.4  # mm in inches
food_color = 'magenta'
horizon_color = 'gray'
gradcolor = 'magenta' # 
def write_dataframe(df, fname):
    pathcsv = 'data/' + fname + '.csv'
    pathtxt = 'data/' + fname + '.txt'
    df.to_csv(pathcsv)
    file = open(pathtxt, "w")
    text = df.to_string()
    file.write(text)
    file.close()

def trajectory_training():
    n_samples = 500 # to be seen in the figure
    print('plot trajectory_training() '.ljust(40, '-'))
    traj_all = helper.unpickle_from('data/sample_trajectory')
    ytar = helper.unpickle_from('data/sample_signal_label')
    N = min(n_samples, len(traj_all))
    traj_list = traj_all[:N]
    fig, ax = fig, ax = setup_trajectory_figure()
    ax.set_title(r'training trajectories')
    # zoomed plot
    axins = zoomed_inset_axes(ax, zoom=4.5, loc='upper right')
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.4")
    axins.set_xticks([])
    axins.set_yticks([])
    # 
    for i, R in enumerate(traj_list):
        Rx = R[:,0]
        Ry = R[:,1]
        if ytar[i]==True:
            ax.plot(Rx, Ry, color='tab:green', lw=0.5)
            axins.plot(Rx, Ry, color='tab:green', lw=0.5)
            dx, dy = Rx[-1] -  Rx[-2], Ry[-1] -  Ry[-2]
            axins.arrow(Rx[-1], Ry[-1], dx, dy, width=0.01, head_width=3.0, overhang=0.1, 
                     length_includes_head=True, color='tab:green')
        else:
            ax.plot(Rx, Ry, color='tomato', lw=0.5)
            axins.plot(Rx, Ry, color='tomato', lw=0.5)
            dx, dy = Rx[-1] -  Rx[-2], Ry[-1] -  Ry[-2]
            axins.arrow(Rx[-1], Ry[-1], dx, dy, width=0.01, head_width=3.0, overhang=0.1, 
                     length_includes_head=True, color='tomato')
    xpos, ypos = -250, 150
    width = 100
    axins.set_xlim([xpos, xpos+width])
    axins.set_ylim([ypos, ypos+width])
    axins.set_aspect('equal')
    # blue arrow towards the target
    bluex, bluey = xpos+width*0.1, ypos+width*0.8
    lenth_arrow = width*0.7
    R = np.array([bluex, bluey])
    r = np.linalg.norm(R)
    e_r =  -R/r # towards center
    dx, dy = e_r[0]*lenth_arrow, e_r[1]*lenth_arrow
    axins.annotate('', xytext=(bluex, bluey), xy=(bluex+dx, bluey+dy),  
                arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': gradcolor},
                va='center', zorder=0)
    axins.text(bluex, bluey, r'$\nabla c$', color=gradcolor, size=8, ha='left', va='bottom')
    scalebar = AnchoredSizeBar(axins.transData,
                           50, r'$50\, \mathrm{\mu m}$', 'lower center', 
                           pad=0.2,
                           color='black',
                           frameon=False,
                           size_vertical=0,
                           label_top=True,
                           fontproperties=fontprops)
    axins.add_artist(scalebar)
    plt.savefig('fig/trajectory_training'+ fig_ext, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.show()
def setup_trajectory_figure():
    from inputs import food_surface, horizon
    fig, ax = plt.subplots(figsize=(55*mm,55*mm))
    food = plt.Circle((0,0), food_surface, color=food_color, zorder=2)
    horz = plt.Circle((0,0), horizon, color='gray', lw=0.7, fill=False)
    ax.add_patch(horz)
    ax.add_patch(food)
    ax.set_xlim([-1.02*horizon, 1.02*horizon])
    ax.set_ylim([-1.02*horizon, 1.02*horizon])
    ax.set_xlabel(r'$x \, [\mu \mathrm{m}]$', rotation=0, labelpad=0)
    ax.set_ylabel(r'$y \, [\mu \mathrm{m}]$', rotation=90, labelpad=-12)
    ax.set_aspect('equal')
    return fig, ax
    
def trajectory_simulation():
    print('plot trajectory_simulation() '.ljust(40, '-'))
    fig, ax = setup_trajectory_figure()
    ax.set_title(r'trained agents', size=10)
    # zoomed plot
    axins = zoomed_inset_axes(ax, zoom=4.5, loc='upper right')
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")
    axins.set_xticks([])
    axins.set_yticks([])
    # succesfull agents first
    dir_name = 'data/traj_succ/'
    N = 10
    files = os.listdir(os.getcwd()+ '/' + dir_name)
#    files = [files[j] for j in [0,1,2]]
    files = [files[j] for j in [10,11,15,19,22,23]]
    for f in files:
        df = pd.read_csv(dir_name + f)
        ax.plot(df['x'], df['y'], color='tab:green', lw=0.5, zorder=0)
        # in zoomed plot
        axins.plot(df['x'], df['y'], color='tab:green', lw=0.5) # to avoid first tumble/initial point
        mask = df['tumble']==True
        dftumble = df[mask]
        axins.scatter(dftumble['x'][1:], dftumble['y'][1:], color='#0012ff', s=4., marker='*', zorder=2)
    # now draw failed ones      
    N = 15
    dir_name = 'data/traj_fail/'
    files = os.listdir(os.getcwd()+ '/' + dir_name)
    for f in files[:N]:
        df = pd.read_csv(dir_name + f)
        ax.plot(df['x'], df['y'], color='gray', lw=0.5, zorder=1) 
    #
    xpos, ypos = -200, -320
    width = 100
    axins.set_xlim([xpos, xpos+width])
    axins.set_ylim([ypos, ypos+width])
    axins.set_aspect('equal')
    scalebar = AnchoredSizeBar(axins.transData,
                            50, r'$50\, \mathrm{\mu m}$', 'lower center', 
                            pad=0.2,
                            color='black',
                            frameon=False,
                            size_vertical=0,
                            label_top=True,
                            fontproperties=fontprops)
    # blue arrow towards the target
    bluex, bluey = xpos+width*0.7, ypos+width*0.4
    lenth_arrow = width*0.5
    R = np.array([bluex, bluey])
    r = np.linalg.norm(R)
    e_r =  -R/r # towards center
    dx, dy = e_r[0]*lenth_arrow, e_r[1]*lenth_arrow
    axins.annotate('', xytext=(bluex, bluey), xy=(bluex+dx, bluey+dy),  
                arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': gradcolor},
                va='center', zorder=0)
    axins.text(bluex, bluey, r'$\nabla c$', color=gradcolor, size=8, ha='left', va='top')
    axins.add_artist(scalebar)
    plt.savefig('fig/trajectory_simulation'+ fig_ext , dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.show()
    

def pick_two_signals_starting_nearly_from(approx_R0, traj_all, label):
    R0_and_label = []
    for i, R in enumerate(traj_all):
        R0_vect = R[0]
        R0 = np.linalg.norm(R0_vect) # radial distance of the initial position
        R0_and_label.append([R0, label[i]])
    df = pd.DataFrame(R0_and_label, columns=['R0', 'label'])
    #print(df)
    r1, r2 = approx_R0-5, approx_R0+5
    mask = (r1<df['R0']) & (df['R0']<r2)
    df_masked = df[mask]
    k = 1 # small int; change to pick a different signal
    index_true = df_masked .loc[df_masked ['label'] == True].index[k]
    index_false = df_masked .loc[df_masked ['label'] == False].index[k]
    snr_index = [index_true, index_false]
    if len(snr_index) != 2:
        print('Check here! redo with r1, r2 = approx_R0-10, approx_R0+10')
    return snr_index 

def signal_and_weight():
    df_signal = pd.read_csv('data/sample_signal.csv', index_col=None)
    label = helper.unpickle_from('data/sample_signal_label')
    traj_all = helper.unpickle_from('data/sample_trajectory')
    signal = df_signal.values
    signal = np.flip(signal, axis=1)
    from inputs import T_max, memory
    dt = T_max/memory
    time = np.arange(memory) * dt
    
    R0_list = [200, 300, 400]
    high_snr_index = pick_two_signals_starting_nearly_from(R0_list[0], traj_all, label)
    moderate_snr_index = pick_two_signals_starting_nearly_from(R0_list[1], traj_all, label)
    low_snr_index = pick_two_signals_starting_nearly_from(R0_list[2], traj_all, label)

    # all_snr = [high_snr_index, moderate_snr_index, low_snr_index]
    # snrtext = [r'high SNR', r'moderate SNR', r'low SNR']
    all_snr = [high_snr_index, low_snr_index]
    snrtext = [r'high SNR', r'low SNR']
    R0_text = [r'$R_0 = 200\, \mu \mathrm{m}$', r'$R_0 = 400\, \mu \mathrm{m}$']
    
    #
    gridspec = dict(hspace=0.25, height_ratios=[1, 1, 0.17, 1.2])
    fig, axs = plt.subplots(1+1+len(all_snr), figsize=(28*mm, 55*mm),
                            gridspec_kw=gridspec)
    axs[-2].set_visible(False)
    title = r'$s(t)/\langle s\rangle$'
    axs[0].set_title(title, y=1.01, size=10)
    for i, ax in enumerate(axs[:len(all_snr)]):
        snr_index = all_snr[i]
        for j in snr_index:
            if label[j]==True:
                ax.plot(time, signal[j], lw=0.5, color='tab:green', label=r'+ve')
            else:
                ax.plot(time, signal[j], lw=0.5, color='tomato', label=r'-ve')
        ax.set_yticks([0,2])
        ax.set_xticks([])
        ax.text(0.98, 0.08, R0_text[i], color='black', size=8, ha='right', va='bottom', 
                    transform=ax.transAxes)#,
                    #bbox=dict(facecolor='cadetblue', edgecolor='white', boxstyle='round'))
    # only for last binding events
    ax = axs[len(all_snr)-1]
    ax.set_xticks([0,1])
    ax.minorticks_on()
    ax.yaxis.set_tick_params(which='minor', left=False)
    ax.set_xlabel(r'$t$ [s]', rotation=0, labelpad=-10)
    # weight profile
    ax = axs[-1]
    from load_data import lambda_index, D_rot_index
    from inputs import T_max, memory
    tau = np.arange(memory) * dt
    df = pd.read_csv('data/weights_mean_' + str(lambda_index) + '.csv', index_col=None)
    item = df.iloc[D_rot_index]
    b, w = item['bias'], item.loc['w0':].values
    # print(f'bias = {b}')
    w = w/np.max(w)
    wmin = np.min(w)
    epsilon = wmin*5/100
    k_wmin = np.argmin(w)
    for k_eff in range(k_wmin, memory):
        if w[k_eff]>epsilon:
            break
    T_eff = (k_eff+1) * dt
    color_gray = '#c6bdba'
    ax.plot(tau, w, color='tab:blue', label = r'memory kernel', lw=0.5)
    height = 0.2
    dist = T_eff*0.4
    ax.annotate(r'$T_{\mathrm{eff}}$', xy=(dist, height+0.1),  
                fontsize=8, va='bottom')
    ax.annotate('', xy=(T_eff, height), xytext=(0, height),  
                fontsize=7,
                arrowprops={'arrowstyle': '<->', 'lw': 0.3, 'color': 'black'},
                va='center')
    ax.minorticks_on()
    ax.set_xticks([0,1])
    ax.axhline(y=0.0, color=color_gray, linestyle=":", lw=1.0)
    ax.set_xlabel(r'$\tau$ [s]', size=10,  rotation=0, labelpad=-5)
    ax.set_ylabel(r'$w(\tau)$', size=10,  rotation=90, labelpad=1)

    #fig.tight_layout(pad=0, w_pad=1, h_pad=0.1)
    figpath = 'fig/signal_weight' + fig_ext
    plt.savefig(figpath, bbox_inches='tight', dpi=400, pad_inches=0.05)
    plt.show()
    

def draw_all():
    args = pd.read_csv('data/sample_args.csv', index_col=0).squeeze('rows')
    print(args)
    trajectory_training()
    trajectory_simulation()
    signal_and_weight()
    
    if os.system("latex figure1.tex"):
        print('Error; no dvi output')
    else:
        print('Done!')
        os.system("dvips figure1.dvi -o figure1.eps")
if __name__ == '__main__':
    pass
    draw_all()


    