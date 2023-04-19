# various different plotting functions

import numpy as np
from numpy.random import RandomState
from scipy.optimize import curve_fit
import time
import matplotlib;
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patches as patches
import libraries.inputs as inputs
from celluloid import Camera
from mpl_toolkits import mplot3d

import libraries.functions as fn

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# predifining some colors
gray=(100/255, 100/255, 100/255)
lightgray=(150/255, 150/255, 150/255)
black=(0,0,0)
white=(1,1,1)

lightred     = matplotlib.colors.to_rgba('#E56767', alpha=None)
red          = matplotlib.colors.to_rgba('#CC414D', alpha=None)
darkred      = matplotlib.colors.to_rgba('#80262F', alpha=None)

lightpurple  = matplotlib.colors.to_rgba('#AC7DF0', alpha=None)
purple       = matplotlib.colors.to_rgba('#7E4BCC', alpha=None)
darkpurple   = matplotlib.colors.to_rgba('#502E66', alpha=None)

lightblue    = matplotlib.colors.to_rgba('#66ADFF', alpha=None)
blue         = matplotlib.colors.to_rgba('#3C7DC4', alpha=None)
darkblue     = matplotlib.colors.to_rgba('#204D80', alpha=None)
darkdarkblue = matplotlib.colors.to_rgba('#0E2D46', alpha=None)

lightgreen   = matplotlib.colors.to_rgba('#3FD1B9', alpha=None)
green        = matplotlib.colors.to_rgba('#00A88C', alpha=None)
darkgreen    = matplotlib.colors.to_rgba('#157267', alpha=None)

lightyellow  = matplotlib.colors.to_rgba('#FAEA82', alpha=None)
yellow       = matplotlib.colors.to_rgba('#F7E04A', alpha=None)
darkyellow   = matplotlib.colors.to_rgba('#9F890D', alpha=None)

lightorange  = matplotlib.colors.to_rgba('#FABB70', alpha=None)
orange       = matplotlib.colors.to_rgba('#F09A35', alpha=None)
darkorange   = matplotlib.colors.to_rgba('#A06302', alpha=None)

#####

color_dict = {
        'black':black,
        'white':white,
        'gray':gray,
        'lightgray':lightgray,
        'lightred':lightred,
        'red':red,
        'darkred':darkred,
        'lightpurple':lightpurple,
        'purple':purple,
        'darkpurple':darkpurple,
        'lightblue':lightblue,
        'blue':blue,
        'darkblue':darkblue,
        'lightgreen':lightgreen,
        'green':green,
        'darkgreen':darkgreen,
        'lightyellow':lightyellow,
        'yellow':yellow,
        'darkyellow':darkyellow,
        'lightorange':lightorange,
        'orange':orange,
        'darkorange':darkorange
        }

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[blue, red, green, orange, purple, yellow, lightblue, lightred, lightgreen, lightorange, lightpurple, lightyellow])


TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
"""
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the subfigure title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize

plt.rc('legend', frameon=False)          # legend frame off

plt.rcParams['axes.xmargin'] = 0.001
plt.rcParams['axes.ymargin'] = 0.001

plt.rcParams["figure.figsize"] = [4.0/2.54, 3.2/2.54]  # figsize in inch, 1' = 2.54cm
"""
plot_linewidth = 1
plt.rcParams['lines.linewidth'] = plot_linewidth   # linewidth of plotted lines

plt.rcParams['lines.dash_capstyle']  = 'round'  #butt # {butt, round, projecting}
plt.rcParams['lines.solid_capstyle'] = 'round'  #projecting {butt, round, projecting}

linewidth_axes = 0.3
plt.rcParams['axes.linewidth']= linewidth_axes     ## axes and ticks linewidths
plt.rcParams['ytick.major.width'] = linewidth_axes
plt.rcParams['xtick.major.width'] = linewidth_axes
plt.rcParams['ytick.minor.width'] = linewidth_axes
plt.rcParams['xtick.minor.width'] = linewidth_axes
plt.rcParams["xtick.major.size"] = 2
plt.rcParams["ytick.major.size"] = 2

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

plt.rcParams["figure.facecolor"] = white

#plt.rcParams["mpl_toolkits.legacy_colorbar"] = False

# Font & Latex
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue']
plt.rcParams['text.usetex'] = True  # set to False to get proper text fonts but improper math fonts
plt.rcParams['pgf.texsystem'] = 'lualatex'
plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['text.latex.preamble'] = (
    r'\usepackage{tgheros}'     # helvetica font
    r'\usepackage{sansmath}'    # math-font matching helvetica
    r'\sansmath'                # actually tell tex to use it
    r'\RequirePackage{amsmath,amsthm}'
    r'\RequirePackage[upint]{newtxsf}'
    r'\DeclareSymbolFont{operators}{T1}{\sfdefault}{m}{n}'
    r'\DeclareMathSymbol{0}{\mathord}{operators}{`0}'  # overwrite italic digits from sansmath package!
    r'\DeclareMathSymbol{1}{\mathord}{operators}{`1}'
    r'\DeclareMathSymbol{2}{\mathord}{operators}{`2}'
    r'\DeclareMathSymbol{3}{\mathord}{operators}{`3}'
    r'\DeclareMathSymbol{4}{\mathord}{operators}{`4}'
    r'\DeclareMathSymbol{5}{\mathord}{operators}{`5}'
    r'\DeclareMathSymbol{6}{\mathord}{operators}{`6}'
    r'\DeclareMathSymbol{7}{\mathord}{operators}{`7}'
    r'\DeclareMathSymbol{8}{\mathord}{operators}{`8}'
    r'\DeclareMathSymbol{9}{\mathord}{operators}{`9}'
    )

##############################################################################################################################################################
##############################################################################################################################################################


###############################################################################
# plot all connectivity matrices
def connectivity_matrix(W, **kwargs):

    N_pc, N_pv = W[1].shape
    N = N_pc + N_pv

    I_color = red
    E_color = blue
    rec_color= [[E_color, E_color], #U #pre_color,post_color
                [I_color, E_color], #M
                [E_color, I_color], #Q
                [I_color, I_color]] #P

    w, h = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(w*1.0, w*1.0))

    grid = AxesGrid(fig, 111,
            nrows_ncols=(2, 2),
            aspect = True,
            axes_pad=0.05,
            cbar_mode='single',
            cbar_location='right',
            cbar_pad=0.05,
            cbar_size=0.05
            )

    vmax = np.zeros(len(W))
    for i in range(len(W)):
        y_count, x_count = W[i].shape
        vmax = np.max(W[i])
        im = grid[i].imshow(W[i], cmap='Greys', interpolation='nearest', vmin=0, vmax=vmax, aspect='auto')  # keep aspect=auto here, otherwise all matrices have different sizes and do not line up
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        grid[i].set_xticklabels([])
        grid[i].set_yticklabels([])
        grid[i].spines['top'].set_visible(False)
        grid[i].spines['right'].set_visible(False)
        grid[i].spines['left'].set_visible(False)
        grid[i].spines['bottom'].set_visible(False)

        # Create a Rectangle patch
        offset = 0  # adds whitespace between matrix and colored rectangles
        spines_on = 0  # adds offset when spines are on, such that spines and rectangles fit perfectly
        rec_width = 2  # width of rectangles
        rect_pre = patches.Rectangle((-0.5-spines_on,-0.5+W[i].shape[0]+offset),W[i].shape[1]+2*spines_on,rec_width,linewidth=0,facecolor=rec_color[i][0], clip_on=False)  # pre
        rect_post = patches.Rectangle((-0.5-offset,-0.5-spines_on),-rec_width,W[i].shape[0]+2*spines_on,linewidth=0,facecolor=rec_color[i][1], clip_on=False)  # post
        # Add the patch to the Axes
        grid[i].add_patch(rect_pre)
        grid[i].add_patch(rect_post)

    cbar = grid[i].cax.colorbar(im)

    cbar.set_label(r'syn. weight', labelpad=-18, y=0.5)
    cbar.set_ticks([0,np.max(vmax)])
    cbar.set_ticklabels(['0',r'$w^{\,AB}_{\text{max}}$'])

    #title & labels
    if ('xlabel' in kwargs):
            grid[2].set_xlabel(kwargs['xlabel'])
            grid[2].xaxis.set_label_coords(0.72, -0.4)
    if ('ylabel' in kwargs):
            grid[0].set_ylabel(kwargs['ylabel'])
            grid[0].yaxis.set_label_coords(-0.12, 0.32)

    # if ('show_plot' in kwargs) and kwargs['show_plot'] == 1:
    #     plt.show(block=False)
    # elif ('save_path' in kwargs):
    #     plt.savefig(kwargs['save_path'], bbox_inches='tight', dpi=300)
    #     plt.close()
    # else:
    #     plt.close()


###############################################################################
# A general line plot
def line_plot(x_data, y_data, **kwargs):

    N_x = len(x_data)
    N_y = len(y_data)

    if N_x < N_y:  # when fewer x_data are available, copy last x_data to match number of y_data
        x_data = x_data + [None]*(N_y-N_x)
        for i in range(N_y-N_x):
            x_data[i+N_x] = x_data[N_x-1]

    for i in np.arange(N_y):
        if y_data[i].ndim == 1:
            y_data[i] = y_data[i].reshape(-1,1)  # reshape (n,) to (n,1)

    if ('colors' in kwargs):
        colors = [color_dict[kwargs['colors'][i]] for i in range(len(kwargs['colors']))]
    else: # use regular color cycle
        colors=[matplotlib.colors.to_rgba("C{}".format(i)) for i in range(N_y)]

    # each y_data list entry can contain multiple lines, each of which is plotted in the same color (unless 'random_colors' is turned on)
    colors = [np.tile(colors[i],(y_data[i].shape[1],1)) for i in range(N_y)]

    if 'random_colors' in kwargs:
        random_colors = kwargs['random_colors']

        if 'random_colors_seed' in kwargs:
            random_colors_seed = kwargs['random_colors_seed']
            rng = RandomState(random_colors_seed)
        else:
            rng = RandomState()

        if len(random_colors) == 1: # if only one random color tuple is given, randomize over y_data list entries
            cmap = fn.make_cmap([color_dict.get(key) for key in random_colors[0]], position=np.linspace(0,1,num=len(random_colors[0])).tolist())  # create colormap to draw colors from
            color_count=0  # stores number of colors that is required
            for i in range(N_y):
                    color_count += y_data[i].shape[1]
            idx = np.linspace(0,1,num=color_count)
            rng.shuffle(idx)
            k = 0
            for i in range(N_y):
                colors[i] = cmap(idx[k:k+y_data[i].shape[1]])  # read out required number of colors from generated random idx array
                k += y_data[i].shape[1]
        else:  # each 'random_colors' list entries is used for the set of lines in the corresponding 'y_data' list entry
            if len(random_colors) != N_y:  # if too few colormaps given, take last given colormap for remaining data entries
                random_colors.extend([random_colors[-1]]*(N_y-len(random_colors)))
            for i in range(N_y):  #
                cmap = fn.make_cmap([color_dict.get(key) for key in random_colors[i]], position=np.linspace(0,1,num=len(random_colors[i])).tolist())  # create colormap to draw colors from
                idx = np.linspace(0,1,num=y_data[i].shape[1])
                rng.shuffle(idx)
                colors[i] = cmap(idx)


    if ('alphas' in kwargs):
        alphas = kwargs['alphas']
        if len(alphas) != N_y:
            alphas = [alphas[0] for i in range(N_y)]  # instead of taking first
    else: # plot all lines as non-transparent
        alphas = [1 for i in range(N_y)]

    if ('clip_on' in kwargs):
        clip_on = kwargs['clip_on']
    else:
        clip_on = False

    if ('labels' in kwargs):
        labels = kwargs['labels']
    else:
        labels = str(np.arange(N_y))

    if ('linestyles' in kwargs):
        linestyles = [ [kwargs['linestyles'][i]] * y_data[i].shape[1] for i in range(N_y)]
    else:
        linestyles = [ ['-']                     * y_data[i].shape[1] for i in range(N_y)]

    if ('vary_linestyles' in kwargs):  # when multiple lines are plotted per y_data entry, vary their style to better distinguish them
        if kwargs['vary_linestyles']:
            for i in range(N_y):
                linestyles[i] = [(0,())] + [(0,(j+1,1)) for j in range(y_data[i].shape[1]-1) ]

    if ('linewidth' in kwargs):
        linewidth = kwargs['linewidth']
    else:
        linewidth = plt.rcParams['lines.linewidth']


    if 'figsize' in kwargs:
        fig_size = np.array(kwargs['figsize'])/2.54  # figsize in cm
    else:
        fig_size = plt.rcParams["figure.figsize"]
    plt.figure(figsize=fig_size, dpi=300)

    # adds plot points
    scatter = False
    if ('scatter' in kwargs):
        if (kwargs['scatter']):
            scatter = True
            markertype = 'o'
            markersize = (linewidth*3)**2

    for i in range(N_y):
        for j in range(y_data[i].shape[1]):
                plt.plot(x_data[i], y_data[i][:,j], linewidth=linewidth, linestyle=linestyles[i][j], color=colors[i][j,:], label=labels[i], alpha=alphas[i], clip_on=clip_on)
                if scatter:
                    plt.scatter(x_data[i], y_data[i][:,j], color=white, edgecolors=colors[i][j,:], linewidths=linewidth*0.75, marker=markertype, s=markersize, zorder=9, clip_on=False)

    if ('y_min' in kwargs):
        y_min = kwargs['y_min']
    else:
        y_min = np.min([np.min(ele) for ele in y_data]) - 0.1*abs(np.min([np.min(ele) for ele in y_data]))
    if ('y_max' in kwargs):
        y_max = kwargs['y_max']
    else:
        y_max = 1.1*np.max([np.max(ele) for ele in y_data])
        y_max = np.max([np.max(ele) for ele in y_data]) + 0.1*abs(np.max([np.max(ele) for ele in y_data]))

    if ('x_min' in kwargs):
        x_min = kwargs['x_min']
    else:
        x_min = np.min([np.min(ele) for ele in x_data])
    if ('x_max' in kwargs):
        x_max = kwargs['x_max']
    else:
        x_max = np.max([np.max(ele) for ele in x_data])

    plt.gca().set_xlim([x_min, x_max])
    plt.gca().set_ylim([y_min, y_max])

    if ('vline' in kwargs):
        vline = kwargs['vline']
        vcolor = gray
        if ('vcolor' in kwargs):
            vcolor = kwargs['vcolor']
        plt.vlines(vline, y_min, y_max, color = vcolor, linestyle='-', linewidth=linewidth_axes, clip_on=clip_on)

    if ('hline' in kwargs):
        hline = kwargs['hline']
        hcolor = gray
        if ('hcolor' in kwargs):
            hcolor = kwargs['hcolor']
        plt.hlines(hline, x_min, x_max, color = hcolor, linestyle='-', linewidth=linewidth_axes, clip_on=clip_on)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if ('spine_visibility' in kwargs):
        spine_visibility = kwargs['spine_visibility']
        plt.gca().spines['bottom'].set_visible(spine_visibility[0])
        plt.gca().spines['left'].set_visible(spine_visibility[1])
        plt.gca().spines['top'].set_visible(spine_visibility[2])
        plt.gca().spines['right'].set_visible(spine_visibility[3])

    spine_offset = 0.05
    if ('spine_offset' in kwargs):
        if not kwargs['spine_offset']:
            spine_offset = 0
    plt.gca().spines['left'].set_position(('axes', -spine_offset))  # shift axes for better visibility of origin
    plt.gca().spines['bottom'].set_position(('axes', -spine_offset))
    plt.gca().spines['right'].set_position(('axes', +spine_offset))
    plt.gca().spines['top'].set_position(('axes', +spine_offset))

    plt.ticklabel_format(style='sci', axis='x', scilimits=(-2,3))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,3))

    #legend
    if ('legend' in kwargs):
            if (kwargs['legend']):
                plt.gca().legend(bbox_to_anchor=(1.1, 1))

    # title & labels
    if ('title' in kwargs):
            plt.gca().set_title(kwargs['title'])
    if ('xlabel' in kwargs):
            plt.gca().set_xlabel(kwargs['xlabel'])
    if ('ylabel' in kwargs):
            plt.gca().set_ylabel(kwargs['ylabel'])

    if ('xticks' in kwargs):
        plt.gca().set_xticks(kwargs['xticks'])
    if ('xticklabels' in kwargs):
        plt.gca().set_xticklabels(kwargs['xticklabels'])
    if ('yticks' in kwargs):
        plt.gca().set_yticks(kwargs['yticks'])
    if ('yticklabels' in kwargs):
        plt.gca().set_yticklabels(kwargs['yticklabels'])

    if ('xaxis_pos' in kwargs):
        plt.gca().spines['bottom'].set_position(('data', kwargs['xaxis_pos']))  #shift x-axis
        plt.gca().spines['bottom'].set_visible(False)
        plt.hlines(kwargs['xaxis_pos'],x_min, np.maximum(x_max, np.max(plt.gca().get_xticks())), color='black', linestyle='-', linewidth=linewidth_axes, alpha=1)  # draw new line for x-axis
        plt.gca().xaxis.set_label_coords(0.5, -0.02)

    # save and close fig
    # if ('show_plot' in kwargs) and kwargs['show_plot'] == 1:
    #     plt.show(block=False)
    # elif ('save_path' in kwargs):
    #     plt.savefig(kwargs['save_path'], bbox_inches='tight', dpi=300)
    #     plt.close()
    # else:
    #     plt.close()


###############################################################################
# A general scatter plot
def scatter_plot(x_data, y_data, **kwargs):

    N_x = len(x_data)
    N_y = len(y_data)

    for i in np.arange(N_y):
        if y_data[i].ndim == 1:
            y_data[i] = y_data[i].reshape(-1,1)  # reshape (n,) to (n,1)

    if ('colors' in kwargs):
        colors = [color_dict[kwargs['colors'][i]] for i in range(len(kwargs['colors']))]
    else: # use regular color cycle
        colors=[matplotlib.colors.to_rgba("C{}".format(i)) for i in range(N_y)]

    # each y_data list entry can contain multiple lines, each of which is plotted in the same color
    colors = [np.tile(colors[i],(y_data[i].shape[1],1)) for i in range(N_y)]


    if ('clip_on' in kwargs):
        clip_on = kwargs['clip_on']
    else:
        clip_on = False

    if ('labels' in kwargs):
        labels = kwargs['labels']
    else:
        labels = str(np.arange(N_y))

    if ('linestyles' in kwargs):
        linestyles = [ [kwargs['linestyles'][i]] * y_data[i].shape[1] for i in range(N_y)]
    else:
        linestyles = [ ['-']                     * y_data[i].shape[1] for i in range(N_y)]

    if ('vary_linestyles' in kwargs):  # when multiple lines are plotted per y_data entry, vary their style to better distinguish them
        if kwargs['vary_linestyles']:
            for i in range(N_y):
                linestyles[i] = [(0,())] + [(0,(j+1,1)) for j in range(y_data[i].shape[1]-1) ]

    if ('linewidth' in kwargs):
        linewidth = kwargs['linewidth']
    else:
        linewidth = plt.rcParams['lines.linewidth']


    if ('size' in kwargs):
        size = kwargs['size']
    else:
        size = plt.rcParams['lines.markersize']

    plt.figure()
    # fit gaussians to scatter data
    if ('fit_gaussian' in kwargs):
        if kwargs['fit_gaussian']:
            def gaussian(x, amp, mean, sigma):
                return amp*np.exp(-(x - mean)**2/(2*sigma**2))

            for i in range(N_y):
                for j in range(y_data[i].shape[1]):
                    amp_0 = np.max(y_data[i])
                    if amp_0 != np.min(y_data[i]):
                        mean_0 = np.mean(x_data[i])
                        std_0 = np.std(x_data[i])/2
                        popt, pcov = curve_fit(f=gaussian, xdata=x_data[i], ydata=y_data[i][:,j], p0=[amp_0, mean_0, std_0], bounds=((-abs(amp_0), -np.inf, -np.inf), (abs(amp_0), np.inf, np.inf)) )
                        x_min = np.min(x_data[i])
                        x_max = np.max(x_data[i])
                        x_samples = np.linspace(x_min, x_max, num = 100)
                        plt.plot(x_samples, gaussian(x_samples, *popt), linewidth = linewidth, linestyle=linestyles[i][j], color=colors[i][j,:], clip_on=clip_on, alpha=0.5)
                        labels[i] += r', $\sigma$='+ str(np.round(popt[2],2)) + r'$\pm$' + str(np.round(np.sqrt(np.diag(pcov))[2],2))

    for i in range(N_y):
        for j in range(y_data[i].shape[1]):
            if N_x>1:
                plt.scatter(x_data[i], y_data[i][:,j], s=size, c=[colors[i][j,:]], label=labels[i], clip_on=clip_on)
            else:
                plt.scatter(x_data[0], y_data[i][:,j], s=size, c=[colors[i][j,:]], label=labels[i], clip_on=clip_on)

    if ('y_min' in kwargs):
        y_min = kwargs['y_min']
    else:
        y_min = 1.1*np.min([np.min(ele) for ele in y_data])  #tbd: wrong for positive min value!(?)
    if ('y_max' in kwargs):
        y_max = kwargs['y_max']
    else:
        y_max = 1.1*np.max([np.max(ele) for ele in y_data])

    if ('x_min' in kwargs):
        x_min = kwargs['x_min']
    else:
        x_min = 1.0*np.min([np.min(ele) for ele in x_data])
    if ('x_max' in kwargs):
        x_max = kwargs['x_max']
    else:
        x_max = 1.0*np.max([np.max(ele) for ele in x_data])

    if ('vline' in kwargs):
        vline = kwargs['vline']
        plt.vlines(vline, y_min, y_max, color = gray, linestyle='-', linewidth=linewidth_axes, clip_on=clip_on)

    if ('hline' in kwargs):
        hline = kwargs['hline']
        plt.hlines(hline, x_min, x_max, color = gray, linestyle='-', linewidth=linewidth_axes, clip_on=clip_on)

    plt.gca().set_ylim([y_min, y_max])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.gca().spines['left'].set_position(('axes', -0.05))  # shift axes for better visibility of origin
    plt.gca().spines['bottom'].set_position(('axes', -0.05))

    plt.ticklabel_format(style='sci', axis='x', scilimits=(-2,3))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,3))

    #legend
    if ('legend' in kwargs):
            if (kwargs['legend']):
                plt.gca().legend(bbox_to_anchor=(1.1, 1))

    # title & labels
    if ('title' in kwargs):
            plt.gca().set_title(kwargs['title'])
    if ('xlabel' in kwargs):
            plt.gca().set_xlabel(kwargs['xlabel'])
    if ('ylabel' in kwargs):
            plt.gca().set_ylabel(kwargs['ylabel'])

    if ('xticks' in kwargs):
        plt.gca().set_xticks(kwargs['xticks'])
    if ('xticklabels' in kwargs):
        plt.gca().set_xticklabels(kwargs['xticklabels'])
    if ('yticks' in kwargs):
        plt.gca().set_yticks(kwargs['yticks'])
    if ('yticklabels' in kwargs):
        plt.gca().set_yticklabels(kwargs['yticklabels'])

    # save and close fig
    # if ('show_plot' in kwargs) and kwargs['show_plot'] == 1:
    #     plt.show(block=False)
    # elif ('save_path' in kwargs):
    #     plt.savefig(kwargs['save_path'], bbox_inches='tight', dpi=300)
    #     plt.close()
    # else:
    #     plt.close()
