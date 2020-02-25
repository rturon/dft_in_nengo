import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_2d(nf_data, time_points, colorbar=False, title=None, save=False):
    '''
        Parameters
        ----------
        nf_data : array_like
        time_points : list of ints
        colorbar : bool, optional
        title : NoneType or str, optional
        save : False or str, optional
            Filepath if not False
    '''
    plt.figure(figsize=(11.5, 9))
    # make title
    if title is not None:
        plt.suptitle(title)
    # plot 15 points in time
    for i, time_point in enumerate(time_points):
        plt.subplot(6, 6, i+1)
        plt.imshow(nf_data[time_point].reshape(50, 50), cmap='jet')
        plt.title('%i ms' % (time_point*10))
        plt.xticks([])
        plt.yticks([])
        if colorbar:
            plt.colorbar()
    if title is not None:
        plt.subplots_adjust(top=0.93, wspace=0.26, hspace=0.23)
    else:
        plt.subplots_adjust(top=0.97, wspace=0.26, hspace=0.23)
    # save image
    if save:
        filedir = save.rsplit('/', 1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)
    # plt.show()


def plot_1d(nf_data, time_points, title=None, save=False):
    plt.figure(figsize=(13, 10))
    min = np.min(nf_data)
    max = np.max(nf_data)
    dif = max-min
    # make title
    if title is not None:
        plt.suptitle(title)

    for i, tp in enumerate(time_points):
        plt.subplot(6, 6, i+1)
        plt.title('%i ms' % (tp*10))
        plt.plot(nf_data[tp])
        plt.ylim(min-0.1*dif, max+0.1*dif)

#     plt.tight_layout(rect=(0,0,1,0.96))
    plt.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)
    # save image
    if save:
        filedir = save.rsplit('/', 1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)
    # plt.show()


def plot_0d(sim, probes, title=None, save=False):

    # change the 10 if save_every is not 0.01
    x = np.arange(0, sim.data[probes['Target Red Memory']].shape[0], 1)*10

    plot_0d = ['Target Red Memory', 'Target Blue Memory', 'Target Cyan Memory',
               'Target Green Memory', 'Target Orange Memory ', 'Target Red Production',
               'Target Blue Production', 'Target Cyan Production', 'Target Green Production',
               'Target Orange Production', 'To the left of Memory', 'To the Right of Memory',
               'Above Memory', 'Below Memory', 'Empty', 'To the left of Production',
               'To the Right of Production', 'Above Production', 'Below Production', 'Empty',
               'Reference Red Memory', 'Reference Blue Memory', 'Reference Cyan Memory',
               'Reference Green Memory', 'Reference Orange Memory',
               'Reference Red Production', 'Reference Blue Production',
               'Reference Cyan Production', 'Reference Green Production',
               'Reference Orange Production']

    plt.figure(figsize=(15, 18))
    if title is not None:
        plt.suptitle('Memory and Production Nodes')
    for i, name in enumerate(plot_0d):
        if name == 'Empty':
            continue
        plt.subplot(6, 5, i+1)
        plt.plot(x, sim.data[probes[name]])
        plt.title(name)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    if save:
        filedir = save.rsplit('/', 1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)


def plot_probes(sim, probes, save=False):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_points = np.linspace(
        0, sim.data[probes["Indeterminent "]].shape[0] - 1, 36, dtype=int
    )

    if save:
        filename = "_Spatial_Scene_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Indeterminent "]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Reference_Field_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Reference"]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Target_Field_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Target"]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Relational_Field_%s.png" % timestamp
        plot_2d(
            sim.data[probes["Object-centered "]],
            time_points,
            colorbar=True,
            title=None,
            save=save + filename,
        )

        filename = "_Colour_%s.png" % timestamp
        plot_1d(
            sim.data[probes["Colour"]],
            time_points,
            title=None,
            save=save + filename,
        )

        filename = "_Memory_and_Production_Nodes_%s.png" % timestamp
        plot_0d(sim, probes, title=None, save=save + filename)

        plt.close("all")

    else:
        plot_2d(
            sim.data[probes["Indeterminent "]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_2d(
            sim.data[probes["Reference"]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_2d(
            sim.data[probes["Target"]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_2d(
            sim.data[probes["Object-centered "]],
            time_points,
            colorbar=True,
            title=None,
            save=save,
        )

        plot_1d(sim.data[probes["Colour"]], time_points, title=None, save=save)

        plot_0d(sim, probes, title=None, save=False)

        plt.show()

###############################################################################
#############                                                   ###############
#############        same plotting functions for cedar          ###############
#############                                                   ###############
###############################################################################

def plot_2d_cedar(nf_data, time_points, colorbar=False, title=None, save=False):
    '''
        Parameters
        ----------
        nf_data : array_like
        time_points : list of ints
        colorbar : bool, optional
        title : NoneType or str, optional
        save : False or str, optional
            Filepath if not False
    '''
    times = nf_data.iloc[time_points, 0]
    data = nf_data.iloc[time_points, 1:]

    plt.figure(figsize=(11.5, 9))
    # make title
    if title is not None:
        plt.suptitle(title)
    # plot 15 points in time
    for i, time_point in enumerate(time_points):
        plt.subplot(6, 6, i+1)
        plt.imshow(np.array(data.iloc[i]).reshape(50,50, order='F'), cmap='jet')
        plt.title(times.iloc[i])
        plt.xticks([])
        plt.yticks([])
        if colorbar:
            plt.colorbar()

    if title is not None:
        plt.subplots_adjust(top=0.93, wspace=0.26, hspace=0.23)
    else:
        plt.subplots_adjust(top=0.97, wspace=0.26, hspace=0.23)
    # save image
    if save:
        filedir = save.rsplit('/', 1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)
    # plt.show()


def plot_1d_cedar(nf_data, time_points, title=None, save=False):
    times = nf_data.iloc[time_points, 0]
    data = nf_data.iloc[time_points, 1:]
    min = np.min(np.min(nf_data.iloc[:, 1:]))
    max = np.max(np.max(nf_data.iloc[:, 1:]))
    dif = max-min
    
    plt.figure(figsize=(13, 10))

    # make title
    if title is not None:
        plt.suptitle(title)

    for i, tp in enumerate(time_points):
        plt.subplot(6, 6, i+1)
        plt.title(times.iloc[i])
        plt.plot(data.iloc[i])
        plt.ylim(min-0.1*dif, max+0.1*dif)

#     plt.tight_layout(rect=(0,0,1,0.96))
    plt.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)
    # save image
    if save:
        filedir = save.rsplit('/', 1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)
    # plt.show()


def plot_0d_cedar(cedar_data, title=None, save=False, interval='all'):

    plot_0d = ['Target Red Memory', 'Target Blue Memory', 'Target Cyan Memory',
               'Target Green Memory', 'Target Orange Memory ', 'Target Red Production',
               'Target Blue Production', 'Target Cyan Production', 'Target Green Production',
               'Target Orange Production', 'To the left of Memory', 'To the Right of Memory',
               'Above Memory', 'Below Memory', 'Empty', 'To the left of Production',
               'To the Right of Production', 'Above Production', 'Below Production', 'Empty',
               'Reference Red Memory', 'Reference Blue Memory', 'Reference Cyan Memory',
               'Reference Green Memory', 'Reference Orange Memory',
               'Reference Red Production', 'Reference Blue Production',
               'Reference Cyan Production', 'Reference Green Production',
               'Reference Orange Production']

    plt.figure(figsize=(15, 18))
    if title is not None:
        plt.suptitle('Memory and Production Nodes')
    for i, name in enumerate(plot_0d):
        if name == 'Empty':
            continue
        plt.subplot(6, 5, i+1)
        x = cedar_data[name][0]
        x = [float(t.rsplit(' ', 1)[0]) for t in x]
        if interval != 'all':
            plt.plot(x[interval[0]:interval[1]], 
                     cedar_data[name][1][interval[0]:interval[1]])
        else:
            plt.plot(x, cedar_data[name][1])
        plt.title(name)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    if save:
        filedir = save.rsplit('/', 1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)

def prepare_data_for_plotting(pd_data):
    # TODO inhere: 1. remove text rows
    #              2. create tps
    # print('Number of datapoints:', pd_data.shape[0])
    # remove text rows from data
    rows_to_drop = pd_data[pd_data[0] == 'Mat'].index
    pd_data.drop(rows_to_drop, inplace=True)
    pd_data[1] = pd_data[1].astype(float)
    # print('Number of datapoints left:', pd_data.shape[0] )
    # create time points
    rows_to_drop = list(rows_to_drop)
    rows_to_drop.append(pd_data.shape[0]+len(rows_to_drop))
    minmax = [[rows_to_drop[i]-i, rows_to_drop[i+1]-i-2] for i in range(len(rows_to_drop)-1)]
    # print(rows_to_drop)
    minmax = [[0, rows_to_drop[0]-1]] + minmax
    # print(minmax)
    tps = []
    for m in minmax:
        tps.append(np.linspace(m[0], m[1], 36, dtype=int))

    return tps


def plot_probes_cedar(cedar_data, save=False):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save:
        tps = prepare_data_for_plotting(cedar_data['Indeterminent '])
        for i,time_points in enumerate(tps):
            filename = "_Spatial_Scene_%s_%i.png" % (timestamp, i)
            plot_2d_cedar(
                cedar_data["Indeterminent "],
                time_points,
                colorbar=True,
                title=None,
                save=save + filename,
            )
        plt.close("all")

        tps = prepare_data_for_plotting(cedar_data['Reference'])
        for i, time_points in enumerate(tps):
            filename = "_Reference_Field_%s_%i.png" % (timestamp, i)
            plot_2d_cedar(
                cedar_data["Reference"],
                time_points,
                colorbar=True,
                title=None,
                save=save + filename,
            )
        plt.close("all")    

        tps = prepare_data_for_plotting(cedar_data['Target'])
        for i, time_points in enumerate(tps):
            filename = "_Target_Field_%s_%i.png" % (timestamp, i)
            plot_2d_cedar(
                cedar_data["Target"],
                time_points,
                colorbar=True,
                title=None,
                save=save + filename,
            )
        plt.close("all")

        tps = prepare_data_for_plotting(cedar_data['Object-centered '])
        for i, time_points in enumerate(tps):
            filename = "_Relational_Field_%s_%i.png" % (timestamp,i)
            plot_2d_cedar(
                cedar_data["Object-centered "],
                time_points,
                colorbar=True,
                title=None,
                save=save + filename,
            )
        plt.close("all")

        tps = prepare_data_for_plotting(cedar_data['Colour'])
        for i, time_points in enumerate(tps):
            filename = "_Colour_%s_%i.png" % (timestamp, i)
            plot_1d_cedar(
                cedar_data["Colour"],
                time_points,
                title=None,
                save=save + filename,
            )
        plt.close("all")
        
        # plot the 0-dimensional neural fields
        times = cedar_data['Target Red Memory'][0]
        times = [float(t.split(' ')[0]) for t in times if not t == 'Mat']
        int_ends = [i for i in range(len(times)-1) if times[i] > 1 and times[i+1] < 1]
        int_ends.append(len(times))
        intervals = [[int_ends[i]+1, int_ends[i+1]+1] for i in range(len(int_ends)-1)]
        intervals = [[0, int_ends[0]+1]] + intervals
        for i, interval in enumerate(intervals):
            filename = "_Memory_and_Production_Nodes_%s_%i.png" % (timestamp, i)
            plot_0d_cedar(cedar_data, title=None, save=save + filename, interval=interval)

        plt.close("all")

    else:
        tps = prepare_data_for_plotting(cedar_data['Indeterminent '])
        for i,time_points in enumerate(tps):
            plot_2d_cedar(
                cedar_data["Indeterminent "],
                time_points,
                colorbar=True,
                title=None,
                save=save,
            )

        
        tps = prepare_data_for_plotting(cedar_data['Reference'])
        for i, time_points in enumerate(tps):
            plot_2d_cedar(
                cedar_data["Reference"],
                time_points,
                colorbar=True,
                title=None,
                save=save,
            )

        
        tps = prepare_data_for_plotting(cedar_data['Target'])
        for i, time_points in enumerate(tps):
            plot_2d_cedar(
                cedar_data["Target"],
                time_points,
                colorbar=True,
                title=None,
                save=save,
            )

        
        tps = prepare_data_for_plotting(cedar_data['Object-centered '])
        for i, time_points in enumerate(tps):
            plot_2d_cedar(
                cedar_data["Object-centered "],
                time_points,
                colorbar=True,
                title=None,
                save=save,
            )

        
        tps = prepare_data_for_plotting(cedar_data['Colour'])
        for i, time_points in enumerate(tps):
            plot_1d_cedar(
                cedar_data["Colour"],
                time_points,
                title=None,
                save=save,
            )

        # need to get slices of each trial of the rest of the data
        times = cedar_data['Target Red Memory'][0]
        times = [float(t.split(' ')[0]) for t in times if not t == 'Mat']
        int_ends = [i for i in range(len(times)-1) if times[i] > 1 and times[i+1] < 1]
        int_ends.append(len(times))
        intervals = [[int_ends[i]+1, int_ends[i+1]+1] for i in range(len(int_ends)-1)]
        intervals = [[0, int_ends[0]+1]] + intervals
        for interval in intervals:
            plot_0d_cedar(cedar_data, title=None, save=False, interval=interval)

        plt.show()