import numpy as np
import matplotlib.pyplot as plt
import os

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
        plt.subplot(6,6,i+1)
        plt.imshow(nf_data[time_point].reshape(50,50), cmap='jet')
        plt.title('%i ms' %(time_point*10))
        plt.xticks([])
        plt.yticks([])
        if colorbar:
            plt.colorbar()
    if title is not None:        
        plt.subplots_adjust(top=0.93,wspace=0.26, hspace=0.23)
    else:
        plt.subplots_adjust(top=0.97,wspace=0.26, hspace=0.23)
    # save image    
    if save:
        filedir = save.rsplit('/',1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)
    # plt.show()
    
def plot_1d(nf_data, time_points, title=None, save=False):
    plt.figure(figsize=(13,10))
    min = np.min(nf_data)
    max = np.max(nf_data)
    dif = max-min
    # make title
    if title is not None: 
        plt.suptitle(title)
    
    for i, tp in enumerate(time_points):
        plt.subplot(6,6,i+1)
        plt.title('%i ms' %(tp*10))
        plt.plot(nf_data[tp])
        plt.ylim(min-0.1*dif, max+0.1*dif)
        
#     plt.tight_layout(rect=(0,0,1,0.96))
    plt.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)
    # save image
    if save:
        filedir = save.rsplit('/',1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)
    # plt.show()

def plot_0d(sim, probes, title=None, save=False):

    x = np.arange(0,sim.data[probes['Target Red Memory']].shape[0],1)*10 # change the 10 if 
                                                                         # save_every is not 0.01

    plot_0d = ['Target Red Memory', 'Target Blue Memory', 'Target Cyan Memory', 
            'Target Green Memory', 'Target Orange Memory ', 'Target Red Production', 
            'Target Blue Production', 'Target Cyan Production', 'Target Green Production', 
            'Target Orange Production', 'To the left of Memory', 'To the Right of Memory',
            'Above Memory', 'Below Memory', 'Empty','To the left of Production', 
            'To the Right of Production', 'Above Production', 'Below Production', 'Empty',
            'Reference Red Memory', 'Reference Blue Memory', 'Reference Cyan Memory', 
            'Reference Green Memory', 'Reference Orange Memory',
            'Reference Red Production', 'Reference Blue Production', 
            'Reference Cyan Production', 'Reference Green Production', 
            'Reference Orange Production']

    plt.figure(figsize=(15,18))
    if title is not None:
        plt.suptitle('Memory and Production Nodes')
    for i, name in enumerate(plot_0d):
        if name == 'Empty':
            continue
        plt.subplot(6,5,i+1)
        plt.plot(x, sim.data[probes[name]])
        plt.title(name)
        
    plt.tight_layout(rect=(0,0,1,0.97))

    if save:
        filedir = save.rsplit('/',1)[0]
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        plt.savefig(save)