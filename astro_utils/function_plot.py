import numpy as np
from matplotlib import pyplot as plt


# Density Plot

def density_plot(fig, 
                 xdisc=[], ydisc=[], xdiscfit = [], ydisc_fit=[], ydisc_model =[], discfitmin=[], discfitmax = [],
                 xbul=[], ybul=[], xbulfit = [], ybul_fit=[], ybul_model=[], bulfitmin=[], bulfitmax = [],
                 xidisc=[], yidisc=[],
                 psize = 3):        
    ax = fig.add_subplot(1,1,1)
    # Fits de las distintas componentes       
    ax.plot(xdisc, ydisc_model, '--b')
    ax.plot(xbul, ybul_model, '--r')
    
    # Definir zona fiteada y zona no fiteada 
    arplot_d_lower = np.where(xdisc < discfitmin)
    arplot_d_upper = np.where(xdisc > discfitmax)
    arplot_d = np.concatenate((arplot_d_lower, arplot_d_upper), axis=None)
    
    arplot_b_lower = np.where(xbul < bulfitmin)
    arplot_b_upper = np.where(xbul > bulfitmax)
    arplot_b = np.concatenate((arplot_b_lower, arplot_b_upper), axis=None)

    rplot_d = xdisc[arplot_d]
    rhoplot_d = ydisc[arplot_d]
    rplot_b = xbul[arplot_b]
    rhoplot_b = ybul[arplot_b]
    
    # Puntos no fiteados
    ax.plot(rplot_d, rhoplot_d,'.b',marker = 'd',markersize = psize, markerfacecolor = 'white')  
    ax.plot(rplot_b, rhoplot_b,'.r', marker = 'v',markersize = psize, markerfacecolor = 'white')
    
    # Plot del inner disc
    ax.plot(xidisc, yidisc,'.g', marker = 'v',markersize = psize)

    # Puntos fiteados
    ax.plot(xdiscfit, ydisc_fit, '.b',marker = 'd',markersize = psize)
    ax.plot(xbulfit, ybul_fit, '.r',marker = 'd',markersize = psize)
    ax.set_yscale('log')     
    ax.set_xlabel('r (kpc)')
    ax.set_ylabel('$\Sigma$')
    

##########  Galaxy Scatter Plot  #########

def scatter_plot(fig, x,y, scale, psize = 1, xlabel = None, ylabel = None, 
                 title = None, cbarlabel = None, percentcolorbar = []):
    ax = fig.add_subplot(1,1,1)
    if len(percentcolorbar)==2: 
        splt = ax.scatter(x,y, c=scale, cmap = 'jet', s = psize, marker = 'o',
                          vmin=np.percentile(scale, percentcolorbar[0]), 
                          vmax=np.percentile(scale, percentcolorbar[1]))
    else:
        splt = ax.scatter(x,y, c=scale, cmap = 'jet', s = psize, marker = 'o')
    
    
    cbar = plt.colorbar(splt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)    
    cbar.set_label(cbarlabel, rotation=270, labelpad=20)
    ax.set_title(title)
        
        
##########  Galaxy Double colour Scatter Plot  #########

def double_scatter_plot(fig, xd, yd, xb, yb, scale, psize = 1.5, xlabel=None, ylabel = None, cbarlabel = ' ', percentcolorbar = []):
    # Representar el disco
    ax = fig.add_subplot(1,1,1)
    if len(percentcolorbar)==2: 
        sd = ax.scatter(xd,yd, c=scale, cmap = 'winter', s=psize, marker = 's', label = 'Disco', 
                        vmin=np.percentile(scale, percentcolorbar[0]),
                        vmax=np.percentile(scale, percentcolorbar[1]))        
    
        sb = ax.scatter(xb,yb, c=scale, cmap = 'autumn', s=psize, marker = 's', label = 'Bulge', 
                        vmin=np.percentile(scale, percentcolorbar[0]),
                        vmax=np.percentile(scale, percentcolorbar[1]))        
    else:
        sd = ax.scatter(xd,yd, c=scale, cmap = 'winter', s=psize, marker = 's', label = 'Disco')            
    
        sb = ax.scatter(xb,yb, c=scale, cmap = 'autumn', s=psize, marker = 's', label = 'Bulge')
    cbar = plt.colorbar(sb, pad =0)
    cbard = plt.colorbar(sd, pad = 0.05)    
    cbar.ax.yaxis.set_ticks_position('left')    
    plt.setp(cbard.ax.get_yticklabels(), visible=False)    
    cbar.set_label('Bulge', labelpad=0, y=1.05, rotation=0)
    cbard.set_label('Disc', labelpad=0.5, y=1.05, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)      
    # Conseguir otro sets de axis para la cbar para poder poner dos labels
    pos = cbar.ax.get_position()
    ax1 = cbar.ax
    ax1.set_aspect('auto')
    ax2 = ax1.twinx()
    ax1.set_position(pos)
    ax2.set_position(pos)
    ax2.set_ylabel('B/T', rotation=270, labelpad=20) 
    ax2.set_yticks([]) 

    

##############  Compute Caustic Curve of Energy  #########

def plotcaustic(fig, ett,lnjzs,lnjinter,x_data_envelope, data_envelope, 
                xname = None, yname = None):   
    ax = fig.add_subplot(1,1,1)
    ax.plot(ett, lnjzs,'.k', markersize = 1)
    ax.plot(ett, lnjinter,'.')
    ax.plot(x_data_envelope, data_envelope)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

#####  Plot Histogram  #########

def plothist(fig, epi, nbin_hist, xname = None ):
    ax = fig.add_subplot(1,1,1)   
    weights = np.ones_like(epi)/float(len(epi))
    ax.hist(epi, bins = nbin_hist, weights=weights, edgecolor=('black'), facecolor = ('white'))
    ax.set_xlabel(xname)
    ax.set_ylabel("Frecuencia")

# Galaxia como scatter plot    
def galplot(fig, xhalo=[], yhalo=[], xhaloex=[], yhaloex=[], 
            xcontradisco=[], ycontradisco=[], xdisco=[],
            ydisco=[], xbulge=[], ybulge=[], psize=1, 
            ropt= 0, nropt=0, xname = None, yname = None):
    
    ax = fig.add_subplot(1,1,1)   
    if ((len(xhaloex)>0)&(len(yhaloex)>0))==True:
        ax.plot(xhaloex, yhaloex,'.y',markersize = psize) 
    if ((len(xhalo)>0)&(len(yhalo)>0))==True:
        ax.plot(xhalo, yhalo,'.y',markersize = psize)
    if ((len(xcontradisco)>0)&(len(ycontradisco)>0))==True:
        ax.plot(xcontradisco, ycontradisco,'.k', markersize = psize)
    if ((len(xdisco)>0)&(len(ydisco)>0))==True:
        ax.plot(xdisco, ydisco,'.b',markersize = psize)
    if ((len(xbulge)>0)&(len(ybulge)>0))==True:
        ax.plot(xbulge, ybulge,'.r',markersize = psize)

    if ropt>0:
        ax.set_ylim([-nropt*ropt,nropt*ropt])   
        ax.set_xlim([-nropt*ropt,nropt*ropt])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)