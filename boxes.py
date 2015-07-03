import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', serif='computer modern roman')
matplotlib.rc('font', **{'sans-serif': 'computer modern sans serif'})
import numpy as np
import pylab as plt
import os
import sys
from astrometry.util.plotutils import *
from astrometry.util.util import *

class MyPlotSequence(PlotSequence):
    def savefig(self, axline=True, yrange=None, yticks=[0, 100],
                yticklabels=None):
        plt.axis([1., 24., -3, 109])
        if axline:
            plt.axhline(0, color='k', alpha=0.5)
        if yrange:
            plt.ylim(yrange)
        plt.xticks([])
        kwa = dict(fontsize=20)
        if yticklabels is not None:
            plt.yticks(yticks, yticklabels, **kwa)
        else:
            plt.yticks(yticks, **kwa)
        super(MyPlotSequence, self).savefig()

ps = MyPlotSequence('boxes')
ps.suffixes=['png','pdf']

def box_resample(xin, yin, xout):
    bwin = xin[1]-xin[0]
    xinlo = xin - bwin/2.
    xinhi = xin + bwin/2.
    bw = xout[1]-xout[0]
    xoutlo = xout - bw/2.
    xouthi = xout + bw/2.
    yout = np.zeros_like(xoutlo)
    for i in range(len(yin)):
        xlo = xinlo[i]
        xhi = xinhi[i]
        J = np.flatnonzero(np.logical_not(
            np.logical_or(xoutlo>xhi, xouthi<xlo)))
        bot = np.maximum(xoutlo[J], xlo)
        top = np.minimum(xouthi[J], xhi)
        yout[J] += (top - bot) * yin[i]
    yout /= bw
    xb = np.vstack((xoutlo,xouthi)).T.ravel()
    return yout, xb

def lan_resample(xin, yin, xout, L=3):
    yout = np.zeros_like(xout, np.float32)
    dxin = xin[1]-xin[0]
    # convert to pixel coords in the input array
    x = (xout - xin[0]) / dxin
    ix = np.round(x).astype(np.int32)
    dx = (x - ix).astype(np.float32)
    img = yin.reshape((1,-1)).astype(np.float32)
    if True:
        if L == 3:
            f = lanczos3_interpolate
        elif L == 5:
            f = lanczos5_interpolate
        else:
            assert(False)
        ok = f(ix, np.zeros_like(ix),
               dx, np.zeros_like(dx), [yout], [img])
    else:
        from astrometry.util.resample import _lanczos_interpolate
        _lanczos_interpolate(L, ix, np.zeros_like(ix), dx, np.zeros_like(dx),
                             [yout], [img], table=False)
    return yout

def plotboxes(xx, yy, cc, **kwa):
    bw = xx[1]-xx[0]
    plt.plot(np.vstack((xx-bw/2.,xx+bw/2.)).T.ravel(), yy.repeat(2),
             '-', color=cc, **kwa)

def plotsamples(xx, yy, cc, **kwa):
    plt.plot(np.vstack((xx,xx)), np.vstack((np.zeros_like(yy),yy)), '-',
             color=cc, **kwa)
    plt.plot(xx, yy, '.', color=cc, **kwa)

def plotgrid(xgrid, ygrid, cc, **kwa):
    kwa = kwa.copy()
    kwa.setdefault('alpha', 0.3)
    plt.plot(xgrid, lygrid, '-', color=cc, **kwa)


if __name__ == '__main__':

    W,H = 4,3
    medfigsize = (W,H)
    medspa = dict(left=0.01, right=0.99, bottom=0.01, top=0.99)

    # Leftmost column has yticks
    lmedfigsize = (W * (0.98/0.88), H)
    lmedspa = dict(left=0.11, right=0.99, bottom=0.01, top=0.99)
    
    plt.figure(num=1, figsize=lmedfigsize)
    plt.subplots_adjust(**lmedspa)

    plt.figure(num=2, figsize=medfigsize)
    plt.subplots_adjust(**medspa)

    
    N = 25
    dN = N-21

    dx = 0.2

    c1 = 'g'
    c2 = (0, 0.35, 0.5)
    c3 = 'b'
    
    xx = np.arange(N)
    
    mus  = np.array([0.35,0.65]) * (N-dN) + dN/2
    sigs = np.array([0.1,0.1]) * (N-dN)
    amps = np.array([100.,80.])

    yy = np.zeros(N)
    for a,mu,sig in zip(amps,mus,sigs):
        yy += a * np.exp(-0.5 * (xx - mu)**2/sig**2)
    #yy[len(yy)/2] = 100.
    
    # Smooth lanczos function
    xgrid = np.linspace(0., N, 500)
    lygrid = lan_resample(xx, yy, xgrid)
    l5ygrid = lan_resample(xx, yy, xgrid, L=5)

    # Sample there
    xre = np.arange(N*2) / 2. + dx
    bxre = xre - 0.25
    lxre = xre[::2]
    bxre2 = lxre
    lyre = lan_resample(xx, yy, lxre)
    l5yre = lan_resample(xx, yy, lxre, L=5)
    byre,boxre = box_resample(xx, yy, bxre)
    byre2,boxre2 = box_resample(xx, yy, bxre2)
    lregrid = lan_resample(xx, lyre, xgrid)
    l5regrid = lan_resample(xx, l5yre, xgrid)
    
    # Sample back
    bxback = xx
    bxback2 = xx
    lxback = xx
    lyback = lan_resample(lxre, lyre, lxback)
    l5yback = lan_resample(lxre, l5yre, lxback, L=5)
    byback,boxback = box_resample(bxre, byre, bxback)
    byback2,boxback2 = box_resample(bxre2, byre2, bxback2)
    lybackgrid = lan_resample(lxback, lyback, xgrid)
    l5backgrid = lan_resample(lxback, l5yback, xgrid)

    box = np.vstack((xx-0.5, xx+0.5)).T.ravel()

    oa = dict(zorder=15, lw=2, alpha=0.3, ms=8)
    og = dict(zorder=15, lw=3, alpha=0.1)
    
    # Original boxes
    # plt.figure(1)
    # plt.clf()
    # plotboxes(xx, yy, c1)
    # ps.savefig()

    # Original samples
    # plt.clf()
    # plotsamples(xx, yy, c1)
    # plotgrid(xgrid, lygrid, c1)
    # ps.savefig()
    
    # Resampled boxes
    # plt.clf()
    # plotboxes(bxre2, byre2, c2)
    # #ps.savefig()
    # # +orig
    # plotboxes(xx, yy, c1, **oa)
    # ps.savefig()
    
    # Resampled fine boxes
    # plt.clf()
    # plotboxes(bxre, byre, c2)
    # #ps.savefig()
    # # +orig
    # plotboxes(xx, yy, c1, **oa)
    # ps.savefig()

    # Resampled L3
    # plt.clf()
    # plotsamples(lxre, lyre, c2)
    # plotgrid(xgrid, lygrid, c2)
    # # +orig
    # plotsamples(xx, yy, c1, **oa)
    # plotgrid(xgrid, lygrid, c1, **og)
    # ps.savefig()

    # # Resampled L5
    # plt.clf()
    # plotsamples(lxre, l5yre, c2)
    # plotgrid(xgrid, l5ygrid, c2)
    # #ps.savefig()   
    # # +orig
    # plotsamples(xx, yy, c1, **oa)
    # plotgrid(xgrid, l5ygrid, c1, **og)
    # ps.savefig()

    # Sampled-back boxes
    plt.figure(1)
    plt.clf()
    plotboxes(bxback2, byback2, c3, zorder=20)
    #ps.savefig()
    # vs originals
    plotboxes(xx, yy, c1, **oa)
    # and resampled
    #plotboxes(bxre2, byre2, c2, **oa)
    #ps.savefig(axline=False)
    ps.savefig()
    plt.figure(2)
    
    # Sampled-back fine boxes
    plt.clf()
    plotboxes(bxback, byback, c3, zorder=20)
    #ps.savefig()
    # vs originals
    plotboxes(xx, yy, c1, **oa)
    # and resampled
    #plotboxes(bxre, byre, c2, **oa)
    #ps.savefig(axline=False)
    ps.savefig()
    
    # Sampled-back L
    for ly,ygrid,yre,ybackgrid in [
            (lyback,  lygrid,  lyre,  lybackgrid),
            (l5yback, l5ygrid, l5yre, l5backgrid)]:
        plt.clf()
        plotsamples(lxback, ly, c3, zorder=20)
        plotgrid(xgrid, ybackgrid, c3, zorder=19)
        #ps.savefig()
        # vs originals
        plotsamples(xx, yy, c1, **oa)
        plotgrid(xgrid, ygrid, c1, **og)
        #ps.savefig(axline=False)
        # and resampled
        #plotsamples(lxre, yre, c2, **oa)
        ps.savefig()

    # deltas
    for i,y in enumerate([byback2, byback, lyback, l5yback]):
        if i == 0:
            plt.figure(1)
        else:
            plt.figure(2)
        plt.clf()
        plt.plot(xx, y - yy, '.-', color=c3)
        ps.savefig(yrange=(-3.3,3.3),
                   yticks=np.arange(-3, 3.1),
                   yticklabels=['','$-2$','','$0$','','$2$',''])
    
    # Back and forth a few times...
    nn = 31
    ninterval = 3

    # Coarse boxes
    xhere = xx
    yhere = yy
    plt.figure(1)
    plt.clf()
    plt.plot(xhere, yhere, '-', color='k')
    for i in range(nn):
        f = float(i)/(nn-1)
        xthere = xhere + dx
        bythere,bxthere = box_resample(xhere, yhere, xthere)
        byhere,bxhere = box_resample(xthere, bythere, xhere)
        yhere = byhere
        if i and i % ninterval == 0:
            plt.plot(xhere, byhere, '-', color=(0, (1-f)*0.8, f),
                     alpha=0.3 + 0.7*f)
    ps.savefig()
    plt.figure(2)

    # Fine Boxes
    xhere = xx
    yhere = yy
    plt.clf()
    plt.plot(xhere, yhere, '-', color='k')
    for i in range(nn):
        f = float(i)/(nn-1)
        xthere = np.arange(N*2) / 2. + dx
        # shift the box pitch
        xthere -= 0.25
        bythere,bxthere = box_resample(xhere, yhere, xthere)
        byhere,bxhere = box_resample(xthere, bythere, xhere)
        yhere = byhere
        if i and i % ninterval == 0:
            plt.plot(xhere, byhere, '-', color=(0, (1-f)*0.8, f),
                     alpha=0.3 + 0.7*f)
    ps.savefig()

    for L in [3,5]:
        # Lanczos
        xhere = xx
        yhere = yy
        plt.clf()
        plt.plot(xhere, yhere, '-', color='k')
        for i in range(nn):
            f = float(i)/(nn-1)
            xthere = xhere + dx
            lythere = lan_resample(xhere, yhere, xthere, L=L)
            lyhere = lan_resample(xthere, lythere, xhere, L=L)
            if i and i % ninterval == 0:
                lygrid = lan_resample(xhere, lyhere, xgrid, L=L)
                plt.plot(xgrid, lygrid, '-', color=(0, (1-f)*0.8, f), alpha=0.5)
            yhere = lyhere
        ps.savefig()

    
