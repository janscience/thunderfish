import sys
import os
import warnings
import argparse
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from audioio.playaudio import PlayAudio, fade
from audioio.audiowriter import write_audio
from .eventdetection import detect_peaks, median_std_threshold
from .pulses import detect_pulses
from .version import __version__, __year__
from .dataloader import open_data


class SignalPlot:
    def __init__(self, data, samplerate, unit, filename,
                 show_channels=[], fcutoff=None, pulses=False):
        self.filename = filename
        self.samplerate = samplerate
        self.data = data
        self.channels = self.data.shape[1]
        self.unit = unit
        self.tmax = (len(self.data)-1)/self.samplerate
        self.toffset = 0.0
        self.twindow = 10.0
        if self.twindow > self.tmax:
            self.twindow = np.round(2 ** (np.floor(np.log(self.tmax) / np.log(2.0)) + 1.0))
        self.pulses = np.zeros((0, 3), dtype=np.int)
        self.fishes = []
        self.pulse_times = []
        self.pulse_gids = []
        if len(show_channels) == 0:
            self.show_channels = np.arange(self.channels)
        else:
            self.show_channels = np.array(show_channels)
        self.traces = len(self.show_channels)
        self.ymin = -1.0 * np.ones(self.traces)
        self.ymax = +1.0 * np.ones(self.traces)
        self.fmax = 100.0
        self.trace_artist = [None] * self.traces
        self.pulse_artist = []
        self.marker_artist = [None] * (self.traces + 1)
        self.ipis_artist = []
        self.ipis_labels = []
        self.figf = None
        self.axf = None
        self.pulse_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        self.help = False
        self.helptext = []

        # filter data:
        if not fcutoff is None:
            sos = butter(2, fcutoff, 'high', fs=samplerate, output='sos')
            self.data = sosfiltfilt(sos, self.data[:], 0)

        # pulse detection:
        if pulses:
            # label, group, channel, peak index, trough index
            all_pulses = np.zeros((0, 5), dtype=np.int)
            for c in range(self.channels):
                #thresh = 1*np.std(self.data[:int(2*self.samplerate),c])
                thresh = median_std_threshold(self.data[:,c], self.samplerate,
                                              thresh_fac=6.0)
                #p, t = detect_peaks(self.data[:,c], thresh)
                p, t, w, h = detect_pulses(self.data[:,c], self.samplerate,
                                           thresh,
                                           min_rel_slope_diff=0.25,
                                           min_width=0.0001,
                                           max_width=0.01,
                                           width_fac=5.0)
                # label, channel, peak, trough:
                pulses = np.hstack((np.ones((len(p), 1), dtype=np.int)*c,
                                    np.zeros((len(p), 1), dtype=np.int),
                                    np.ones((len(p), 1), dtype=np.int)*c,
                                    p[:,np.newaxis], t[:,np.newaxis]))
                all_pulses = np.vstack((all_pulses, pulses))
            self.pulses = all_pulses[np.argsort(all_pulses[:,3]),:]
            # grouping over channels:
            max_di = int(0.0002*self.samplerate)   # TODO: parameter
            l = -1
            k = 0
            while k < len(self.pulses):
                tp = self.pulses[k,3]
                tt = self.pulses[k,4]
                height = self.data[self.pulses[k,3],self.pulses[k,2]] - \
                    self.data[self.pulses[k,4],self.pulses[k,2]]
                channel_counts = np.zeros(self.channels, dtype=np.int)
                channel_counts[self.pulses[k,2]] += 1
                for c in range(1, 3*self.channels):
                    if k+c >= len(self.pulses):
                        break
                    # pulse too far away:
                    if channel_counts[self.pulses[k+c,2]] > 1 or \
                       (np.abs(self.pulses[k+c,3] - tp) > max_di and
                        np.abs(self.pulses[k+c,3] - tt) > max_di and
                        np.abs(self.pulses[k+c,4] - tp) > max_di and
                        np.abs(self.pulses[k+c,4] - tt) > max_di):
                        break
                    channel_counts[self.pulses[k+c,2]] += 1
                    height_kc = self.data[self.pulses[k+c,3],self.pulses[k+c,2]] - \
                        self.data[self.pulses[k+c,4],self.pulses[k+c,2]]
                    # heighest pulse sets time reference:
                    if height_kc > height:
                        tp = self.pulses[k+c,3]
                        tt = self.pulses[k+c,4]
                        height = height_kc
                # all pulses too small:
                if height < 0.04:    # TODO parameter
                    self.pulses[k:k+c,0] = -1
                    k += c
                    continue
                # new label:
                l += 1
                ll = l % len(self.pulse_colors)
                #if self.pulses[k,2]/self.samplerate < 0.172:
                #    print()
                #    for p in self.pulses[k:k+c]:
                #        print(f'{p[0]} {p[1]} {p[2]/self.samplerate:.4f} {p[3]/self.samplerate:.4f}')
                # remove lost pulses:
                for j in range(c):
                    if (np.abs(self.pulses[k+j,3] - tp) > max_di and
                        np.abs(self.pulses[k+j,3] - tt) > max_di and
                        np.abs(self.pulses[k+j,4] - tp) > max_di and
                        np.abs(self.pulses[k+j,4] - tt) > max_di):
                        self.pulses[k+j,0] = -1
                        channel_counts[self.pulses[k+j,2]] -= 1
                    else:
                        self.pulses[k+j,0] = ll
                        self.pulses[k+j,1] = l
                # remove remaining multiple peaks per channel:
                for dc in np.where(channel_counts > 1)[0]:
                    idx = np.where(self.pulses[k:k+c,2] == dc)[0]
                    heights = self.data[self.pulses[k:k+c,:][idx,3],dc] - \
                        self.data[self.pulses[k:k+c,:][idx,4],dc]
                    for i in range(len(idx)):
                        if i != np.argmax(heights):
                            channel_counts[self.pulses[k+idx[i],2]] -= 1
                            self.pulses[k+idx[i],0] = -1
                k += c
            self.pulses = self.pulses[self.pulses[:,0] >= 0,:]
            # clustering:
            min_dists = []
            recent = []
            k = 0
            while k < len(self.pulses):
                # select pulse group:
                j = k
                gid = self.pulses[j,1]
                for c in range(self.channels):
                    k += 1
                    if k >= len(self.pulses) or \
                       self.pulses[k,1] != gid:
                        break
                heights = np.zeros(self.channels)
                heights[self.pulses[j:k,2]] = \
                    self.data[self.pulses[j:k,3],self.pulses[j:k,2]] - \
                    self.data[self.pulses[j:k,4],self.pulses[j:k,2]]
                h_idx = np.argsort(heights)
                heights[h_idx[:-4]] = 0.0    # dist for 4 largest only
                i = np.where(self.pulses[j:k,2] == h_idx[-1])[0][0]
                t = self.pulses[j+i,3]
                if len(self.pulse_times) == 0:
                    label = len(self.pulse_times)
                    self.pulse_times.append([])
                    self.pulse_gids.append([])
                else:
                    # compute metrics of recent fishes:
                    # distance between pulses:
                    ipis = np.array([(self.pulses[j,3] - tt)/self.samplerate
                                     for ll, tt, hh in recent])
                    # relative difference in maximum height:
                    delta_h = np.array([np.abs(np.max(hh) -
                                               np.max(heights))/np.max(hh)
                                        for ll, tt, hh in recent])
                    # number of overlapping meaningful heights:
                    overlaps = np.array([np.sum((hh > 0) & (heights > 0))
                                         for ll, tt, hh in recent])
                    # absolute root mean square:
                    dists = np.array([np.sqrt(np.mean((hh - heights)**2))
                                      for ll, tt, hh in recent])
                    # check distance:
                    thresh = 0.03
                    thresh = 0.02
                    # ensure minimum IP distance:
                    dists[1/ipis > 300.0] = 2*np.max(dists)  # TODO: make parameter
                    min_dist_idx = np.argmin(dists)
                    min_dists.append(dists[min_dist_idx])
                    #print(dists[min_dist_idx])
                    if dists[min_dist_idx] < thresh and \
                       overlaps[min_dist_idx] >= 2:
                        label = recent[min_dist_idx][0]
                    else:
                        label = len(self.pulse_times)
                        self.pulse_times.append([])
                        self.pulse_gids.append([])
                self.pulses[j:k,0] = label
                self.pulse_times[label].append(t)
                self.pulse_gids[label].append(gid)
                self.fishes.append([label, self.pulses[j,3], heights])
                recent.append([label, self.pulses[j,3], heights])
                # remove old fish:
                for i, (ll, tt, hh) in enumerate(recent):
                    # TODO: make parameter:
                    if (self.pulses[j,3] - tt)/self.samplerate <= 1.0:
                        recent = recent[i:]
                        break
                # only consider the n most recent pulses of a fish:
                n = 3
                labels = np.array([ll for ll, tt, hh in recent])
                if np.sum(labels == label) > n:
                    del recent[np.where(labels == label)[0][0]]
            # pulse times to arrays:
            for k in range(len(self.pulse_times)):
                self.pulse_times[k] = np.array(self.pulse_times[k])
            # all labels:
            self.labels = np.unique(self.pulses[:,0])
            # report:
            print(f'found {len(self.pulse_times)} fish:')
            for k in range(len(self.pulse_times)):
                print(f'{k:3d}: {len(self.pulse_times[k]):5d} pulses')
            ## plot histogtram of distances:
            #plt.hist(min_dists, 100)
            #plt.show()
            ## plot features:
            """
            nn = np.array([(k, len(self.pulse_times[k]))
                           for k in range(len(self.pulse_times))])
            fig, axs = plt.subplots(5, 5, figsize=(15, 9),
                                    constrained_layout=True)
            ni = np.argsort(nn[:,1])           # largest cluster ...
            ln = np.sort(nn[ni[-axs.size:],0]) # ... sort by label
            for l, ax in zip(ln, axs.flat):
                h = np.array([hh for ll, tt, hh in self.fishes if ll == l])
                ax.plot(h.T, 'o-', ms=2, lw=0.5,
                        color=self.pulse_colors[l%len(self.pulse_colors)])
                ax.text(0.05, 0.9, f'label: {l}', transform=ax.transAxes)
            """
            
        # audio output:
        self.audio = PlayAudio()
        
        # set key bindings:
        plt.rcParams['keymap.fullscreen'] = 'f'
        plt.rcParams['keymap.pan'] = 'ctrl+m'
        plt.rcParams['keymap.quit'] = 'ctrl+w, alt+q, q'
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.xscale'] = ''
        plt.rcParams['keymap.grid'] = ''
        #plt.rcParams['keymap.all_axes'] = ''

        # the figure:
        plt.ioff()
        splts = self.traces
        if len(self.pulses) > 0:
            splts += 1
        self.fig, self.axs = plt.subplots(splts, 1, squeeze=False,
                                          figsize=(15, 9), sharex=True)
        self.axs = self.axs.flat
        if self.traces == self.channels:
            self.fig.canvas.set_window_title(self.filename)
        else:
            cs = ' c%d' % self.show_channels[0]
            self.fig.canvas.set_window_title(self.filename + ' ' + cs)
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.fig.canvas.mpl_connect('resize_event', self.resize)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        # trace plots:
        for t in range(self.traces):
            self.axs[t].set_ylabel(f'C-{self.show_channels[t]+1} [{self.unit}]')
        #for t in range(self.traces-1):
        #    self.axs[t].xaxis.set_major_formatter(plt.NullFormatter())
        if len(self.pulses) > 0:
            self.axs[-1].set_ylim(0, self.fmax)
            self.axs[-1].set_ylabel('IP freq [Hz]')
        self.axs[-1].set_xlabel('Time [s]')
        ht = self.axs[0].text(0.98, 0.05, '(ctrl+) page and arrow up, down, home, end: scroll', ha='right',
                           transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.1, '+, -, X, x: zoom time in/out', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.2, 'y, Y, v, V, ctrl+v: zoom amplitudes out/in/max/default/max per trace', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.3, 'i, I: zoom IPI frequency in/out', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.4, 'p, P: play audio (display, all)', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.5, 'f: full screen', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.6, 'w: plot waveforms into png file', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.7, 'S: save audiosegment', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.8, 'q: quit', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        ht = self.axs[0].text(0.98, 0.9, 'h: toggle this help', ha='right', transform=self.axs[0].transAxes)
        self.helptext.append(ht)
        # plot:
        for ht in self.helptext:
            ht.set_visible(self.help)
        self.update_plots()
        # feature plot:
        self.figf, self.axf = plt.subplots()
        plt.show()

    def __del(self):
        self.audio.close()

    def plot_pulses(self, axs, plot=True, tfac=1.0):
        pak = 0
        for l in self.labels:
            pulses = self.pulses[self.pulses[:,0] == l,:]
            for t in range(self.traces):
                c = self.show_channels[t]
                p = pulses[pulses[:,2] == c,3]
                if plot or pak >= len(self.pulse_artist):
                    pa, = axs[t].plot(tfac*p/self.samplerate,
                                      self.data[p,c], 'o', picker=5,
                                      color=self.pulse_colors[l%len(self.pulse_colors)])
                    if not plot:
                        self.pulse_artist.append(pa)
                else:
                    self.pulse_artist[pak].set_data(tfac*p/self.samplerate,
                                                    self.data[p,c])
                pak += 1
            if l < len(self.pulse_times):
                pt = self.pulse_times[l]/self.samplerate
                if len(pt) > 10:
                    if plot or not l in self.ipis_labels:
                        pa, = axs[-1].plot(tfac*pt[:-1], 1.0/np.diff(pt),
                                           '-o', picker=5,
                                           color=self.pulse_colors[l%len(self.pulse_colors)])
                        if not plot:
                            self.ipis_artist.append(pa)
                            self.ipis_labels.append(l)
                    else:
                        iak = self.ipis_labels.index(l)
                        self.ipis_artist[iak].set_data(tfac*pt[:-1],
                                                       1.0/np.diff(pt))

    def update_plots(self):
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        if t1 > len(self.data):
            t1 = len(self.data)
        time = np.arange(t0, t1) / self.samplerate
        for t in range(self.traces):
            c = self.show_channels[t]
            self.axs[t].set_xlim(self.toffset, self.toffset + self.twindow)
            if self.trace_artist[t] == None:
                self.trace_artist[t], = self.axs[t].plot(time, self.data[t0:t1,c])
            else:
                self.trace_artist[t].set_data(time, self.data[t0:t1,c])
            if t1 - t0 < 200:
                self.trace_artist[t].set_marker('o')
                self.trace_artist[t].set_markersize(3)
            else:
                self.trace_artist[t].set_marker('None')
            self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
        self.plot_pulses(self.axs, False)
        self.fig.canvas.draw()

    def on_pick(self, event):
        # index of pulse artist:
        pk = -1
        for k, pa in enumerate(self.pulse_artist):
            if event.artist == pa:
                pk = k
                break
        li = -1
        pi = -1
        if pk >= 0:
            # find label and pulses of pulse artist:
            ll = self.labels[pk//self.traces]
            cc = self.show_channels[pk % self.traces]
            pulses = self.pulses[self.pulses[:,0] == ll,:]
            gid = pulses[pulses[:,2] == cc,1][event.ind[0]]
            if ll in self.ipis_labels:
                li = self.ipis_labels.index(ll)
                pi = self.pulse_gids[ll].index(gid)
        else:
            ik = -1
            for k, ia in enumerate(self.ipis_artist):
                if event.artist == ia:
                    ik = k
                    break
            if ik < 0:
                return
            li = ik
            ll = self.ipis_labels[li]
            pi = event.ind[0]
            gid = self.pulse_gids[ll][pi]
        # mark pulses:
        pulses = self.pulses[self.pulses[:,0] == ll,:]
        pulses = pulses[pulses[:,1] == gid,:]
        for t in range(self.traces):
            c = self.show_channels[t]
            pt = pulses[pulses[:,2] == c,3]
            if len(pt) > 0:
                if self.marker_artist[t] is None:
                    pa, = self.axs[t].plot(pt[0]/self.samplerate,
                                           self.data[pt[0],c], 'o', ms=10,
                                           color=self.pulse_colors[ll%len(self.pulse_colors)])
                    self.marker_artist[t] = pa
                else:
                    self.marker_artist[t].set_data(pt[0]/self.samplerate,
                                                   self.data[pt[0],c])
                    self.marker_artist[t].set_color(self.pulse_colors[ll%len(self.pulse_colors)])
            elif self.marker_artist[t] is not None:
                self.marker_artist[t].set_data([], [])
        # mark ipi:
        if pi >= 0:
            pt0 = self.pulse_times[ll][pi]/self.samplerate
            pt1 = self.pulse_times[ll][pi+1]/self.samplerate
            pf = 1.0/(pt1-pt0)
            if self.marker_artist[self.traces] is None:
                pa, = self.axs[self.traces].plot(pt0, pf, 'o', ms=10,
                                                 color=self.pulse_colors[ll%len(self.pulse_colors)])
                self.marker_artist[self.traces] = pa
            else:
                self.marker_artist[self.traces].set_data(pt0, pf)
                self.marker_artist[self.traces].set_color(self.pulse_colors[ll%len(self.pulse_colors)])
        elif not self.marker_artist[self.traces] is None:
            self.marker_artist[self.traces].set_data([], [])
        self.fig.canvas.draw()
        # show features:
        heights = np.zeros(self.channels)
        heights[pulses[:,2]] = \
            self.data[pulses[:,3],pulses[:,2]] - \
            self.data[pulses[:,4],pulses[:,2]]
        self.axf.plot(heights, color=self.pulse_colors[ll%len(self.pulse_colors)])
        self.figf.canvas.draw()

    def resize(self, event):
        # print('resized', event.width, event.height)
        leftpixel = 80.0
        rightpixel = 20.0
        bottompixel = 50.0
        toppixel = 20.0
        x0 = leftpixel / event.width
        x1 = 1.0 - rightpixel / event.width
        y0 = bottompixel / event.height
        y1 = 1.0 - toppixel / event.height
        self.fig.subplots_adjust(left=x0, right=x1, bottom=y0, top=y1,
                                 hspace=0)

    def keypress(self, event):
        # print('pressed', event.key)
        if event.key in '+=X':
            if self.twindow * self.samplerate > 20:
                self.twindow *= 0.5
                self.update_plots()
        elif event.key in '-x':
            if self.twindow < self.tmax:
                self.twindow *= 2.0
                self.update_plots()
        elif event.key == 'pagedown':
            if self.toffset + 0.5 * self.twindow < self.tmax:
                self.toffset += 0.5 * self.twindow
                self.update_plots()
        elif event.key == 'pageup':
            if self.toffset > 0:
                self.toffset -= 0.5 * self.twindow
                if self.toffset < 0.0:
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'ctrl+pagedown':
            if self.toffset + 5.0 * self.twindow < self.tmax:
                self.toffset += 5.0 * self.twindow
                self.update_plots()
        elif event.key == 'ctrl+pageup':
            if self.toffset > 0:
                self.toffset -= 5.0 * self.twindow
                if self.toffset < 0.0:
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'down':
            if self.toffset + self.twindow < self.tmax:
                self.toffset += 0.05 * self.twindow
                self.update_plots()
        elif event.key == 'up':
            if self.toffset > 0.0:
                self.toffset -= 0.05 * self.twindow
                if self.toffset < 0.0:
                    self.toffset = 0.0
                self.update_plots()
        elif event.key == 'home':
            if self.toffset > 0.0:
                self.toffset = 0.0
                self.update_plots()
        elif event.key == 'end':
            toffs = np.floor(self.tmax / self.twindow) * self.twindow
            if self.tmax - toffs <= 0.0:
                toffs -= self.twindow
            if self.tmax - toffs < self.twindow/2:
                toffs -= self.twindow/2
            if self.toffset < toffs:
                self.toffset = toffs
                self.update_plots()
        elif event.key == 'y':
            for t in range(self.traces):
                h = self.ymax[t] - self.ymin[t]
                c = 0.5 * (self.ymax[t] + self.ymin[t])
                self.ymin[t] = c - h
                self.ymax[t] = c + h
                self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
            self.fig.canvas.draw()
        elif event.key == 'Y':
            for t in range(self.traces):
                h = 0.25 * (self.ymax[t] - self.ymin[t])
                c = 0.5 * (self.ymax[t] + self.ymin[t])
                self.ymin[t] = c - h
                self.ymax[t] = c + h
                self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
            self.fig.canvas.draw()
        elif event.key == 'v':
            t0 = int(np.round(self.toffset * self.samplerate))
            t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
            min = np.min(self.data[t0:t1,self.show_channels])
            max = np.max(self.data[t0:t1,self.show_channels])
            h = 0.53 * (max - min)
            c = 0.5 * (max + min)
            self.ymin[:] = c - h
            self.ymax[:] = c + h
            for t in range(self.traces):
                self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
            self.fig.canvas.draw()
        elif event.key == 'ctrl+v':
            t0 = int(np.round(self.toffset * self.samplerate))
            t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
            for t in range(self.traces):
                min = np.min(self.data[t0:t1,self.show_channels[t]])
                max = np.max(self.data[t0:t1,self.show_channels[t]])
                h = 0.53 * (max - min)
                c = 0.5 * (max + min)
                self.ymin[t] = c - h
                self.ymax[t] = c + h
                self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
            self.fig.canvas.draw()
        elif event.key == 'V':
            self.ymin[:] = -1.0
            self.ymax[:] = +1.0
            for t in range(self.traces):
                self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
            self.fig.canvas.draw()
        elif event.key == 'c':
            for t in range(self.traces):
                dy = self.ymax[t] - self.ymin[t]
                self.ymin[t] = -dy/2
                self.ymax[t] = +dy/2
                self.axs[t].set_ylim(self.ymin[t], self.ymax[t])
            self.fig.canvas.draw()
        elif event.key == 'i':
            if len(self.pulses) > 0:
                self.fmax *= 2
                self.axs[-1].set_ylim(0.0, self.fmax)
                self.fig.canvas.draw()
        elif event.key == 'I':
            if len(self.pulses) > 0:
                self.fmax /= 2
                self.axs[-1].set_ylim(0.0, self.fmax)
                self.fig.canvas.draw()
        elif event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            cc = int(event.key)
            # TODO: this is not yet what we want:
            """
            if cc < self.channels:
                self.axs[cc].set_visible(not self.axs[cc].get_visible())
            self.fig.canvas.draw()
            """
        elif event.key in 'h':
            self.help = not self.help
            for ht in self.helptext:
                ht.set_visible(self.help)
            self.fig.canvas.draw()
        elif event.key in 'p':
            self.play_segment()
        elif event.key in 'P':
            self.play_all()
        elif event.key in 'S':
            self.save_segment()
        elif event.key in 'w':
            self.plot_traces()

    def play_segment(self):
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        playdata = 1.0 * np.mean(self.data[t0:t1,self.show_channels], 1)
        f = 0.1 if self.twindow > 0.5 else 0.1*self.twindow
        fade(playdata, self.samplerate, f)
        self.audio.play(playdata, self.samplerate, blocking=False)
        
    def play_all(self):
        self.audio.play(np.mean(self.data[:,self.show_channels], 1),
                        self.samplerate, blocking=False)

    def save_segment(self):
        t0s = int(np.round(self.toffset))
        t1s = int(np.round(self.toffset + self.twindow))
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        filename = self.filename.split('.')[0]
        if self.traces == self.channels:
            segment_filename = f'{filename}-{t0s:.4g}s-{t1s:.4g}s.wav'
            write_audio(segment_filename, self.data[t0:t1,:], self.samplerate)
        else:
            segment_filename = f'{filename}-{t0s:.4g}s-{t1s:.4g}s-c{self.show_channels[0]}.wav'
            write_audio(segment_filename,
                        self.data[t0:t1,self.show_channels], self.samplerate)
        print('saved segment to: ' , segment_filename)

    def plot_traces(self):
        splts = self.traces
        if len(self.pulses) > 0:
            splts += 1
        fig, axs = plt.subplots(splts, 1, squeeze=False, sharex=True,
                                figsize=(15, 9))
        axs = axs.flat
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.05, top=0.97,
                            hspace=0)
        name = self.filename.split('.')[0]
        figfile = f'{name}-{self.toffset:.4g}s-traces.png'
        if self.traces < self.channels:
            figfile = f'{name}-{self.toffset:.4g}s-c{self.show_channels[0]}-traces.png'
        axs[0].set_title(self.filename)
        t0 = int(np.round(self.toffset * self.samplerate))
        t1 = int(np.round((self.toffset + self.twindow) * self.samplerate))
        if t1>len(self.data):
            t1 = len(self.data)
        time = np.arange(t0, t1)/self.samplerate
        if self.toffs < 1.0 and self.twindow < 1.0:
            axs[-1].set_xlabel('Time [ms]')
            for t in range(self.traces):
                c = self.show_channels[t]
                axs[t].set_xlim(1000.0 * self.toffset,
                                1000.0 * (self.toffset + self.twindow))
                axs[t].plot(1000.0 * time, self.data[t0:t1,c])
            self.plot_pulses(axs, True, 1000.0)
        else:
            axs[-1].set_xlabel('Time [s]')
            for t in range(self.traces):
                c = self.show_channels[t]
                axs[t].set_xlim(self.toffset, self.toffset + self.twindow)
                axs[t].plot(time, self.data[t0:t1,c])
            self.plot_pulses(axs, True, 1.0)
        for t in range(self.traces):
            c = self.show_channels[t]
            axs[t].set_ylim(self.ymin[t], self.ymax[t])
            axs[t].set_ylabel(f'C-{c+1} [{self.unit}]')
        if len(self.pulses) > 0:
            axs[-1].set_ylabel('IP freq [Hz]')
            axs[-1].set_ylim(0.0, self.fmax)
        #for t in range(self.traces-1):
        #    axs[t].xaxis.set_major_formatter(plt.NullFormatter())
        fig.savefig(figfile, dpi=200)
        plt.close(fig)
        print('saved waveform figure to', figfile)
        

def short_user_warning(message, category, filename, lineno, file=None, line=''):
    if file is None:
        file = sys.stderr
    if category == UserWarning:
        file.write('%s line %d: %s\n' % ('/'.join(filename.split('/')[-2:]), lineno, message))
    else:
        s = warnings.formatwarning(message, category, filename, lineno, line)
        file.write(s)


def main(cargs=None):
    warnings.showwarning = short_user_warning

    # config file name:
    cfgfile = __package__ + '.cfg'

    # command line arguments:
    if cargs is None:
        cargs = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Browse mutlichannel EOD recordings.',
        epilog='version %s by Benda-Lab (2022-%s)' % (__version__, __year__))
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose')
    parser.add_argument('-c', dest='channels', default='',
                        type=str, metavar='CHANNELS',
                        help='Comma separated list of channels to be displayed (first channel is 0).')
    parser.add_argument('-f', dest='fcutoff', default=None,
                        type=float, metavar='FREQ',
                        help='Cutoff frequency of optional high-pass filter.')
    parser.add_argument('-p', dest='pulses', action='store_true',
                        help='detect pulse fish EODs')
    parser.add_argument('file', nargs=1, default='', type=str,
                        help='name of the file with the time series data')
    args = parser.parse_args(cargs)
    filepath = args.file[0]
    cs = [s.strip() for s in args.channels.split(',')]
    channels = [int(c) for c in cs if len(c)>0]
    fcutoff = args.fcutoff
    pulses = args.pulses

    # set verbosity level from command line:
    verbose = 0
    if args.verbose != None:
        verbose = args.verbose

    # load data:
    filename = os.path.basename(filepath)
    with open_data(filepath, -1, 20.0, 5.0, verbose) as data:
        SignalPlot(data, data.samplerate, data.unit, filename,
                   channels, fcutoff, pulses)
        

        
if __name__ == '__main__':
    main()
