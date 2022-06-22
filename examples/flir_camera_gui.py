#!/bin/env python
"""
"""
from time import time, sleep, clock
import sys
import os.path
import struct
from pdb import pm
import traceback
from time import gmtime, strftime
import logging

#from setting import setting
from threading import Thread, RLock
lock = RLock()

from logging import error,warning,info,debug
import platform

import matplotlib.pyplot as plt
import matplotlib
if platform.system() == 'Windows':
    matplotlib.use('TkAgg')
else:
    matplotlib.use('WxAgg')

from datetime import datetime

from math import log10, floor
from numpy import arange, asfarray, argmax, amax, amin, argmin, gradient, size, random, nan, inf, mean, std, asarray, where, array, concatenate, delete, shape, round, vstack, hstack, zeros, transpose, split, unique, nonzero, take, savetxt, min, max, savetxt, median
import wx # wxPython GUI library


"""Graphical User Interface"""
class GUI(wx.Frame):

    def __init__(self):
        self.name = platform.node() + '_'+'GUI'
        self.lastN_history = 0
        self.lastM_history = 10000
        self.gui_labels = {}
        self.gui_fields = {}
        self.gui_sizers = {}
        self.box_sizer = {}
        self.sizer_main = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_left = wx.BoxSizer(wx.VERTICAL)
        self.sizer_right = wx.BoxSizer(wx.VERTICAL)

        self.box_sizer[b'graph_image'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'graph_cols'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'graph_rows'] = wx.BoxSizer(wx.VERTICAL)

        self.box_sizer[b'c_gain'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'c_exposure'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'c_roi_rs'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'c_roi_re'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'c_roi_cs'] = wx.BoxSizer(wx.VERTICAL)
        self.box_sizer[b'c_roi_ce'] = wx.BoxSizer(wx.VERTICAL)


    def init(self):
        self.create_GUI()

    def create_GUI(self):

        #self.selectedPressureUnits = 'kbar'
        self.xs_font = 10
        self.s_font = 12
        self.m_font = 16
        self.l_font = 24
        self.xl_font = 32
        self.xl_font = 60
        self.wx_xs_font = wx_xs_font=wx.Font(self.xs_font,wx.DEFAULT,wx.NORMAL,wx.NORMAL)
        self.wx_s_font = wx_s_font=wx.Font(self.s_font,wx.DEFAULT,wx.NORMAL,wx.NORMAL)
        self.wx_m_font = wx_m_font=wx.Font(self.m_font,wx.DEFAULT,wx.NORMAL,wx.NORMAL)
        self.wx_l_font = wx_l_font=wx.Font(self.l_font,wx.DEFAULT,wx.NORMAL,wx.NORMAL)
        self.wx_xl_font = wx_xl_font=wx.Font(self.xl_font,wx.DEFAULT,wx.NORMAL,wx.NORMAL)
        self.wx_xxl_font = wx_xxl_font=wx.Font(self.xl_font,wx.DEFAULT,wx.NORMAL,wx.NORMAL)



        frame = wx.Frame.__init__(self, None, wx.ID_ANY, "FLIR cameras")#, size = (1200,1000))#, style= wx.SYSTEM_MENU | wx.CAPTION)
        self.panel = wx.Panel(self, wx.ID_ANY, style=wx.BORDER_THEME)#, size = (1200,1000))
        self.SetBackgroundColour('white')
        self.Bind(wx.EVT_CLOSE, self.onQuit)
        self.statusbar = self.CreateStatusBar() # Will likely merge the two fields unless we can think of a reason to keep them split
        self.statusbar.SetStatusText('This goes field one')
        #self.statusbar.SetStatusText('Field 2 here!', 1)
        self.statusbar.SetBackgroundColour('green')


        ###########################################################################
        ##MENU for the GUI
        ###########################################################################
        file_item = {}
        about_item = {}

        self.setting_item = {}



        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        file_item[0] = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit application')
        self.Bind(wx.EVT_MENU, self.onQuit, file_item[0])

        aboutMenu = wx.Menu()
        about_item[0]= aboutMenu.Append(wx.ID_ANY,  'About')
        self.Bind(wx.EVT_MENU, self._on_about, about_item[0])

        menubar.Append(fileMenu, '&File')

        menubar.Append(aboutMenu, '&About')

        self.SetMenuBar(menubar)


        self.Centre()
        self.Show(True)
        sizer = wx.GridBagSizer(hgap = 0, vgap = 0)#(13, 11)

        ###########################################################################
        ###MENU ENDS###
        ###########################################################################

        ###########################################################################
        ###FIGURE####
        ###########################################################################

        self.bitmap = wx.StaticBitmap(self.panel)
        self.gui_sizers[b'graph_image'] = wx.BoxSizer(wx.VERTICAL)
        self.gui_sizers[b'graph_image'].Add(self.bitmap,0)
        ###########################################################################
        ###FIGURE ENDS####
        ###########################################################################


        ###########################################################################
        ###On Button Press###
        ###########################################################################
        ###Sidebar###
        ###
        lst = ['dm16', 'dm34', 'dm4']

        self.gui_fields[b'camera'] = wx.RadioBox(self.panel, label = 'camera', choices = lst,
                                majorDimension = 1, style = wx.RA_SPECIFY_ROWS|wx.ALIGN_CENTER_HORIZONTAL, size = (400,-1))

        self.gui_sizers[b'camera'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'camera'].Add(self.gui_fields[b'camera'],0)
        self.gui_fields[b'camera'].Bind(wx.EVT_RADIOBOX,self.onCamera)

        self.gui_labels[b'gain'] = wx.StaticText(self.panel, label= 'gain [dB]')
        self.gui_labels[b'gain'].SetFont(wx_m_font)
        self.gui_fields[b'gain'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (60,-1), value = 'nan')
        self.gui_sizers[b'gain'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'gain'].Add(self.gui_labels[b'gain'],0)
        self.gui_sizers[b'gain'].Add(self.gui_fields[b'gain'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_set_gain, self.gui_fields[b'gain'])

        self.gui_labels[b'exposure'] = wx.StaticText(self.panel, label= 'exposure time [ms]')
        self.gui_labels[b'exposure'].SetFont(wx_m_font)
        self.gui_fields[b'exposure'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (180,-1), value = 'nan')
        self.gui_sizers[b'exposure'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'exposure'].Add(self.gui_labels[b'exposure'],0)
        self.gui_sizers[b'exposure'].Add(self.gui_fields[b'exposure'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_set_exposure, self.gui_fields[b'exposure'])

        self.gui_labels[b'roiRows'] = wx.StaticText(self.panel, label= 'roi rows')
        self.gui_labels[b'roiRows'].SetFont(wx_m_font)
        self.gui_fields[b'roiRows_s'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (100,-1), value = 'nan')
        self.gui_fields[b'roiRows_e'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (100,-1), value = 'nan')
        self.gui_sizers[b'roiRows'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'roiRows'].Add(self.gui_labels[b'roiRows'],0)
        self.gui_sizers[b'roiRows'].Add(self.gui_fields[b'roiRows_s'],0)
        self.gui_sizers[b'roiRows'].Add(self.gui_fields[b'roiRows_e'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_roiRows_s, self.gui_fields[b'roiRows_s'])
        self.Bind(wx.EVT_TEXT_ENTER, self.on_roiRows_e, self.gui_fields[b'roiRows_e'])

        self.gui_labels[b'roiCols'] = wx.StaticText(self.panel, label= 'roi cols')

        self.gui_labels[b'roiCols'].SetFont(wx_m_font)
        self.gui_fields[b'roiCols_s'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (100,-1), value = 'nan')
        self.gui_fields[b'roiCols_e'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (100,-1), value = 'nan')
        self.gui_sizers[b'roiCols'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'roiCols'].Add(self.gui_labels[b'roiCols'],0)
        self.gui_sizers[b'roiCols'].Add(self.gui_fields[b'roiCols_s'],0)
        self.gui_sizers[b'roiCols'].Add(self.gui_fields[b'roiCols_e'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_roiCols_s, self.gui_fields[b'roiCols_s'])
        self.Bind(wx.EVT_TEXT_ENTER, self.on_roiCols_e, self.gui_fields[b'roiCols_e'])

        self.gui_labels[b'position'] = wx.StaticText(self.panel, label= 'position [inches] ')
        self.gui_labels[b'position'].SetFont(wx_m_font)
        self.gui_fields[b'position'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (200,-1), value = 'nan')
        self.gui_sizers[b'position'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'position'].Add(self.gui_labels[b'position'],0)
        self.gui_sizers[b'position'].Add(self.gui_fields[b'position'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_position, self.gui_fields[b'position'])

        self.gui_labels[b'gamma'] = wx.StaticText(self.panel, label= 'gamma')
        self.gui_labels[b'gamma'].SetFont(wx_m_font)
        self.gui_fields[b'gamma'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (200,-1), value = 'nan')
        self.gui_sizers[b'gamma'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'gamma'].Add(self.gui_labels[b'gamma'],0)
        self.gui_sizers[b'gamma'].Add(self.gui_fields[b'gamma'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_set_gamma, self.gui_fields[b'gamma'])

        self.gui_labels[b'filename'] = wx.StaticText(self.panel, label= 'filename prefix')
        self.gui_labels[b'filename'].SetFont(wx_m_font)
        self.gui_fields[b'filename'] = wx.TextCtrl(self.panel,-1, style = wx.TE_PROCESS_ENTER, size = (200,-1), value = 'nan')
        self.gui_sizers[b'filename'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'filename'].Add(self.gui_labels[b'filename'],0)
        self.gui_sizers[b'filename'].Add(self.gui_fields[b'filename'],0)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_filename, self.gui_fields[b'filename'])

        self.gui_fields[b'save_to_file'] = wx.Button(self.panel, label = 'Save')
        self.gui_fields[b'save_to_file'].Bind(wx.EVT_BUTTON, self.on_save_to_file)
        self.gui_fields[b'save_to_file'].SetFont(wx_m_font)
        self.gui_sizers[b'save_to_file'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'save_to_file'].Add(self.gui_fields[b'save_to_file'],0)

        self.gui_fields[b'set_background'] = wx.Button(self.panel, label = 'Set Background')
        self.gui_fields[b'set_background'].Bind(wx.EVT_BUTTON, self.on_set_background)
        self.gui_fields[b'set_background'].SetFont(wx_m_font)
        self.gui_sizers[b'set_background'] = wx.BoxSizer(wx.HORIZONTAL)
        self.gui_sizers[b'set_background'].Add(self.gui_fields[b'set_background'],0)


        self.gui_labels[b'intensity'] = wx.StaticText(self.panel, size = (220,-1),label= 'M00 intensity')
        self.gui_labels[b'intensity'].SetFont(wx_l_font)
        self.gui_fields[b'intensity'] = wx.StaticText(self.panel, size = (100,-1), label = 'nan')
        self.gui_fields[b'intensity'].SetFont(wx_l_font)

        self.gui_sizers[b'intensity'] = wx.BoxSizer(wx.VERTICAL)
        self.gui_sizers[b'intensity'].Add(self.gui_labels[b'intensity'],0)
        self.gui_sizers[b'intensity'].Add(self.gui_fields[b'intensity'],0)

        self.gui_labels[b'moments_max'] = wx.StaticText(self.panel, size = (370,-1),label= 'Max Pixel')
        self.gui_labels[b'moments_max'].SetFont(wx_m_font)
        self.gui_fields[b'moments_max_value'] = wx.StaticText(self.panel, size = (300,-1), label = 'Imax = nan')
        self.gui_fields[b'moments_max_x'] = wx.StaticText(self.panel, size = (300,-1), label = 'Imax,x = nan')
        self.gui_fields[b'moments_max_y'] = wx.StaticText(self.panel,-1, size = (300,-1), label = 'Imax,y = nan')
        self.gui_sizers[b'moments_max'] = wx.BoxSizer(wx.VERTICAL)
        self.gui_fields[b'moments_max_value'].SetFont(wx_l_font)
        self.gui_fields[b'moments_max_x'].SetFont(wx_l_font)
        self.gui_fields[b'moments_max_y'].SetFont(wx_l_font)
        self.gui_sizers[b'moments_max'].Add(self.gui_labels[b'moments_max'],0)
        self.gui_sizers[b'moments_max'].Add(self.gui_fields[b'moments_max_value'],0)
        self.gui_sizers[b'moments_max'].Add(self.gui_fields[b'moments_max_x'],0)
        self.gui_sizers[b'moments_max'].Add(self.gui_fields[b'moments_max_y'],0)

        self.gui_labels[b'moments_mean'] = wx.StaticText(self.panel, size = (170,-1),label= 'moments mean')
        self.gui_labels[b'moments_mean'].SetFont(wx_m_font)
        self.gui_fields[b'moments_mean_x'] = wx.StaticText(self.panel, size = (200,-1), label = 'nan')
        self.gui_fields[b'moments_mean_y'] = wx.StaticText(self.panel,-1, size = (200,-1), label = 'nan')
        self.gui_sizers[b'moments_mean'] = wx.BoxSizer(wx.VERTICAL)
        self.gui_fields[b'moments_mean_x'].SetFont(wx_l_font)
        self.gui_fields[b'moments_mean_y'].SetFont(wx_l_font)
        self.gui_sizers[b'moments_mean'].Add(self.gui_labels[b'moments_mean'],0)
        self.gui_sizers[b'moments_mean'].Add(self.gui_fields[b'moments_mean_x'],0)
        self.gui_sizers[b'moments_mean'].Add(self.gui_fields[b'moments_mean_y'],0)

        self.gui_labels[b'moments_var'] = wx.StaticText(self.panel, size = (170,-1),label= 'moments var')
        self.gui_labels[b'moments_var'].SetFont(wx_m_font)
        self.gui_fields[b'moments_var_x'] = wx.StaticText(self.panel, size = (200,-1), label = 'nan')
        self.gui_fields[b'moments_var_y'] = wx.StaticText(self.panel, size = (200,-1), label = 'nan')
        self.gui_fields[b'moments_var_x'].SetFont(wx_l_font)
        self.gui_fields[b'moments_var_y'].SetFont(wx_l_font)
        self.gui_sizers[b'moments_var'] = wx.BoxSizer(wx.VERTICAL)
        self.gui_sizers[b'moments_var'].Add(self.gui_labels[b'moments_var'],0)
        self.gui_sizers[b'moments_var'].Add(self.gui_fields[b'moments_var_x'],0)
        self.gui_sizers[b'moments_var'].Add(self.gui_fields[b'moments_var_y'],0)



        self.sizer_left.Add(self.gui_sizers[b'graph_image'])

        self.sizer_right.Add(self.gui_sizers[b'camera'])
        self.sizer_right.Add(self.gui_sizers[b'gain'])
        self.sizer_right.Add(self.gui_sizers[b'exposure'])
        self.sizer_right.Add(self.gui_sizers[b'roiRows'])
        self.sizer_right.Add(self.gui_sizers[b'roiCols'])
        self.sizer_right.Add(self.gui_sizers[b'position'])
        self.sizer_right.Add(self.gui_sizers[b'gamma'])
        self.sizer_right.Add(self.gui_sizers[b'filename'])
        self.sizer_right.Add(self.gui_sizers[b'save_to_file'])
        self.sizer_right.Add(self.gui_sizers[b'set_background'])

        self.sizer_right.Add(wx.StaticLine(self.panel), 0, wx.ALL|wx.EXPAND, 5)
        self.sizer_right.Add(self.gui_sizers[b'intensity'])
        self.sizer_right.Add(self.gui_sizers[b'moments_max'])
        self.sizer_right.Add(self.gui_sizers[b'moments_mean'])
        self.sizer_right.Add(self.gui_sizers[b'moments_var'])

        self.sizer_right.Add(wx.StaticLine(self.panel), 0, wx.ALL|wx.EXPAND, 5)


        self.sizer_main.Add(self.sizer_left,0)
        self.sizer_main.Add(self.sizer_right,0)

        #self.sizer_left.Hide()

        self.draw()

    def draw(self):
        self.Show()

        self.panel.SetSizer(self.sizer_main)
        self.sizer_main.Fit(self)
        self.Layout()
        self.panel.Layout()
        self.panel.Fit()
        self.Fit()
        self.Update()
    #----------------------------------------------------------------------


    def onCamera(self,event):
        print('onRadiobBox selected: %r' % self.gui_fields[b'camera'].GetStringSelection())
        if self.gui_fields[b'camera'].GetStringSelection() == 'dm16':
            try:
                gui_backend.camera = camera_dm16
            except:
                print('no camera dm16')
        if self.gui_fields[b'camera'].GetStringSelection() == 'dm34':
            try:
                gui_backend.camera = camera_dm34
            except:
                print('no camera dm34')
        if self.gui_fields[b'camera'].GetStringSelection() == 'dm4':
            try:
                gui_backend.camera = camera_dm4

            except:
                print('no camera dm4')
        #gui_backend.set_gain_exp()
        self.draw()

    def on_roiRows_s(self,event):
        gui_backend.roi_row_s = int(event.GetString())

    def on_roiRows_e(self,event):
        gui_backend.roi_row_e = int(event.GetString())

    def on_roiCols_s(self,event):
        gui_backend.roi_col_s = int(event.GetString())

    def on_roiCols_e(self,event):
        gui_backend.roi_col_e = int(event.GetString())

    def on_filename(self,event):
        gui_backend.filename_prefix = event.GetString()

    def on_position(self,event):
        gui_backend.position = int(event.GetString())
        gui_backend.center_on_feducial(pos = gui_backend.position)

    def on_set_background(self,event):
        print('on_set_background')
        gui_backend.set_background()

    def on_save_to_file(self,event):
        print('on_save_to_file')
        gui_backend.save_to_file()

    def _on_about(self,event):
        message = str(__doc__)
        wx.MessageBox(message,'About', wx.OK | wx.ICON_INFORMATION)

##    def UpdateValues(self,event): # legacy code, used just for checking buffer and will likely  be removed later
##        self.buffer_text.SetLabel(str(DAQ.RingBuffer.buffer[:,DAQ.RingBuffer.pointer])+'\n'+ str(DAQ.RingBuffer.pointer) +' : '+ str(time()))

    def onQuit(self,event):
        #FIXIT uncomment all
        #icarus_AL.GUI_running = False
        #icarus_AL.kill()
        del self
        os._exit(1)

    def on_set_gamma(self,event):
        gamma = float(event.GetString())
        print('Gmma pressed',gamma)
        if gamma > 0:
            gui_backend.camera.gamma_enable = True
            gui_backend.camera.gamma = gamma
        else:
            gui_backend.camera.gamma_enable = False

    def on_set_gain(self,event):
         gui_backend.camera.gain = float(event.GetString())

    def on_set_exposure(self,event):
        gui_backend.camera.exposure_time = float(event.GetString())*1000

    def update_moments(self, moments):
        #moments = {'x_mean':0,'y_mean':0,'x_var':0.0,'y_var':0.0}
        self.gui_fields[b'intensity'].SetLabel(str(moments['intensity']))
        self.gui_fields[b'moments_mean_x'].SetLabel(str(moments['x_mean']))
        self.gui_fields[b'moments_mean_y'].SetLabel(str(moments['y_mean']))
        self.gui_fields[b'moments_var_x'].SetLabel(str(moments['x_var']))
        self.gui_fields[b'moments_var_y'].SetLabel(str(moments['y_var']))

        self.gui_fields[b'moments_max_x'].SetLabel('Imax,x = ' + str(moments['x_max']))
        self.gui_fields[b'moments_max_y'].SetLabel('Imax,y = ' + str(moments['y_max']))
        self.gui_fields[b'moments_max_value'].SetLabel('Imax = '+str(round(moments['pixel_max'],1)))


    def draw_figure(self, buf = None):
        """
        shows the bitmap generated in the
        """
        def buf2wx (buf):
            import PIL
            image = PIL.Image.open(buf)
            width, height = image.size
            return wx.Bitmap.FromBuffer(width, height, image.tobytes())
        try:
            self.bitmap.SetBitmap(buf2wx(buf))
        except:
            print(traceback.format_exc())

        self.panel.Layout()
        self.panel.Fit()
        self.Layout()
        self.Fit()

    ###########################################################################
    ###END: On Button Press###
    ###########################################################################

class GuiBackend():

    def __init__(self, camera = None):
        self.camera = camera
        self.roi_row_s = 0
        self.roi_row_e = -1
        self.roi_col_s = 0
        self.roi_col_e = -1
        self.filename_prefix = 'no assigned'
        self.feducials = feducials
        self.gamma = nan

    def run(self):
        from time import sleep
        while True:
            self.redraw_figure()
            sleep(1)

    def start(self):
        from ubcs_auxiliary.threading import new_thread
        new_thread(self.run)


    def redraw_figure(self):
        self.buf = self.plot_image()
        wx.CallAfter(frame.draw_figure,self.buf)


    def get_image(self):
        from numpy import random, mean
        if self.camera is None:
            image = random.randint(2**12, size=(3000, 4096), dtype = 'int16')
        else:
            image = mean(self.camera.queue.peek_last_N(19), axis = 0)
        return image

    def update_moments(self, m = None):
        from numpy import nan
        if m == None:
            moments = {'x_mean':0,'y_mean':0,'x_var':0.0,'y_var':0.0}
        else:
            moments = {}

            if m['m00'] == 0:
                m['m00'] = nan
            moments['x_max'] = m['max_x']
            moments['y_max'] = m['max_y']
            moments['pixel_max'] = m['max_pixel']
            moments['x_mean'] = round((m['m10']/m['m00']) + self.roi_col_s,1)
            moments['y_mean'] = round((m['m01']/m['m00']) + self.roi_row_s,1)
            moments['x_var'] = round(m['m20']/m['m00'] - (m['m10']/m['m00'])**2,2)
            moments['y_var'] = round(m['m02']/m['m00'] - (m['m01']/m['m00'])**2,2)
            moments['intensity'] = round(m['m00'],1)
        wx.CallAfter(frame.update_moments,moments)

    def save_to_file(self):
        """
        """
        from tempfile import gettempdir
        from os.path import join
        from ubcs_auxiliary.save_load_object import save_to_file
        from time import sleep
        if self.camera is None:
            name = 'None'
        else:
            name = self.camera.name
        string = 'feducials_pos_'+str(self.position)
        self.camera.recording_init(N_frames = 200, comments = string)
        sleep(200/19)
        self.camera.recording_start()

    def set_background(self):
        from time import sleep
        length = self.camera.queue.shape[0]
        self.camera.queue.reset()
        while self.camera.queue.length < length:
            sleep(1)
        self.camera.get_background()
        self.camera.background_flag = True


    def plot_image(self):

        import io
        from matplotlib.figure import Figure
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from scipy import stats
        from numpy import nonzero, zeros,nan, ones, argwhere, mean, nanmean

        fig = Figure(figsize=(10,10),dpi=80)#figsize=(7,5))
        grid = plt.GridSpec(3, 3, hspace=0.025, wspace=0.025)
        t1 = time()

        if self.camera is not None:
            if self.camera.background_flag:
                print('background corrected')
                image = self.get_image()-self.camera.background
            else:
                image = self.get_image()
        else:
            image = self.get_image()
        roi_row_s = self.roi_row_s
        roi_row_e = self.roi_row_e
        roi_col_s = self.roi_col_s
        roi_col_e = self.roi_col_e
        img = image[roi_row_s:roi_row_e,roi_col_s:roi_col_e]
        bckg = (img[:,0:5].mean() + img[:,-5:-1].mean() + img[0:5,:].mean() +  img[-5:-1,:].mean())/4
        ax1 = fig.add_subplot(grid[0:2,0:2])
        ax1.imshow(img)
        vrow = img.sum(axis = 1)

        y = arange(0,vrow.shape[0])
        axv = fig.add_subplot(grid[0:2,2], sharey = ax1)
        axv.plot(vrow,y)

        axh = fig.add_subplot(grid[2,0:2], sharex = ax1 )
        vcol = img.sum(axis = 0)
        x = arange(0,vcol.shape[0])
        axh.plot(x,vcol)

        from lcp_video.analysis import get_moments
        img = img.astype('float64')

        m = get_moments(img-bckg)
        idx = where(img == img.max())
        m['max_x'] = idx[1][0] + roi_col_s
        m['max_y'] = idx[0][0] + roi_row_s
        m['max_pixel'] = img.max()
        self.update_moments(m = m)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg')
        buf.seek(0)
        return buf

    def center_on_feducial(self, pos):
        """
        tc - dm4 , 8mm - dm16, 12mm - dm34
        """
        lst = ['dm4','dm16','dm34']
        fds = self.feducials
        name = self.camera.name
        from numpy import where
        import traceback
        step = 50
        try:
            col_c = int(fds[where(fds[:,0] == pos)][0][2*(lst.index(name)+1)+1])
            row_c = int(fds[where(fds[:,0] == pos)][0][2*(lst.index(name)+1)])
            self.roi_col_e = col_c + step
            if col_c - step >= 0:
                self.roi_col_s = col_c - step
            else:
                self.roi_col_s = 0

            self.roi_row_e = row_c + step
            if row_c - step >= 0:
                self.roi_row_s = row_c - step
            else:
                self.roi_row_s = 0

            self.set_roi()
        except:
            print(traceback.format_exc())



    def set_roi(self):
        wx.CallAfter(frame.gui_fields[b'roiCols_s'].SetValue,str(self.roi_col_s))
        wx.CallAfter(frame.gui_fields[b'roiCols_e'].SetValue,str(self.roi_col_e))
        wx.CallAfter(frame.gui_fields[b'roiRows_s'].SetValue,str(self.roi_row_s))
        wx.CallAfter(frame.gui_fields[b'roiRows_e'].SetValue,str(self.roi_row_e))


    def set_gain_exp(self):
        wx.CallAfter(frame.gui_fields[b'gain'].SetValue,str(self.camera.gain))
        wx.CallAfter(frame.gui_fields[b'exposure'].SetValue,str(self.camera.exposure_time))


def init_camera():
    from time import time, sleep
    from numpy import zeros, right_shift, array
    import PySpin
    from PySpin import System
    from lcp_video.flir_camera.flir_camera_DL import FlirCamera

    system = System.GetInstance()

    camera_dm4 = FlirCamera(name = 'dm4',system = system)
    camera_dm4.init(serial_number = '19490369', settings = 1)

    camera_dm34 = FlirCamera(name = 'dm34',system = system)
    camera_dm34.init(serial_number = '18159480', settings = 1)

    camera_dm16 = FlirCamera(name = 'dm16',system = system)
    camera_dm16.init(serial_number = '18159488', settings = 1)

    return camera_dm16, camera_dm34, camera_dm4

def start():
    camera_dm34.start_thread()
    camera_dm4.start_thread()
    camera_dm16.start_thread()

def stop():
    from ubcs_auxiliary.threading import new_thread
    new_thread(camera_dm4.stop_thread)
    new_thread(camera_dm16.stop_thread)
    new_thread(camera_dm34.stop_thread)

if __name__ == "__main__":
    from numpy import loadtxt
    feducials = loadtxt('feducials.dat', delimiter = ',')
    camera_dm16,camera_dm34,camera_dm4 = init_camera()
    start()
    app = wx.App(False)
    camera = None
    frame = GUI()
    frame.init()
    gui_backend = GuiBackend(camera)
    gui_backend.start()

    #app.MainLoop()

    from tempfile import gettempdir
    # logging.basicConfig(filename=gettempdir()+'/flir_camera_gui.log',
    #                     level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
