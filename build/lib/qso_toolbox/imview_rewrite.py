
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib import rc

import numpy as np
import matplotlib.pyplot as plt


from qso_toolbox import image_utils as imu
from qso_toolbox import cat_utils as cu

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# GLOBAL USER INPUT -- TO CHECK BEFORE STARTING THE PYTHON ROUTINE
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# INPUT DATA FILES
#------------------------------------------------------------------------------

# Input File (hdf5, astropy fits table, ...)
catalog_filename = ""
# Image path
image_path = "./cutouts"
# Coordinate column names, either string or list of strings with length N
ra_column_name = 'vhs_ra_j'
dec_column_name = 'vhs_dec_j'
# List of surveys, list with length N
surveys = ['desdr1','desdr1', 'desdr1']
# List of survey bands, list with length N
bands = ['i','z','Y']

# kwargs
# List of psf sizes, either None, float or list with length N
psf_size = None
# List of aperture sizes, either None (automatic) or list with length N
apertures = None

# List of magnitude column names, list with length N
mag_column_names = None
# List of magnitude error column names, list with length N
magerr_column_names = None
# List of S/N column names, list with length N
sn_column_names = None

# List of forced magnitude column names, list with length N
forced_mag_column_names = None
# List of forced magnitude error column names, list with length N
forced_magerr_column_names = None
# List of S/N column names, list with length N
forced_sn_column_names = None

# List of custom visual classification classes (default is point, extended,
# bad pixel, artifact, other)
visual_classes = None



# File with selected QUASAR CANDIDATES
# selected_candidates_filename = 'wise_tmass_candidates_QSO_2_8_tmass.hdf5'
selected_candidates_filename = 'checked_candidates_LW_jun.hdf5'


SDSS_ra_name = 'ps_ra'
SDSS_dec_name = 'ps_dec'

WISE_ra_name = 'wise_ra'
WISE_dec_name = 'wise_dec'

#WISE_ra_name = 'wise_ra'
#WISE_dec_name = 'wise_dec'

TM_ra_name = 'tmass_ra'
TM_dec_name = 'tmass_dec'

# Filter information
SDSS_filters = ['g','r','i','z','y']
WISE_filters = ['W1','W2','W2']
TM_filters = ['j','h','k']
sdss_survey = ['PS', 'PS', 'PS', 'PS', 'PS']
wise_survey = ['WISE', 'WISE', 'WISE']
tmass_survey = ['2Mass', '2Mass', '2Mass']

PS1_filters = ['g', 'r', 'i', 'z', 'y']
ps1_survey = ['PS', 'PS', 'PS', 'PS', 'PS']

# all_surveys = ['SDSS','SDSS','SDSS','SDSS','SDSS','2Mass','2Mass','2Mass','WISE','WISE']
all_surveys = sdss_survey + wise_survey
all_filters = SDSS_filters + WISE_filters

psfmag = ['gMeanPSFMag','rMeanPSFMag','iMeanPSFMag','zMeanPSFMag','yMeanPSFMag','w1mpro','w2mpro','w2mpro']
psfmag_err = ['gMeanPSFMagErr', 'rMeanPSFMagErr', 'iMeanPSFMagErr', 'zMeanPSFMagErr', 'yMeanPSFMagErr', 'w1sigmpro', 'w2sigmpro', 'w2sigmpro']

#psfmag = ['psfmag_u','psfmag_g','psfmag_r','psfmag_i','psfmag_z', 'j_m','h_m','k_m','w1mpro','w2mpro']
#psfmag_err = ['psfmagerr_u', 'psfmagerr_g', 'psfmagerr_r', 'psfmagerr_i', 'psfmagerr_z', 'j_msigcom', 'h_msigcom', 'k_msigcom', 'w1sigmpro', 'w2sigmpro']
psf_size = np.array([1.3,1.3,1.3,1.3,1.3,8.5,8.5]) # in arcsec


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class AppForm(QMainWindow):
    def __init__(self, catalog_filename, ra_column_name, dec_column_name,
                 surveys, bands, psf_size=None, apertures=None,
                 mag_column_names=None, magerr_column_names=None,
                 sn_column_names=None, forced_mag_column_names=None,
                 forced_magerr_column_names=None, forced_sn_column_names=None,
                 auto_download= False, auto_forced_phot=False,
                 visual_classes=None):

        QMainWindow.__init__(self, parent)
        self.setWindowTitle("JT's magic cutout GUI")

        # Read in the catalog file
        # TODO allow for hdf5 and astropy fits table formats
        # TODO save format for output
        self.df = pd.read_hdf(catalog_filename, 'data')
        try:
            self.df['vis_id'] = self.df.vis_id.values
        except:
            self.df['vis_id'] = np.nan

        # Populating class variables
        if visual_classes is not None:
            self.vis_classes = visual_classes
        else:
            self.vis_classes = ['point', 'extended', 'bad pixel', 'artifact',
                                'other']


        # Length of catalog file
        self.len_df = self.df.shape[0]

        self.candidate_number = -1

        self.ind_array = np.arange(self.len_df)

        # Create the GUI components
        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        # TODO Implement a Finding Chart window
        # TODO Implement a spectral view window

        # Draw first canvas
        self.on_draw()

        self.status_dialog = MyDialog(self)

        self.show()

    def on_draw(self):
        """ Redraws the cutout figure
        """

        # TODO
        imu.make_mult_png()
        # TODO returns list of visual class numbers
        calc_visual_class_numbers(df, self.vis_classes)

        # TODO show visual class numbers
        self.bad_num.set_text(r'$'+str(bad)+'$')
        self.good_num.set_text(r'$'+str(good)+'$')
        self.close_num.set_text(r'$'+str(close)+'$')
        self.wblend_num.set_text(r'$'+str(wblend)+'$')
        self.blend_num.set_text(r'$'+str(blend)+'$')
        self.drops_num.set_text(r'$'+str(drops)+'$')
        self.ext_num.set_text(r'$'+str(ext)+'$')



        #-------------------------------
        # Redraw canvas
        #-------------------------------
        self.canvas.draw()


    def goto_cutout(self):

        new_candidate_number = int(self.goto_le.text())

        if new_candidate_number < self.n_candidates-1 and new_candidate_number > 0:
          self.candidate_number = new_candidate_number

          self.download_cutout()

        else:
          print('Candidate number invalid!')

    def next_cutout(self):

        if self.candidate_number < self.n_candidates-1:

          self.candidate_number += 1

          self.download_cutout()
        else:
          print('Maximum candidate number reached')


    def previous_cutout(self):

        if self.candidate_number > 0:

          self.candidate_number -= 1

          self.download_cutout()
        else:
          print('Minimum candidate number reached')

    def save_vis_id(self):
        """ Saves the clicked visual identification and continues to next object

        :return:
        """
        pass

    # def save_close(self):
    #
    # new_visid = 'close'
    #
    # self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    # self.next_cutout()
    #
    # def save_wblend(self):
    #
    # new_visid = 'wblend'
    #
    # self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    # self.next_cutout()
    #
    # def save_ext(self):
    #
    # new_visid = 'ext'
    #
    # self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    # self.next_cutout()
    #
    # def save_blend(self):
    #
    #     new_visid = 'blend'
    #
    #     self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    #     self.next_cutout()
    #
    # def save_bright(self):
    #
    #     new_visid = 'bright'
    #
    #     self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    #     self.next_cutout()
    #
    # def save_good(self):
    #
    # new_visid = 'good'
    #
    # self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    # self.next_cutout()
    #
    # def save_bad(self):
    #
    # new_visid = 'bad'
    #
    # self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    # self.next_cutout()
    #
    #
    # def save_visid(self):
    #
    # new_visid = str(self.visid_le.text())
    #
    # self.candidate_df.loc[self.candidate_df.index[self.candidate_number],'vis_id'] = new_visid
    #
    # self.on_draw()

    def save_candidate_list(self):
        """ Saves the dataframe in a hdf5 or astropy fits table format

        :return:
        """
        candidate_list_filename = str(self.output_le.text())

        self.candidate_df.to_hdf(candidate_list_filename,'data')

        print('Checked candidate list SAVED')



    def create_main_frame(self):
        self.main_frame = QWidget()

        # Create the mpl Figure and FigCanvas objects.
        self.fig, self.axes = plt.subplots(nrows=2, ncols=4, figsize=(15,12),
                                           dpi=140)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        # self.fig.subplots_adjust(hspace=0, wspace=0.5, left = 0.05, right = 0.85, bottom =0.05, top = 0.95)

        #-------------------------------
        # Adding candidate information
        #-------------------------------

        # self.type_text = plt.figtext(0.67,0.93,r'$\rm{Type}$',horizontalalignment='left',color='k',fontsize = 10)
        # self.type_number = plt.figtext(0.75,0.93,r'$?$',horizontalalignment='left',color='k',fontsize = 10)
        #
        # self.number_text = plt.figtext(0.87,0.93,r'$\#???/'+str(self.n_candidates)+'$',horizontalalignment='left',color='k',fontsize = 10)
        # self.visid_text = plt.figtext(0.87, 0.85,r'$\rm{Visual\ ID:} $', horizontalalignment='left',color='k',fontsize = 9)
        # self.index_text = plt.figtext(0.87, 0.80,r'$\rm{Index:} index$', horizontalalignment='left',color='k',fontsize = 9)

        #self.w1_unwise_text = plt.figtext(0.58, 0.02,r'$\rm{UW1}=$', horizontalalignment='left',color='k',fontsize = 9)
        #self.w1_unwise_value =  plt.figtext(0.63, 0.02,r'$?$', horizontalalignment='left',color='k',fontsize = 9)

        #self.w2_unwise_text = plt.figtext(0.75, 0.02,r'$\rm{UW2}=$', horizontalalignment='left',color='k',fontsize = 9)
        #self.w2_unwise_value =  plt.figtext(0.80, 0.02,r'$?$', horizontalalignment='left',color='k',fontsize = 9)

        # self.ug_text  = plt.figtext(0.87, 0.75,r'$psf-ap (i) :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.gr_text  = plt.figtext(0.87, 0.70,r'$g-r :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.ri_text  = plt.figtext(0.87, 0.65,r'$r-i :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.zj_text  = plt.figtext(0.87, 0.60,r'$z-j :$', horizontalalignment='left',color='k',fontsize = 9)
        #
        # self.ug_val  = plt.figtext(0.93, 0.75,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        # self.gr_val  = plt.figtext(0.93, 0.70,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        # self.ri_val  = plt.figtext(0.93, 0.65,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        # self.zj_val  = plt.figtext(0.93, 0.60,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        #
        # self.emp_z_text  = plt.figtext(0.87, 0.55,r'$z_{rf,emp} :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.emp_z_val  = plt.figtext(0.93, 0.55,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        #
        # self.sim_z_text  = plt.figtext(0.87, 0.50,r'$z_{rf,sim} :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.sim_z_val  = plt.figtext(0.93, 0.50,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        #
        # self.pf_z_text  = plt.figtext(0.87, 0.45,r'$z_{pf} :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.pf_z_val  = plt.figtext(0.93, 0.45,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        #
        # self.p_text  = plt.figtext(0.87, 0.40,r'$p :$', horizontalalignment='left',color='k',fontsize = 9)
        # self.p_val  = plt.figtext(0.93, 0.40,r'$?$', horizontalalignment='left',color='k',fontsize = 9)
        #
        # #self.zero_text = plt.figtext(0.87, 0.40,r'$\rm{0}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.bad_text = plt.figtext(0.87, 0.35,r'$\rm{bad}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.good_text = plt.figtext(0.87, 0.31,r'$\rm{good}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.close_text = plt.figtext(0.87, 0.27,r'$\rm{close}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.wblend_text = plt.figtext(0.87, 0.23,r'$\rm{wblend}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.blend_text = plt.figtext(0.87, 0.19,r'$\rm{blend}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.ext_text = plt.figtext(0.87, 0.15,r'$\rm{ext}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.drops_text = plt.figtext(0.87, 0.11,r'$\rm{mag\ drops}:$', horizontalalignment='left',color='k',fontsize = 8)
        # self.info_text = plt.figtext(0.855, 0.07,r'$\rm{Wise\ Neighbors:}$', horizontalalignment='left',color='k',fontsize = 6)
        #
        # #self.zero_num = plt.figtext(0.97, 0.40,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.bad_num = plt.figtext(0.97, 0.35,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.good_num = plt.figtext(0.97, 0.31,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.close_num = plt.figtext(0.97, 0.27,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.wblend_num = plt.figtext(0.97, 0.23,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.blend_num = plt.figtext(0.97, 0.19,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.ext_num = plt.figtext(0.97, 0.15,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.drops_num = plt.figtext(0.97, 0.11,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        # self.info_num = plt.figtext(0.97, 0.07,r'$?$', horizontalalignment='center',color='k',fontsize = 8)
        #-------------------------------

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.cmap_lbl = QLabel("Color map:")
        self.cmap_le = QLineEdit('viridis')
        self.cmap_le.setMaxLength(15)

        self.visid_lbl = QLabel("Visual identification:")
        self.visid_le = QLineEdit('new vis_id')
        self.visid_le.setMaxLength(15)

        self.clipsize_lbl = QLabel("Clipsize in arcsec:")
        self.clipsize_input = QLineEdit('30')
        self.clipsize_input.setMaxLength(3)

        self.nsigma_lbl = QLabel("Color scale # of std:")
        self.nsigma_input = QLineEdit('5')
        self.nsigma_input.setMaxLength(3)

        self.output_lbl = QLabel("Output filename:")
        self.output_le = QLineEdit('checked_candidates.hdf5')
        self.output_le.setMaxLength(30)

        self.goto_le = QLineEdit('1')
        self.goto_le.setMaxLength(4)

        self.save_list_button = QPushButton("Save candidate list")
        self.save_list_button.clicked.connect(self.save_candidate_list)

        self.save_visid_button = QPushButton("Save vis_id")
        self.save_visid_button.clicked.connect(self.save_visid)

        self.good_button = QPushButton("good")
        self.good_button.clicked.connect(self.save_good)

        self.close_button = QPushButton("close")
        self.close_button.clicked.connect(self.save_close)

        self.wblend_button = QPushButton("wblend")
        self.wblend_button.clicked.connect(self.save_wblend)

        self.ext_button = QPushButton("ext")
        self.ext_button.clicked.connect(self.save_ext)

        self.blend_button = QPushButton("bright")
        self.blend_button.clicked.connect(self.save_bright)

        self.bad_button = QPushButton("bad")
        self.bad_button.clicked.connect(self.save_bad)

        self.previous_button = QPushButton("Previous Cutout")
        self.previous_button.clicked.connect(self.previous_cutout)

        self.next_button = QPushButton("Next Cutout")
        self.next_button.clicked.connect(self.next_cutout)

        self.goto_button = QPushButton("Go to")
        self.goto_button.clicked.connect(self.goto_cutout)

        self.redraw_button = QPushButton("Redraw")
        self.redraw_button.clicked.connect(self.download_cutout)

        self.download_button = QPushButton("Full Download")
        self.download_button.clicked.connect(self.download_all_image_data)

        # Layout with box sizers
        hbox = QHBoxLayout()
        hbox2 = QHBoxLayout()
        # hbox3 = QHBoxLayout()

        for w in [self.download_button, self.redraw_button, self.previous_button, self.next_button, self.goto_button, self.goto_le]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)

        for w in [self.clipsize_lbl, self.clipsize_input, self.cmap_lbl, self.cmap_le, self.nsigma_lbl, self.nsigma_input]:
            hbox2.addWidget(w)
            hbox2.setAlignment(w, Qt.AlignVCenter)

        # for w in [self.visid_lbl, self.visid_le,self.save_visid_button, self.good_button,
        #   self.close_button, self.ext_button,self.wblend_button, self.blend_button, self.bad_button, self.output_lbl, self.output_le, self.save_list_button]:
        #     hbox3.addWidget(w)
        #     hbox3.setAlignment(w, Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        # vbox.addLayout(hbox3)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)


    def create_status_bar(self):
        self.status_text = QLabel("This is the development version of the "
                                  "JT's magic cutout GUI")
        self.statusBar().addWidget(self.status_text, 1)


    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        load_file_action = self.create_action("&Save plot",
            shortcut="Ctrl+S", slot=self.save_plot,
            tip="Save the plot")
        quit_action = self.create_action("&Quit", slot=self.close,
            shortcut="Ctrl+Q", tip="Close the application")

        self.add_actions(self.file_menu,
            (load_file_action, None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About",
            shortcut='F1', slot=self.on_about,
            tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(  self, text, slot=None, shortcut=None,
                        icon=None, tip=None, checkable=False,
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        #if slot is not None:
            #self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action

    def save_plot(self):
        file_choices = "PNG (*.png)|*.png"

        path = unicode(QFileDialog.getSaveFileName(self,
                        'Save file', '',
                        file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

    def on_about(self):
        msg = """ JT's magic cutout image view GUI
        Author: Jan-Torge Schindler (schindler@mpia.de)
        Last modified: 10/22/14
        """
        QMessageBox.about(self, "About", msg.strip())



class MyDialog(QDialog):
    def __init__(self, parent = AppForm):
        super(MyDialog, self).__init__(parent)


        self.resize(680, 600)

        self.close = QPushButton()
        self.close.setObjectName("close")
        self.close.setText("Close")



        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)

        self.buttonBox.addButton(self.close, self.buttonBox.ActionRole)

        self.textBrowser = QTextBrowser(self)
        #self.textBrowser.append("This is a QTextBrowser!")

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)


        self.close.clicked.connect(self.close_click)
        #self.connect(self.close, SIGNAL("clicked()"),self.close_click)


    def close_click(self):
        self.reject()



# class FindingChart(QDialog):
#     def __init__(self, parent = AppForm):
#         super(FindingChart, self).__init__(parent)
#
#
#         # a figure instance to plot on
#         self.figure = plt.figure()
#
#         # this is the Canvas Widget that displays the `figure`
#         # it takes the `figure` instance as a parameter to __init__
#         self.canvas = FigureCanvas(self.figure)
#
#         # this is the Navigation widget
#         # it takes the Canvas widget and a parent
#         self.toolbar = NavigationToolbar(self.canvas, self)
#
#
#         # set the layout
#         layout = QVBoxLayout()
#         #layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)
#         self.setLayout(layout)
#
#
#
#
#     def updateme(self,img):
#
#         # create an axis
#         ax = self.figure.add_subplot(111)
#
#         # discards the old graph
#         ax.hold(False)
#
#         # plot data
#         ax.imshow(img)
#
#         # refresh canvas
#         self.canvas.draw()


def main():


    app = QApplication(sys.argv)
    form = AppForm()

    app.exec_()



if __name__ == "__main__":
    #change font to TEX
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    main()
