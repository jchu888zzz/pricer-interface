from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon,QAction
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSpacerItem, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget
)

from .HomePage import Ui_HomePage
from .EquityPage import Ui_EquityPage
from  app.ui_pages.RatePage import Ui_RatePage
from  app.ui_pages.CMTPage import Ui_CMTPage
from  app.ui_pages.SpreadCMTPage import Ui_SpreadCMTPage

class Ui_MainWindow(object):

    def setupUi(self, MainWindow: QMainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 640)
        MainWindow.setWindowTitle("Funding Application")

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)


        # Sidebar (left)
        self.sidebar = QFrame(self.centralwidget)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setMaximumWidth(320)
        self.sidebar.setFrameShape(QFrame.NoFrame)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Header / logo
        self.logo = QLabel("App", self.sidebar)
        self.logo.setObjectName("logo")
        self.logo.setAlignment(Qt.AlignCenter)
        self.logo.setMinimumHeight(48)
        sidebar_layout.addWidget(self.logo)

        # Navigation buttons
        self.btnHome = QPushButton("Home", self.sidebar)
        self.btnHome.setObjectName("btnHome")
        self.btnHome.setCheckable(True)
        self.btnHome.setChecked(True)
        self.btnHome.setFlat(True)
        self.btnHome.setIcon(QIcon.fromTheme("go-home"))
        self.btnHome.setIconSize(QSize(18, 18))
        sidebar_layout.addWidget(self.btnHome)


        # self.btnSettings = QPushButton(" Settings", self.sidebar)
        # self.btnSettings.setObjectName("btnSettings")
        # self.btnSettings.setCheckable(True)
        # self.btnSettings.setFlat(True)
        # self.btnSettings.setIcon(QIcon.fromTheme("settings"))
        # self.btnSettings.setIconSize(QSize(18, 18))
        # sidebar_layout.addWidget(self.btnSettings)

        self.btnEquity = QPushButton("Equity", self.sidebar)
        self.btnEquity.setObjectName("btnEquity")

        self.btnRate = QPushButton("Rate", self.sidebar)
        self.btnRate.setObjectName("btnRate")

        self.btnCMT = QPushButton("CMT", self.sidebar)
        self.btnCMT.setObjectName("btnCMT")
        
        self.btnSpreadCMT = QPushButton("Spread CMT", self.sidebar)
        self.btnSpreadCMT.setObjectName("btnSpreadCMT")

        self._setup_button_side_bar(sidebar_layout,[self.btnEquity,self.btnRate,self.btnCMT,self.btnSpreadCMT])

        # Spacer to push collapse button to bottom
        spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        sidebar_layout.addItem(spacer)

        self.horizontalLayout.addWidget(self.sidebar)

        # Content area (right)
        self.content = QFrame(self.centralwidget)
        self.content.setObjectName("content")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Stacked pages
        self.stack = QStackedWidget(self.content)
        self.stack.setObjectName("stack")

        # Home page
        self.pageHome = Ui_HomePage()
        self.stack.addWidget(self.pageHome)

        # Settings page
        # self.pageSettings = QWidget()
        # settings_layout = QVBoxLayout(self.pageSettings)
        # self.settingsLabel = QLabel("Settings page content", self.pageSettings)
        # self.settingsLabel.setAlignment(Qt.AlignCenter)
        # settings_layout.addWidget(self.settingsLabel)
        # self.stack.addWidget(self.pageSettings)

        # Equity page
        self.pageEquity = Ui_EquityPage()
        self.stack.addWidget(self.pageEquity)

        #Rate page
        self.pageRate = Ui_RatePage()
        self.stack.addWidget(self.pageRate)

        #CMT page
        self.pageCMT = Ui_CMTPage()
        self.stack.addWidget(self.pageCMT)

        #CMT page
        self.pageSpreadCMT = Ui_SpreadCMTPage()
        self.stack.addWidget(self.pageSpreadCMT)

        content_layout.addWidget(self.stack)
        self.horizontalLayout.addWidget(self.content)

        MainWindow.setCentralWidget(self.centralwidget)

        # Menubar and statusbar placeholders
        self.menubar = MainWindow.menuBar()
        self.menubar.setObjectName("menubar")
        self.statusbar = MainWindow.statusBar()
        self.statusbar.setObjectName("statusbar")

        # Action (Exit)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setText("E&xit")
        MainWindow.addAction(self.actionExit)

        self.retranslateUi(MainWindow)

    def _setup_button_side_bar(self,layout,btns):
        for btn in btns:
            btn.setCheckable(True)
            layout.addWidget(btn)

    def retranslateUi(self, MainWindow):
        # keep function for compatibility if generated-style usage required later
        pass