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

PAGE_CONFIG = [
        ("Home",Ui_HomePage),
        ("Equity", Ui_EquityPage),
        ("Rate", Ui_RatePage),
        ("CMT", Ui_CMTPage),
        ("Spread CMT", Ui_SpreadCMTPage)
    ]

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

        # Content area (right)
        self.content = QFrame(self.centralwidget)
        self.content.setObjectName("content")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        # Stacked pages
        self.stack = QStackedWidget(self.content)
        self.stack.setObjectName("stack")
        
        self._pages={}
        self._side_btns={}
        for name, PageClass in PAGE_CONFIG:
            page = PageClass()
            #remove space in name
            new_name="".join(name.split())
            page.setObjectName(f"page_{new_name}")
            self.stack.addWidget(page)
            self._pages[name] = page
            
            #setup  Navigation buttonns
            btn=QPushButton(name, self.sidebar)
            btn.setObjectName(f"btn_{new_name}")
            btn.setCheckable(True)
            if name=="Home":
                btn.setFlat(True)
                btn.setIcon(QIcon.fromTheme("go-home"))
                btn.setIconSize(QSize(18, 18))
            self._side_btns[name]=btn
            sidebar_layout.addWidget(btn)
            
        # Spacer to push collapse button to bottom
        spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        sidebar_layout.addItem(spacer)

        self.horizontalLayout.addWidget(self.sidebar)
        
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

    def retranslateUi(self, MainWindow):
        # keep function for compatibility if generated-style usage required later
        pass
