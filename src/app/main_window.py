from PySide6.QtWidgets import QMainWindow, QWidget, QButtonGroup,QMessageBox
from PySide6.QtGui import QCloseEvent
import os
import QuantLib as ql
import pandas as pd

from .workers.CustomWorkers import PriceManager,MktDataManager
from .ui_pages.MainWindow import Ui_MainWindow
from .result_dialog import  ResultDialog

import Pricing.Rates.GetResults as RateGetResults

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._setup_logic()
        self._apply_style_from_qss()
        #self.menuBar().hide()
        self.statusBar().hide()

        self.pricing_manager=PriceManager(dialog_class=ResultDialog)
        self.pricing_manager.start()
        self.data_manager=MktDataManager()

    def _apply_style_from_qss(self, path: str = None):
        """
        Load a .qss file and apply as the app stylesheet.
        Default path: src/app/styles/theme.qss (project-relative).        """
        if path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            path = os.path.join(project_root,"app", "styles", "theme.qss")
        try:
            with open(path, "r") as f:
                qss = f.read()
            self.setStyleSheet(qss)
            #print(f"Loaded QSS: {path}")
        except Exception as e:          
            pass
            #print(f"Failed to load QSS from {path}: {e}")

    def load_data(self):
        self.ui.pageHome.action_button.setEnabled(False)
        self.ui.pageHome._animate_loading()
        self.data_manager.completed.connect(self.retrieve_mkt_data)
        self.data_manager.failed.connect(lambda err: self.ui.pageHome.setMessage(err))
        #date=ql.Date.todaysDate()
        date=ql.Date(11,11,2025)
        self.data_manager.retrieve(date)

    def retrieve_mkt_data(self,mkt_data:dict[str:pd.DataFrame]):
        self.ui.pageHome.action_button.setEnabled(True)
        self.ui.pageHome._timer.stop()
        self.ui.pageHome.loading_label.setText("Market Data imported")
        self.mkt_data=mkt_data

    def get_result(self,input:dict):
        if not hasattr(self,'mkt_data'):
            QMessageBox.warning(self, "Validation error", "Please load data before pricing")
        PAGE_MAPPING={"Rate":RateGetResults.compute_result_rate,
                      "CMT":RateGetResults.compute_result_cmt,
                      "SpreadCMT":RateGetResults.compute_result_cmt}
        
        self.pricing_manager.add_task("task_rate",PAGE_MAPPING.get(input["_source_page"]),args=(self.mkt_data,input))
        

    def _setup_logic(self):
        # Wire nav buttons to stacked widget pages
        self.ui.btnHome.clicked.connect(lambda: self._set_page(self.ui.pageHome))
        self.ui.btnEquity.clicked.connect(lambda: self._set_page(self.ui.pageEquity))
        self.ui.btnRate.clicked.connect(lambda: self._set_page(self.ui.pageRate))
        self.ui.btnCMT.clicked.connect(lambda: self._set_page(self.ui.pageCMT))
        self.ui.btnSpreadCMT.clicked.connect(lambda: self._set_page(self.ui.pageSpreadCMT))

        self.ui.pageHome.action_button.clicked.connect(self.load_data)
        self.ui.pageRate.submitted.connect(self.get_result)
        self.ui.pageCMT.submitted.connect(self.get_result)
        self.ui.pageSpreadCMT.submitted.connect(self.get_result)

        # Keep nav buttons exclusive
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_group.addButton(self.ui.btnHome)
        self._nav_group.addButton(self.ui.btnEquity)
        self._nav_group.addButton(self.ui.btnRate)
        self._nav_group.addButton(self.ui.btnCMT)
        self._nav_group.addButton(self.ui.btnSpreadCMT)


        # Exit action
        self.ui.actionExit.triggered.connect(self.close)
        # expose a simple main action button behavior if needed
        # maintain initial page
        self._set_page(self.ui.pageHome)

    def _set_page(self, page:QWidget):
        self.ui.stack.setCurrentWidget(page)

    def closeEvent(self, event: QCloseEvent):
        """Clean up threads before closing."""
        try:
            if hasattr(self, 'pricing_manager'):
                self.pricing_manager.stop()
        except:
            pass
        
        try:
            if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'thread'):
                if self.data_manager.thread and self.data_manager.thread.isRunning():
                    self.data_manager.thread.quit()
                    self.data_manager.thread.wait()
        except:
            pass
        
        event.accept()
        # update checked states handled by QButtonGroup



