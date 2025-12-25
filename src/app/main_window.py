from PySide6.QtWidgets import QMainWindow, QWidget, QButtonGroup,QMessageBox
from PySide6.QtCore import QThread,QThreadPool
import os
import QuantLib as ql
import pandas as pd

from .workers.CustomWorkers import SnapshotWorker,RateWorker,CMTWorker,EquityWorker
from .ui_pages.MainWindow import Ui_MainWindow
from .result_dialog import  ResultDialog,ResultDialogEquity


def _setup_result_worker_to_thread(thread,worker,result_dialog):
    result_dialog.show_loading_page()
    worker.finished.connect(result_dialog.show_table_page)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._setup_logic()
        self._apply_style_from_qss()
        #self.menuBar().hide()
        self.statusBar().hide()


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
        date=ql.Date.todaysDate()
        self.thread=QThread()
        self.worker=SnapshotWorker(date)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.ui.pageHome._animate_loading()
        self.worker.finished.connect(self.retrieve_mkt_data)
        self.worker.finished_msg.connect(self.ui.pageHome.setMessage)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def retrieve_mkt_data(self,dic_dataframe:dict[str:pd.DataFrame]):
        self.ui.pageHome.action_button.setEnabled(True)
        self.dic_dataframe=dic_dataframe
    
    def check_data_loaded(self):
        if not hasattr(self,'dic_dataframe'):
            QMessageBox.warning(self, "Validation error", "Please load data before pricing")

    def result_for_rate_product(self,dic_contract:dict):
        self.check_data_loaded()
        self.thread=QThread()
        self.worker=RateWorker(self.dic_dataframe,dic_contract)
        self.worker.moveToThread(self.thread)
        
        if "UF" in dic_contract['param']:
            self.thread.started.connect(self.worker.solve_coupon)
        else:
            self.thread.started.connect(self.worker.compute_price)
        self.result_dialog = ResultDialog()
        _setup_result_worker_to_thread(self.thread,self.worker,self.result_dialog)
    
    def result_for_cmt_product(self,dic_contract:dict):
        self.check_data_loaded()
        self.thread=QThread()
        self.worker=CMTWorker(self.dic_dataframe,dic_contract)
        self.worker.moveToThread(self.thread)
        if "UF" in dic_contract['param']:
            self.thread.started.connect(self.worker.solve_coupon)
        else:
            self.thread.started.connect(self.worker.compute_price)
        self.result_dialog = ResultDialog()
        _setup_result_worker_to_thread(self.thread,self.worker,self.result_dialog)

    def result_for_equity_product(self,dic_contract:dict):
        self.thread=QThread()
        self.worker=EquityWorker(dic_contract)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.compute_result)
        self.result_dialog = ResultDialogEquity()
        _setup_result_worker_to_thread(self.thread,self.worker,self.result_dialog)


    def _setup_logic(self):
        # Wire nav buttons to stacked widget pages
        self.ui.btnHome.clicked.connect(lambda: self._set_page(self.ui.pageHome))
        self.ui.btnEquity.clicked.connect(lambda: self._set_page(self.ui.pageEquity))
        self.ui.btnRate.clicked.connect(lambda: self._set_page(self.ui.pageRate))
        self.ui.btnCMT.clicked.connect(lambda: self._set_page(self.ui.pageCMT))
        self.ui.btnSpreadCMT.clicked.connect(lambda: self._set_page(self.ui.pageSpreadCMT))

        self.ui.pageHome.action_button.clicked.connect(self.load_data)
        self.ui.pageRate.submitted.connect(self.result_for_rate_product)
        self.ui.pageCMT.submitted.connect( self.result_for_cmt_product )
        self.ui.pageSpreadCMT.submitted.connect( self.result_for_cmt_product )
        self.ui.pageEquity.submitted.connect( self.result_for_equity_product )
        
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
        # update checked states handled by QButtonGroup



