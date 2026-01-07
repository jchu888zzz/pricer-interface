import os

from functools import partial
import QuantLib as ql
import pandas as pd
import json

from PySide6.QtWidgets import QMainWindow, QWidget, QButtonGroup,QMessageBox,QApplication
from PySide6.QtGui import QCloseEvent

from .workers.CustomWorkers import PriceManager,MktDataManager
from .ui_pages.MainWindow import Ui_MainWindow
from .result_dialog import  ResultDialog

import Pricing.Rates.GetResults as RateGetResults
import Pricing.Equity.GetResults as EquityGetResults

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
        self.ui._pages["Home"].action_button.setEnabled(False)
        self.ui._pages["Home"]._animate_loading()
        self.data_manager.completed.connect(self.retrieve_mkt_data)
        self.data_manager.failed.connect(lambda err: self.ui._pages["Home"].setMessage(err))
        date=ql.Date.todaysDate()
        self.data_manager.retrieve(date)

    def retrieve_mkt_data(self,mkt_data:dict[str:pd.DataFrame]):
        self.ui._pages["Home"].action_button.setEnabled(True)
        self.ui._pages["Home"]._timer.stop()
        self.ui._pages["Home"].loading_label.setText("Market Data imported")
        self.mkt_data=mkt_data

    def get_result(self,input:dict):
        if not hasattr(self,'mkt_data'):
            QMessageBox.warning(self, "Validation error", "Please load data before pricing")
        PAGE_MAPPING={"Rate":RateGetResults.compute_result_rate,
                        "CMT":RateGetResults.compute_result_cmt,
                        "SpreadCMT":RateGetResults.compute_result_cmt}
        
        self.pricing_manager.add_task(f"task_{self.pricing_manager.task_count}",
                                        PAGE_MAPPING.get(input["_source_page"]),args=(self.mkt_data,input))
    
    def get_result_equity(self,input:dict):
        
        self.pricing_manager.add_task(f"task_{self.pricing_manager.task_count}",
                                        EquityGetResults.compute_result,args=(input,))

    def _dict_to_txt(self,data:dict,indent:int=0)->str:
        return json.dumps(data,indent=indent)
    
    def export_input_to_clipboard(self,input:dict):
        text=self._dict_to_txt(input)
        clipboard=QApplication.clipboard()
        clipboard.setText(text)
    
    def _setup_logic(self):
        # Wire nav buttons to stacked widget pages
        # Keep nav buttons exclusive
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for name, btn in self.ui._side_btns.items():
            btn.clicked.connect(partial(self._set_page, name=name))
            self._nav_group.addButton(btn)

        #Setup button for data
        self.ui._pages["Home"].action_button.clicked.connect(self.load_data)
        
        #Setup Submissions
        for name in ["Rate","CMT","Spread CMT"]:
            self.ui._pages[name].submitted.connect(self.get_result)
        self.ui._pages["Equity"].submitted.connect(self.get_result_equity)
        
        #Setup Input
        for name in ["Rate","CMT","Spread CMT","Equity"]:
            self.ui._pages[name].copy_input.connect(self.export_input_to_clipboard)

        # Exit action
        self.ui.actionExit.triggered.connect(self.close)
        # expose a simple main action button behavior if needed
        # maintain initial page
        self._set_page("Home")

    def _set_page(self, name:str):
        page=self.ui._pages[name]
        self.ui.stack.setCurrentWidget(page)
        
    def closeEvent(self, event: QCloseEvent):
        """Clean up threads and workers before closing application."""
        # Stop pricing manager
        if hasattr(self, 'pricing_manager') and self.pricing_manager is not None:
            try:
                self.pricing_manager.stop()
            except (RuntimeError, AttributeError):
                pass

        # Stop data manager thread
        if hasattr(self, 'data_manager') and self.data_manager is not None:
            try:
                thread = getattr(self.data_manager, 'thread', None)
                if thread is not None and thread.isRunning():
                    thread.quit()
                    if not thread.wait(2000):
                        thread.terminate()
                        thread.wait(500)
            except (RuntimeError, AttributeError):
                pass

        event.accept()


