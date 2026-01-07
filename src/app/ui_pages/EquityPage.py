from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
import pandas as pd

from .Forms.Equity import Ui_Autocall

#path_markit_names=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\19_Quant\pricer-interface-main2\Markit_names.xlsx"
path_markit_names=r"C:\Users\jorda\OneDrive\Documents\pricer-interface-main2\Markit_names.xlsx"
df=pd.read_excel(path_markit_names)
dic_currency=dict(tuple(df.groupby('Currency')['Underlyings']))
dic_currency={key:value.to_list() for key,value in dic_currency.items()}
dic_currency={ key: dic_currency[key] for key in ['EUR','USD']}

class Ui_EquityPage(QWidget):
    """A tabbed widget containing several form tabs."""
    # re-emit
    submitted = Signal(dict)
    copy_input=Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tabsPage")
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.Equitytabs = QTabWidget(self)
        self.Equitytabs.setObjectName("EquityTabs")

        # Tab 1 - Autocall
        self.form_autocall = Ui_Autocall(dic_currency)
        self.form_autocall.setObjectName("tab_autocallEquity")
        self.Equitytabs.addTab(self.form_autocall, "Autocall")

        layout.addWidget(self.Equitytabs)

    def _connect_signals(self):
        # re-emit submitted signal with source tab info
        self.form_autocall.submitted.connect(lambda d: self._on_submitted(d, "Autocall"))
        self.form_autocall.copy_input.connect(lambda d: self._on_copy(d, "Autocall"))

    def _retrieve_param(self, input_data: dict, source_tab: str):
        # add source metadata
        param = {'param':input_data,
                "_source_tab":source_tab,
                "_source_page":"Equity"}
        return param
    
    def _on_copy(self, input_data: dict, source_tab: str):
        param=self._retrieve_param(input_data,source_tab)
        self.copy_input.emit(param)
            
    def _on_submitted(self, input_data: dict, source_tab: str):
        param=self._retrieve_param(input_data,source_tab)
        self.submitted.emit(param)