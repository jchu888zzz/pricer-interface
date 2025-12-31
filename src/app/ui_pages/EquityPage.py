from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
import pandas as pd

from .Forms.Equity import Ui_Autocall

path_markit_names=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\19_Quant\pricer-interface-main1\Markit_names.xlsx"
df=pd.read_excel(path_markit_names)
dic_currency=dict(tuple(df.groupby('Currency')['Underlyings']))
dic_currency={key:value.to_list() for key,value in dic_currency.items()}
dic_currency={ key: dic_currency[key] for key in ['EUR','USD']}

class Ui_EquityPage(QWidget):
    """A tabbed widget containing several form tabs."""
    submitted = Signal(dict)  # re-emit form submissions

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

        # Tab 1 - General
        self.form_general = Ui_Autocall(dic_currency)
        self.form_general.setObjectName("tab_autocallEquity")
        self.Equitytabs.addTab(self.form_general, "Autocall")

        # Tab 2 - Details
        # self.form_details = Ui_FormPage()
        # self.form_details.setObjectName("tab_details")
        # self.form_details.name_edit.setPlaceholderText("Full name (Details)")
        # self.form_details.email_edit.setPlaceholderText("work.email@company.com")
        # self.Equitytabs.addTab(self.form_details, "Details")

        # Tab 3 - Financial
        # self.form_financial = Ui_FormPage()
        # self.form_financial.setObjectName("tab_financial")
        # self.form_financial.name_edit.setPlaceholderText("Account holder name")
        # self.form_financial.desc_edit.setPlaceholderText("Notes about the transaction")
        # # tweak numeric widget defaults for financial tab
        # self.form_financial.amount.setDecimals(2)
        # self.form_financial.amount.setSingleStep(0.01)
        # self.Equitytabs.addTab(self.form_financial, "Financial")

        layout.addWidget(self.Equitytabs)

    def _connect_signals(self):
        # re-emit submitted signal with source tab info
        self.form_general.submitted.connect(lambda d: self._on_submitted(d, "Autocall"))
        # self.form_details.submitted.connect(lambda d: self._on_submitted(d, "Details"))
        # self.form_financial.submitted.connect(lambda d: self._on_submitted(d, "Financial"))

    def _on_submitted(self, data: dict, source_tab: str):
        # add source metadata and re-emit
        data = dict(data)
        data["_source_tab"] = source_tab
        self.submitted.emit(data)
