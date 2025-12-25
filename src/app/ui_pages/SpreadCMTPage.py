from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .Forms.SpreadRate import Ui_Autocall,Ui_Digit,Ui_RangeAccrual,Ui_Tarn

dic_currency={'EUR':(['BFRTEC10','SOLDE10E','SOLBE10E','SOITA10Y','SOLIT1OE'],['EUR ' +'CMS ' + '10Y']),
            'USD':(['H15T10Y'],['USD ' +'CMS ' + '10Y'])}

class Ui_SpreadCMTPage(QWidget):
    """A tabbed widget containing several form tabs."""
    submitted = Signal(dict)  # re-emit form submissions
    sub1=Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tabsPage")
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("SpreadCMTTabs")

        # Tab 1 - Autocall
        self.form_autocall = Ui_Autocall(dic_currency)
        self.form_autocall.setObjectName("tab_autocall")
        self.tabs.addTab(self.form_autocall, "Autocall")

        # Tab 2 - Digit
        self.form_digit = Ui_Digit(dic_currency)
        self.form_digit.setObjectName("tab_digit")
        self.tabs.addTab(self.form_digit, "Digit")

        # Tab 3 - Range Accrual
        self.form_range = Ui_RangeAccrual(dic_currency)
        self.form_range.setObjectName("tab_rangeaccrual")
        self.tabs.addTab(self.form_range, "RangeAccrual")

        #Tab 4 -TARN
        self.form_tarn = Ui_Tarn(dic_currency)
        self.form_tarn.setObjectName("tab_tarn")
        self.tabs.addTab(self.form_tarn, "TARN")

        layout.addWidget(self.tabs)

    def _connect_signals(self):
        # re-emit submitted signal with source tab info
        self.form_autocall.submitted.connect(lambda d: self._on_submitted(d, "Autocall"))
        self.form_digit.submitted.connect(lambda d: self._on_submitted(d, "Digit"))
        self.form_range.submitted.connect(lambda d: self._on_submitted(d, "RangeAccrual"))
        self.form_range.submitted.connect(lambda d: self._on_submitted(d, "TARN"))

    def _on_submitted(self, input_data: dict, source_tab: str):
        # add source metadata and re-emit
        data = {'param':input_data,
                "_source_tab":source_tab}
        self.submitted.emit(data)