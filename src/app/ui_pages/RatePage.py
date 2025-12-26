from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .Forms.SingleRate import Ui_Autocall,Ui_Digit,Ui_RangeAccrual,Ui_Tarn,Ui_FixedRate

dic_currency={'EUR':['EUR CMS ' + str(i) +'Y' for i in [5,10,15,20,25,30]]+ ['EUR Euribor ' +x for x in ['3M','12M']],
                'USD':['USD CMS ' + str(i) +'Y' for i in [5,10,15,20,25,30]]}

class Ui_RatePage(QWidget):
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

        self.Ratetabs = QTabWidget(self)
        self.Ratetabs.setObjectName("RateTabs")

        # Tab 1 - Autocall
        self.form_autocall = Ui_Autocall(dic_currency)
        self.form_autocall.setObjectName("tab_autocall")
        self.Ratetabs.addTab(self.form_autocall, "Autocall")

        # Tab 2 - Digit
        self.form_digit = Ui_Digit(dic_currency)
        self.form_digit.setObjectName("tab_digit")
        self.Ratetabs.addTab(self.form_digit, "Digit")

        # Tab 3 - Range Accrual
        self.form_range = Ui_RangeAccrual(dic_currency)
        self.form_range.setObjectName("tab_rangeaccrual")
        self.Ratetabs.addTab(self.form_range, "RangeAccrual")

        #Tab 4 -TARN
        self.form_tarn = Ui_Tarn(dic_currency)
        self.form_tarn.setObjectName("tab_tarn")
        self.Ratetabs.addTab(self.form_tarn, "TARN")

        #Tab 5 -FixedRate
        self.form_fixedrate = Ui_FixedRate(dic_currency)
        self.form_fixedrate.setObjectName("tab_fixedrate")
        self.Ratetabs.addTab(self.form_fixedrate, "FixedRate")


        layout.addWidget(self.Ratetabs)

    def _connect_signals(self):
        # re-emit submitted signal with source tab info
        self.form_autocall.submitted.connect(lambda d: self._on_submitted(d, "Autocall"))
        self.form_digit.submitted.connect(lambda d: self._on_submitted(d, "Digit"))
        self.form_range.submitted.connect(lambda d: self._on_submitted(d, "RangeAccrual"))
        self.form_tarn.submitted.connect(lambda d: self._on_submitted(d, "Tarn"))
        self.form_fixedrate.submitted.connect(lambda d: self._on_submitted(d, "FixedRate"))

    def _on_submitted(self, input_data: dict, source_tab: str):
        # add source metadata and re-emit
        data = {'param':input_data,
                "_source_tab":source_tab,
                "_source_page":"Rate"}
        self.submitted.emit(data)
        #↨self.sub1.emit()