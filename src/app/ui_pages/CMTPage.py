from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .Forms.Rate import Ui_Autocall,Ui_Digit,Ui_RangeAccrual,Ui_Tarn,Ui_MinMax

DIC_CURRENCY={'EUR':['BFRTEC10','SOLDE10E','SOLBE10E','SOITA10Y','SOLIT1OE'],
            'USD':['H15T10Y']}

class Ui_CMTPage(QWidget):
    """A tabbed widget containing several form tabs."""
    submitted = Signal(dict)  # re-emit form submissions
    copy_input=Signal(dict)
    
    TABS_CONFIG = [
        ("Autocall", Ui_Autocall),
        ("Digit", Ui_Digit),
        ("RangeAccrual", Ui_RangeAccrual),
        ("Tarn", Ui_Tarn),
        ("MinMax", Ui_MinMax),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tabsPage")
        self._forms={}
        self._setup_ui()
        self._connect_signals()

    def _add_forms(self):
        for name, FormClass in self.TABS_CONFIG:
            form = FormClass(DIC_CURRENCY)
            form.setObjectName(f"tab_{name.lower()}")
            self.tabs.addTab(form, name)
            self._forms[name] = form
            
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("CMTTabs")
        
        self._add_forms()

        #Set Standard value
        # Tab 1 - Autocall
        self._forms['Autocall'].autocall_level.setValue(3.5)
        self._forms['Autocall'].coupon_level.setValue(4.5)
        # Tab 2 - Digit
        self._forms['Digit'].coupon_level.setValue(3.5)

        # Tab 3 - Range Accrual

        #Tab 4 -TARN
        self._forms['Tarn'].coupon_level.setValue(3.5)
        
        #Tab 5 -MinMax
        self._forms['MinMax'].floor.setValue(3)
        self._forms['MinMax'].floor.setValue(5)

        layout.addWidget(self.tabs)

    def _connect_signals(self):
        # re-emit submitted signal with source tab info
        for name, form in self._forms.items():
            form.submitted.connect(partial(self._on_submitted, source_tab=name))
            form.copy_input.connect(partial(self._on_copy, source_tab=name))
            
    def _retrieve_param(self, input_data: dict, source_tab: str):
        # add source metadata
        param = {'param':input_data,
                "_source_tab":source_tab,
                "_source_page":"CMT"}
        return param
    
    def _on_copy(self, input_data: dict, source_tab: str):
        param=self._retrieve_param(input_data,source_tab)
        self.copy_input.emit(param)
            
    def _on_submitted(self, input_data: dict, source_tab: str):
        param=self._retrieve_param(input_data,source_tab)
        self.submitted.emit(param)

