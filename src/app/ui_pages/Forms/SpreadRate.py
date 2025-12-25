from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QFormLayout,QHBoxLayout,QLabel,QMessageBox
)

from app.widgets.CustomWidgets import (IssueDate,Currency,Maturity,FixingOffset,
                                    Frequency,FixingType,AutocallLevel,CouponLevel,MemoryEffect,
                                    InFine,NC,CallableWidget,LowerboundLevel,UpperboundLevel,
                                    Target,SpreadUnderlying
                                    )


from app.ui_pages.Forms.SolvingChoice import Ui_SolvingForm
from typing import Union

def setup_widget_with_suffix(item,suffix:str):
    sub_layout=QHBoxLayout()
    sub_layout.setContentsMargins(0,0,0,0)
    sub_layout.setSpacing(5)
    sub_layout.addWidget(item)
    sub_layout.addWidget(QLabel(suffix))
    widget=QWidget()
    widget.setLayout(sub_layout)
    return widget

def bool_to_str(b:bool):
    if b :
        return "true"
    else:
        return "false"
    
class Ui_Autocall(QWidget):
    """Form for Autocall"""
    submitted = Signal(dict)

    def __init__(self,dic_currency:dict[str:str]):
        """ dic of undl per currency"""
        super().__init__()
        self.dic_currency=dic_currency
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("formAutocall")
        layout = QHBoxLayout(self)
        # layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(50)

        layout1 = QFormLayout()
        layout1.setContentsMargins(20, 12, 20, 12)
        layout1.setSpacing(12)
        # Fields        
        self.currency=Currency(self.dic_currency)
        layout1.addRow("Currency :",self.currency)

        self.undl=SpreadUnderlying(self.dic_currency)
        layout1.addRow("Underlying :",self.undl)

        layout.addLayout(layout1)

        self.issue_date=IssueDate()
        layout1.addRow("Issue Date :",self.issue_date)

        self.maturity=Maturity()
        self.maturity.setValue(10)
        layout1.addRow("Maturity (in years) :",self.maturity)

        self.fixing_offset = FixingOffset()
        self.fixing_offset.setValue(-5)
        layout1.addRow("Fixing Days Offset :", self.fixing_offset)

        self.frequency = Frequency()
        layout1.addRow("Frequency :", self.frequency)

        self.fixing_type = FixingType()
        layout1.addRow("Fixing Type :", self.fixing_type)

        self.NC = NC()
        layout1.addRow("NC :", self.NC)

        self.autocall_level = AutocallLevel()
        layout1.addRow("Autocall Level :", self.autocall_level)

        self.coupon_level = CouponLevel()
        layout1.addRow("Coupon Level :", self.coupon_level)

        self.memory_effect = MemoryEffect()
        layout1.addRow("Memory effect :", self.memory_effect)

        self.in_fine = InFine()
        layout1.addRow("In fine :", self.in_fine)

        layout2=QHBoxLayout()
        
        self.solving_layout=Ui_SolvingForm()
        layout2.addLayout(self.solving_layout)
        layout.addLayout(layout2)
        self._setup_logic()

    def _setup_logic(self):
        #Create connections to change based on currency
        self.currency.currentTextChanged.connect(self.undl._display)
        #Create connections for solving_choice
        self.solving_layout.choice.currentTextChanged.connect(self.solving_layout._display)
        # Connect submit
        self.solving_layout.submit_btn.clicked.connect(self._on_submit)

    def _validate(self) -> Union[bool, str]:
        """Validate required fields and logical constraints. Returns (ok, message)."""
        if not self.issue_date.date():
            return False, "Issue Date is required."
        if not self.maturity.value():
            return False, "Maturity is required"
        
        return True, ""

    def _on_submit(self):
        ok, msg = self._validate()
        if not ok:
            QMessageBox.warning(self, "Validation error", msg)
            return
        cur=self.currency.currentText()

        param = {
            "currency":cur,
            "issue_date": self.issue_date.date().toString("dd.MM.yyyy"),
            "maturity": self.maturity.text(),
            "fixing_days_offset": str(self.fixing_offset.value()),
            "frequency": self.frequency.currentText(),
            "fixing_type":self.fixing_type.currentText(),
            "NC":str(self.NC.value()),
            "autocall_level": str(self.autocall_level.value())+'%',
            "coupon_level": str(self.coupon_level.value())+'%',
            "memory_effect":bool_to_str(self.memory_effect.isChecked()),
            "in-fine":bool_to_str(self.in_fine.isChecked())
        }
        
        undl_data=self.undl._retrieve_input()
        param.update(undl_data)

        solving_data=self.solving_layout._retrieve_input()
        param.update(solving_data)

        res={'contract_type':'Autocall',
            'param':param}
        # Emit structured data and show brief confirmation
        self.submitted.emit(res)

class Ui_Tarn(QWidget):
    """Form for TARN"""
    submitted = Signal(dict)

    def __init__(self,dic_currency:dict[str:str]):
        """ dic of undl per currency"""
        super().__init__()
        self.dic_currency=dic_currency
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("formAutocall")
        layout = QHBoxLayout(self)
        # layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(50)

        layout1 = QFormLayout()
        layout1.setContentsMargins(20, 12, 20, 12)
        layout1.setSpacing(12)
        # Fields
        
        self.currency=Currency(self.dic_currency)
        layout1.addRow("Currency :",self.currency)

        self.undl=SpreadUnderlying(self.dic_currency)
        layout1.addRow("Underlying :",self.undl)

        layout.addLayout(layout1)

        self.issue_date=IssueDate()
        layout1.addRow("Issue Date :",self.issue_date)

        self.maturity=Maturity()
        layout1.addRow("Maturity (in years) :",self.maturity)

        self.fixing_offset = FixingOffset()
        layout1.addRow("Fixing Days Offset :", self.fixing_offset)

        self.frequency = Frequency()
        layout1.addRow("Frequency :", self.frequency)

        self.fixing_type = FixingType()
        layout1.addRow("Fixing Type :", self.fixing_type)

        self.target = Target()
        layout1.addRow("Target (nb of coupons) :", self.target)

        self.coupon_level = CouponLevel()
        layout1.addRow("Coupon Level :", self.coupon_level)

        self.in_fine = InFine()
        layout1.addRow("In fine :", self.in_fine)

        layout2=QHBoxLayout()
        
        self.solving_layout=Ui_SolvingForm()
        layout2.addLayout(self.solving_layout)
        layout.addLayout(layout2)
        self._setup_logic()

    def _setup_logic(self):
        #Create connections to change based on currency
        self.currency.currentTextChanged.connect(self.undl._display)
        #Create connections for solving_choice
        self.solving_layout.choice.currentTextChanged.connect(self.solving_layout._display)
        # Connect submit
        self.solving_layout.submit_btn.clicked.connect(self._on_submit)

    def _validate(self) -> Union[bool, str]:
        """Validate required fields and logical constraints. Returns (ok, message)."""
        if not self.issue_date.date():
            return False, "Issue Date is required."
        if not self.maturity.value():
            return False, "Maturity is required"
        
        if self.target.value() > self.maturity.value():
            return False, "Target must be inferior to maturity"
        
        return True, ""

    def _on_submit(self):
        ok, msg = self._validate()
        if not ok:
            QMessageBox.warning(self, "Validation error", msg)
            return
        cur=self.currency.currentText()

        param = {
            "currency":cur,
            "issue_date": self.issue_date.date().toString("dd.MM.yyyy"),
            "maturity": self.maturity.text(),
            "fixing_days_offset": str(self.fixing_offset.value()),
            "frequency": self.frequency.currentText(),
            "fixing_type":self.fixing_type.currentText(),
            "target":str(self.target.value()),
            "coupon_level": str(self.coupon_level.value())+'%',
            "memory_effect":"false",
            "in-fine":bool_to_str(self.in_fine.isChecked())
        }
        
        undl_data=self.undl._retrieve_input()
        param.update(undl_data)

        solving_data=self.solving_layout._retrieve_input()
        param.update(solving_data)

        res={'contract_type':'TARN',
            'param':param}
        # Emit structured data and show brief confirmation
        self.submitted.emit(res)


class Ui_Digit(QWidget):
    submitted = Signal(dict)
    sub1=Signal()

    def __init__(self,dic_currency:dict[str:str]):
        """ dic of undl per currency"""
        super().__init__()
        self.dic_currency=dic_currency
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("formAutocall")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(50)

        # Fields
        
        layout1 = QFormLayout()
        layout1.setContentsMargins(20, 12, 20, 12)
        layout1.setSpacing(12)

        self.currency=Currency(self.dic_currency)
        layout1.addRow("Currency :",self.currency)

        self.undl=SpreadUnderlying(self.dic_currency)
        layout1.addRow("Spread Underlying :",self.undl)

        self.issue_date=IssueDate()
        layout1.addRow("Issue Date :",self.issue_date)

        self.maturity=Maturity()
        layout1.addRow("Maturity (in years) :", self.maturity)

        self.fixing_offset = FixingOffset()
        layout1.addRow("Fixing Days Offset :", self.fixing_offset)

        self.frequency = Frequency()
        layout1.addRow("Frequency :", self.frequency)

        self.fixing_type = FixingType()
        layout1.addRow("Fixing Type :", self.fixing_type)

        self.coupon_level = CouponLevel()
        layout1.addRow("Coupon Level :", self.coupon_level)

        self.memory_effect = MemoryEffect()
        layout1.addRow("Memory effect :", self.memory_effect)

        self.in_fine = InFine()
        layout1.addRow("In fine :", self.in_fine)

        self.callable_widget = CallableWidget()
        layout1.addRow(self.callable_widget)

        layout.addLayout(layout1)

        self.solving_layout=Ui_SolvingForm()
        layout.addLayout(self.solving_layout)
        self._setup_logic()

    def _setup_logic(self):
        #Create connections to change based on currency
        self.currency.currentTextChanged.connect(self.undl._display)
        #Create connections for solving_choice
        self.solving_layout.choice.currentTextChanged.connect(self.solving_layout._display)
        # checkbox toggles visibility of specific calendar
        self.callable_widget.is_callable.toggled.connect(self.callable_widget.stack.setCurrentIndex)
        self.callable_widget.diff_calendar.toggled.connect(self.callable_widget.param_stack.setCurrentIndex)
        # Connect submit
        self.solving_layout.submit_btn.clicked.connect(self._on_submit)

    def _validate(self) -> Union[bool, str]:
        """Validate required fields and logical constraints. Returns (ok, message)."""
        if not self.issue_date.date():
            return False, "Issue Date is required."
        if not self.maturity.value():
            return False, "Maturity is required"

        if self.callable_widget.diff_calendar.isChecked() :
            if not  self.issue_date.date() < self.callable_widget.first_call_date.date() :
                return False, "First call date cannot be before start date."
        
        return True, ""

    def _on_submit(self):
        ok, msg = self._validate()
        if not ok:
            QMessageBox.warning(self, "Validation error", msg)
            return
        cur=self.currency.currentText()

        param = {
            "currency":cur,
            "issue_date": self.issue_date.date().toString("dd.MM.yyyy"),
            "maturity": self.maturity.text(),
            "fixing_days_offset": str(self.fixing_offset.value()),
            "frequency": self.frequency.currentText(),
            "fixing_type":self.fixing_type.currentText(),
            "coupon_level": str(self.coupon_level.value())+'%',
            "memory_effect":bool_to_str(self.memory_effect.isChecked()),
            "in-fine":bool_to_str(self.in_fine.isChecked())
        }
        
        undl_data=self.undl._retrieve_input()
        param.update(undl_data)

        solving_data=self.solving_layout._retrieve_input()
        param.update(solving_data)

        callable_data=self.callable_widget._retrieve_input()
        param.update(callable_data)
        res={'contract_type':'Digit',
            'param':param}
        # Emit structured data and show brief confirmation
        print(res['param'])
        self.submitted.emit(res)

class Ui_RangeAccrual(QWidget):
    submitted = Signal(dict)
    sub1=Signal()

    def __init__(self,dic_currency:dict[str:str]):
        """ dic of undl per currency"""
        super().__init__()
        self.dic_currency=dic_currency
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("formAutocall")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(50)

        # Fields
        
        layout1 = QFormLayout()
        layout1.setContentsMargins(20, 12, 20, 12)
        layout1.setSpacing(12)

        self.currency=Currency(self.dic_currency)
        layout1.addRow("Currency :",self.currency)

        self.undl=SpreadUnderlying(self.dic_currency)
        layout1.addRow("Spread Underlying :",self.undl)

        self.issue_date=IssueDate()
        layout1.addRow("Issue Date :",self.issue_date)

        self.maturity=Maturity()
        layout1.addRow("Maturity (in years) :", self.maturity)

        self.fixing_offset = FixingOffset()
        layout1.addRow("Fixing Days Offset :", self.fixing_offset)

        self.frequency = Frequency()
        layout1.addRow("Frequency :", self.frequency)

        self.fixing_type = FixingType()
        layout1.addRow("Fixing Type :", self.fixing_type)

        self.lowerbound_level = LowerboundLevel()
        layout1.addRow("Lower Bound Level :", self.lowerbound_level)

        self.upperbound_level = UpperboundLevel()
        layout1.addRow("Upper Bound Level :", self.upperbound_level)

        self.in_fine = InFine()
        layout1.addRow("In fine :", self.in_fine)

        self.callable_widget = CallableWidget()
        layout1.addRow(self.callable_widget)

        layout.addLayout(layout1)

        self.solving_layout=Ui_SolvingForm()
        layout.addLayout(self.solving_layout)
        self._setup_logic()

    def _setup_logic(self):
        #Create connections to change based on currency
        self.currency.currentTextChanged.connect(self.undl._display)
        #Create connections for solving_choice
        self.solving_layout.choice.currentTextChanged.connect(self.solving_layout._display)
        # checkbox toggles visibility of specific calendar
        self.callable_widget.is_callable.toggled.connect(self.callable_widget.stack.setCurrentIndex)
        self.callable_widget.diff_calendar.toggled.connect(self.callable_widget.param_stack.setCurrentIndex)
        # Connect submit
        self.solving_layout.submit_btn.clicked.connect(self._on_submit)

    def _validate(self) -> Union[bool, str]:
        """Validate required fields and logical constraints. Returns (ok, message)."""
        if not self.issue_date.date():
            return False, "Issue Date is required."
        if not self.maturity.value():
            return False, "Maturity is required"
        
        if self.upperbound_level.value() <= self.lowerbound_level.value():
            return False, "Upper level must be superior to Lower level"

        if self.callable_widget.diff_calendar.isChecked() :
            if not  self.issue_date.date() < self.callable_widget.first_call_date.date() :
                return False, "First call date cannot be before start date."
        
        return True, ""

    def _on_submit(self):
        ok, msg = self._validate()
        if not ok:
            QMessageBox.warning(self, "Validation error", msg)
            return
        cur=self.currency.currentText()
        param = {
            "currency":cur,
            "issue_date": self.issue_date.date().toString("dd.MM.yyyy"),
            "maturity": self.maturity.text(),
            "fixing_days_offset": str(self.fixing_offset.value()),
            "frequency": self.frequency.currentText(),
            "fixing_type":self.fixing_type.currentText(),
            "lower_bound": str(self.lowerbound_level.value())+'%',
            "upper_bound":str(self.upperbound_level.value())+'%',
            "in-fine":bool_to_str(self.in_fine.isChecked())
        }
        
        undl_data=self.undl._retrieve_input()
        param.update(undl_data)

        solving_data=self.solving_layout._retrieve_input()
        param.update(solving_data)

        callable_data=self.callable_widget._retrieve_input()
        param.update(callable_data)
        res={'contract_type':'RangeAccrual',
            'param':param}
        # Emit structured data and show brief confirmation
        self.submitted.emit(res)