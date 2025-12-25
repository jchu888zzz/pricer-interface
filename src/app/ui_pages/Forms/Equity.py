from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QFormLayout,QHBoxLayout,QLabel,QMessageBox
)

from app.widgets.CustomWidgets import (IssueDate,Currency,Maturity,FixingOffset,
                                    Frequency,AutocallLevel,CouponLevel,MemoryEffect,
                                    InFine,Underlying,NC,InitialStrikeDate
                                    )


from app.ui_pages.Forms.SolvingChoice import Ui_SolvingFormEquity
from typing import Union


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

        self.undl=Underlying(self.dic_currency)
        layout1.addRow("Underlying :",self.undl)

        layout.addLayout(layout1)

        self.issue_date=IssueDate()
        layout1.addRow("Issue Date:",self.issue_date)

        self.initial_strike_date=InitialStrikeDate()
        layout1.addRow("Initial Strike Date:",self.initial_strike_date)

        self.maturity=Maturity()
        self.maturity.setValue(8)
        layout1.addRow("Maturity (in years) :",self.maturity)

        self.fixing_offset = FixingOffset()
        self.fixing_offset.setValue(-5)
        layout1.addRow("Fixing Days Offset :", self.fixing_offset)

        self.frequency = Frequency()
        layout1.addRow("Frequency :", self.frequency)

        self.NC = NC()
        layout1.addRow("NC :", self.NC)

        self.autocall_level = AutocallLevel()
        self.autocall_level.setValue(100)
        layout1.addRow("Autocall Level :", self.autocall_level)


        layout2=QHBoxLayout()
        
        self.solving_layout=Ui_SolvingFormEquity()
        layout2.addLayout(self.solving_layout)
        layout.addLayout(layout2)
        self._setup_logic()

    def _setup_logic(self):
        #Create connections to change based on currency
        self.currency.currentTextChanged.connect(self.undl._display)
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
            "initial_strike_date":self.initial_strike_date.date().toString("dd.MM.yyyy"),
            "maturity": self.maturity.text(),
            "fixing_offset": str(self.fixing_offset.value()),
            "frequency": self.frequency.currentText(),
            "periods_no_autocall":str(self.NC.value()),
            "autocall_level": str(self.autocall_level.value()),
            "underlying":self.undl.currentText()
        }
        
        # Emit structured data and show brief confirmation
        self.submitted.emit(param)

