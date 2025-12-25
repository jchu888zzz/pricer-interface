from PySide6.QtWidgets import (
    QWidget, QFormLayout,QComboBox,QStackedWidget,
    QDoubleSpinBox,QAbstractSpinBox,QDateEdit,QSpinBox,QHBoxLayout,
    QCheckBox,QPushButton,QSizePolicy,QLabel)

from PySide6.QtCore import QDate

class IssueDate(QDateEdit):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setCalendarPopup(True)
        self.setDate(QDate.currentDate())
        self.setObjectName("issue_date")

class InitialStrikeDate(QDateEdit):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setCalendarPopup(True)
        self.setDate(QDate.currentDate())
        self.setObjectName("initial_strike_date")

class Maturity(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("maturity")
        self.setRange(0,20)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)


class Currency(QComboBox):
    def __init__(self,dic_currency:dict[str:list]):
        super().__init__()
        self.setup_ui(dic_currency)

    def setup_ui(self,dic_currency:dict[str:list]):
        self.setObjectName("currency")
        self.addItems(list(dic_currency.keys()))


class Underlying(QComboBox):

    def __init__(self,dic_currency:dict[str:list]):
        super().__init__()
        self.setup_ui(dic_currency)
    
    def setup_ui(self,dic_currency):
        self.setObjectName("undl")
        self.dic_currency=dic_currency
        self.addItems(list(self.dic_currency.values())[0])
    
    def _display(self,cur:str):
        self.clear()
        self.addItems(self.dic_currency[cur])

class SpreadUnderlying(QWidget):

    def __init__(self,dic_cur:dict[str:list]):
        super().__init__()
        self.dic_cur=dic_cur
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("undl_spread")
        layout=QHBoxLayout()
        self.combo1=QComboBox()
        self.combo1.setObjectName("undl")
        self.label=QLabel("-")
        self.combo2=QComboBox()
        self.combo2.setObjectName("undl")
        cur=list(self.dic_cur.keys())[0]
        self.combo1.addItems(self.dic_cur[cur][0])
        self.combo2.addItems(self.dic_cur[cur][1])
        layout.addWidget(self.combo1)
        layout.addWidget(self.label)
        layout.addWidget(self.combo2)
        self.setLayout(layout)
    
    def _display(self,cur:str):
        self.combo1.clear()
        self.combo2.clear()
        self.combo1.addItems(self.dic_cur[cur][0])
        self.combo2.addItems(self.dic_cur[cur][1])
    
    def _retrieve_input(self):
        return {"underlying1":self.combo1.currentText(),
                "underlying2":self.combo2.currentText()}


class FixingOffset(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("fixing_offset")
        self.setRange(-20,0)
        self.setValue(-5)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class Frequency(QComboBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("frequency")
        self.addItems(['Annually','Semi-annually','Quarterly','Monthly'])

class FixingType(QComboBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("fixing_type")
        self.addItems(['in arrears','in advance'])


class DiffCallCalendar(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("diff_call_calendar")

class NC(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("NC")
        self.setValue(1)
        self.setRange(1,20)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class MultiCall(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("multi_call")

class FistCallDate(QDateEdit):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setCalendarPopup(True)
        self.setDate(QDate.currentDate())
        self.setObjectName("first_call_date")

class CallFrequency(QComboBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("call_frequency")
        self.addItems(['Annually','Semi-annually','Quarterly','Monthly'])

class IsCallable(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("is_callable")


class CallableWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout=QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)

        self.is_callable= IsCallable()
        layout.addRow("Is Callable :", self.is_callable)
        
        self.stack=QStackedWidget()

        page_non_call=QWidget()
        self.stack.addWidget(page_non_call)

        page_call=QWidget()
        page_call_layout=QFormLayout(page_call)
        page_call_layout.setContentsMargins(0,0,0,0)

        self.diff_calendar= DiffCallCalendar()
        page_call_layout.addRow("Different Call Calendar :", self.diff_calendar)

        self.param_stack=QStackedWidget()
        page_same=QWidget()
        page_same_layout=QFormLayout(page_same)
        page_same_layout.setContentsMargins(0,0,0,0)
        
        self.NC = NC()
        page_same_layout.addRow("NC :", self.NC)
        
        self.multicall=MultiCall()
        page_same_layout.addRow("Multi call :", self.multicall)        
        self.param_stack.addWidget(page_same)

        page_custom=QWidget()
        page_custom_layout=QFormLayout(page_custom)
        page_custom_layout.setContentsMargins(0,0,0,0)
        self.first_call_date = FistCallDate()
        page_custom_layout.addRow("First Call Date :",self.first_call_date)

        self.call_frequency = CallFrequency()
        page_custom_layout.addRow("Call Frequency :", self.call_frequency)
        self.param_stack.addWidget(page_custom)

        page_call_layout.addRow(self.param_stack)
        self.stack.addWidget(page_call)

        layout.addRow("",self.stack)
    
    def _retrieve_input(self) -> dict:

        if self.diff_calendar.isChecked():
            return {'first_call_date':self.first_call_date.date().toString("dd.MM.yyyy"),
                        'call_frequency':self.call_frequency.currentText()}
        else:
            return {"NC":str(self.NC.value()),
                    "multi-call":"true" if self.multicall.isChecked() else "false"}


class HasGuaranteedCoupon(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("has_guaranteedcoupon")

class GuaranteedCouponNumber(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setRange(0,10)
        self.setObjectName("nb_guaranteedcoupon")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class GuaranteedCouponWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout=QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)

        self.has_guaranteedcoupon= HasGuaranteedCoupon()
        layout.addRow("Has Guaranteed Coupon :", self.has_guaranteedcoupon)
        
        self.stack=QStackedWidget()

        page1=QWidget()
        self.stack.addWidget(page1)

        page2=QWidget()
        page2_layout=QFormLayout(page2)
        page2_layout.setContentsMargins(0,0,0,0)
        
        self.nb_coupon = GuaranteedCouponNumber()
        page2_layout.addRow("Number of Guaranteed Coupon :", self.nb_coupon)
        
        self.coupon=Coupon()
        page2_layout.addRow("Guaranteed Coupon :", self.coupon)        
        self.stack.addWidget(page2)

        layout.addRow(self.stack)
    
    def _retrieve_input(self) -> dict:
        if self.has_guaranteedcoupon:
            return {"nb_guaranteed_coupon":str(self.nb_coupon.value()),
                    "guaranteed_coupon": str(self.coupon.value())+'%' }
        else:
            return {}
        
class Target(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("Target")
        self.setRange(0.0,200.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class AutocallLevel(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("autocall_level")
        self.setRange(0.0,500.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class CouponLevel(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("coupon_level")
        self.setRange(0.0,500.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class MemoryEffect(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("memory_effect")

class InFine(QCheckBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("in_fine")


class LowerboundLevel(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("lowerbound_level")
        self.setRange(-10.0,10.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class UpperboundLevel(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        self.setObjectName("upperbound_level")
        self.setRange(-10.0,10.0)
        self.setDecimals(2)
        self.setValue(5.0)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

##################Solving

class StructureType(QComboBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("structure_type")
        self.addItems(['Bond','Swap'])

class SolvingComboBox(QComboBox):

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("call_frequency")
        self.addItems(['Solve coupon','Price'])

class UF(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("UF")
        self.setValue(2.0)
        self.setRange(0.0,100.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class Buffer(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("buffer")
        self.setRange(0.0,10.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

class Coupon(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("coupon")
        self.setRange(0.0,20.0)
        self.setDecimals(2)
        self.setSuffix("%")
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
