from PySide6.QtWidgets import (QFormLayout,QPushButton,QSizePolicy,
                            QStackedWidget,QWidget,QHBoxLayout,QSpacerItem)


from app.widgets.CustomWidgets import (SolvingComboBox,StructureType,UF,Buffer,Coupon
                                )

class Ui_SolvingForm(QFormLayout):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):

        self.setContentsMargins(20, 12, 20, 12)
        self.setSpacing(10)

        # self.structure_type=StructureType()
        # self.addRow("Structure :", self.structure_type)

        self.choice=SolvingComboBox()
        self.addRow("Solving choice :",self.choice)

        self.stack=QStackedWidget()
        self.stack.setContentsMargins(0,0,0,0)
        self.page_UF=QWidget()
        page_UF_layout=QFormLayout(self.page_UF)
        page_UF_layout.setContentsMargins(0,0,0,0)
        self.UF=UF()
        page_UF_layout.addRow("Upfront :", self.UF)
        
        self.buffer = Buffer()
        page_UF_layout.addRow("Buffer (per year) :",self.buffer)

        self.stack.addWidget(self.page_UF)

        self.page_coupon=QWidget()
        page_coupon_layout=QFormLayout(self.page_coupon)
        page_coupon_layout.setContentsMargins(0,0,0,0)
        self.coupon = Coupon()
        page_coupon_layout.addRow("Coupon :", self.coupon)
        self.stack.addWidget(self.page_coupon)
        self.addRow(self.stack)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setObjectName("submit_btn")
        self.submit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        btn_layout.addWidget(self.submit_btn)
        self.addRow(btn_layout)
        
    def _display(self,char:str):
        if char=='Solve coupon':
            self.stack.setCurrentWidget(self.page_UF)
        else:
            self.stack.setCurrentWidget(self.page_coupon)
    
    def _retrieve_input(self) ->dict[str:str]:
        
        data={"solving_choice":self.choice.currentText()}
        if self.choice.currentText()=="Solve coupon":
            data.update({"UF":str(self.UF.value())+'%',
                    "yearly_buffer":str(self.buffer.value())+'%',
                    "structure_type":"Swap"})
            return data
        else:
            data.update({"coupon":str(self.coupon.value())+'%',
                        "structure_type":"Bond"})
            return data


class Ui_PricingForm(QFormLayout):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):

        self.setContentsMargins(20, 12, 20, 12)
        self.setSpacing(10)

        # self.structure_type=StructureType()
        # self.addRow("Structure :", self.structure_type)

        btn_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Price")
        self.submit_btn.setObjectName("submit_btn")
        self.submit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        btn_layout.addWidget(self.submit_btn)
        self.addRow(btn_layout)
    
    def _retrieve_input(self) ->dict[str:str]:
        return {"solving_choice":"Price",
              "structure_type":"Bond"}

class Ui_SolvingFormEquity(QFormLayout):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):

        self.setContentsMargins(20, 12, 20, 12)
        self.setSpacing(10)

        self.structure_type=StructureType()
        self.addRow("Structure :", self.structure_type)
        # Action buttons
        btn_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Find Funding")
        self.submit_btn.setObjectName("submit_btn")
        self.submit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        btn_layout.addWidget(self.submit_btn)
        self.addRow(btn_layout)
        