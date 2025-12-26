from PySide6.QtWidgets import (
    QWidget, QVBoxLayout,QHBoxLayout,QTableWidget, QTableWidgetItem,QLabel,QFormLayout,QApplication,
    QHeaderView,QPushButton,QTreeWidget,QTreeWidgetItem, QSizePolicy
)
from PySide6.QtCore import Qt
import json


from Pricing.Utilities import Display,Dates

class Ui_ResultPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)

        layout1=QFormLayout()

        self.label_price=QLabel()
        self.label_price.setObjectName('label_uf')
        layout1.addRow("Price :",self.label_price)

        self.label_coupon=QLabel()
        self.label_coupon.setObjectName('label_coupon')
        layout1.addRow("Coupon :",self.label_coupon)

        self.label_duration=QLabel()
        self.label_duration.setObjectName('label_duration')
        layout1.addRow("Duration:",self.label_duration)

        self.label_spread=QLabel()
        self.label_spread.setObjectName('label_spread')
        layout1.addRow("Funding spread :",self.label_spread)

        layout.addLayout(layout1)

        self.table = QTableWidget()
        self.table.setObjectName("resultTable")
        layout.addWidget(self.table)

        btn_copy=QPushButton("Copy Table")
        btn_copy.setObjectName("copyTable")
        btn_copy.clicked.connect(self.copy_table)
        layout.addWidget(btn_copy)
    
    def set_data(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("set_data expects a dict")
        self.label_price.setText(str(Display.truncate(data['Price']*100,2)) +'%')
        self.label_coupon.setText(str(Display.truncate(data['coupon']*100,2)) +'%')
        self.label_duration.setText(str(Display.truncate(data['duration'],2)))
        self.label_spread.setText(str(Display.truncate(data['funding_spread']*10000,2)) +'bps')
        
        n=len(data['Payment Dates'])
        self.table.setRowCount(n)

        lines=[]

        if "Model Forward" in data.keys():
            headers=["Payment Dates","Model Forward","Proba","Zero Coupon"]
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(headers)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)

            lines.append('\t'.join(headers))
            for i in range(n):
                date=Dates.ql_to_string(data['Payment Dates'][i],format="%d/%m/%Y")
                self.table.setItem(i,0,QTableWidgetItem(date))
                fwd=data['Model Forward'][i]
                if fwd <=1:
                    self.table.setItem(i,1,QTableWidgetItem(str(Display.truncate(fwd*100,2))+'%'))
                else:
                    self.table.setItem(i,1,QTableWidgetItem(str(Display.truncate(fwd,2))))
                proba=data['Early Redemption Proba'][i]
                self.table.setItem(i,2,QTableWidgetItem(str(Display.truncate(proba*100,2))+'%'))
                ZC=data['Zero Coupon'][i]
                self.table.setItem(i,3,QTableWidgetItem(str(Display.truncate(ZC*100,2))+'%'))
                lines.append( '\t'.join([date,f"{fwd:.4f}",f"{proba:.4%}",f"{ZC:.4%}"]))

        else:
            headers=["Payment Dates","Proba","Zero Coupon"]
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(headers)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)

            lines.append('\t'.join(headers))
            for i in range(n):
                date=Dates.ql_to_string(data['Payment Dates'][i],format="%d/%m/%Y")
                self.table.setItem(i,0,QTableWidgetItem(date))
                proba=data['Early Redemption Proba'][i]
                self.table.setItem(i,1,QTableWidgetItem(str(Display.truncate(proba*100,2))+'%'))
                ZC=data['Zero Coupon'][i]
                self.table.setItem(i,2,QTableWidgetItem(str(Display.truncate(ZC*100,2))+'%'))
                lines.append( '\t'.join([date,f"{proba:.4%}",f"{ZC:.4%}"]))

        self.table.resizeRowsToContents()
        self.text="\n".join(lines)
    
    def copy_table(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text)

class Ui_ResultPageRate(QWidget):

    def __init__(self,title1:str="Input",title2:str="Results"):
        super().__init__()
        self.title1=title1
        self.title2=title2
    
    def setup_ui(self,input:dict,result:dict):
        
        main_layout=QVBoxLayout(self)
        main_layout.setContentsMargins(10,10,10,10)
        main_layout.setSpacing(10)

        layout=QHBoxLayout()
        layout.setSpacing(10)

        #Left Side - Input
        left_widget=QWidget()
        left_layout=QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(8)

        self.left_title=QLabel(self.title1)
        self.left_title.setObjectName("input_title")
        self.left_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.left_title)

        self.tree_input=QTreeWidget()
        self.tree_input.setObjectName("input_tree")
        self.tree_input.setHeaderLabels(["Key","Value"])
        self.tree_input.setAlternatingRowColors(True)
        left_layout.addWidget(self.tree_input)

        self.display_left(input)

        btn_copy_input=QPushButton("Copy Input")
        btn_copy_input.setObjectName("copy_input")
        btn_copy_input.clicked.connect(self.export_input_to_clipboard)
        left_layout.addWidget(btn_copy_input)

        #Right Side - Result
        right_widget=QWidget()
        self.right_layout=QVBoxLayout(right_widget)
        self.right_layout.setContentsMargins(0,0,0,0)
        self.right_layout.setSpacing(8)

        self.right_title=QLabel(self.title2)
        self.right_title.setObjectName("result_title")
        self.right_title.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.right_title)

        # Results form for labels
        self.results_form=QVBoxLayout()
        self.results_form.setContentsMargins(0,0,0,0)
        self.results_form.setSpacing(5)
        self.right_layout.addLayout(self.results_form)

        self.table = QTableWidget()
        self.table.setObjectName("resultTable")
        self.display_right(result)

        btn_copy_res=QPushButton("Copy Result")
        btn_copy_res.setObjectName("copy_result")
        btn_copy_res.clicked.connect(self.export_result_to_clipboard)
        self.right_layout.addWidget(btn_copy_res)

        layout.addWidget(left_widget, 2)
        layout.addWidget(right_widget, 3)

        main_layout.addLayout(layout)

    def retrieve_data(self,input:dict,result:dict):
        self.input=input
        self.result=self._format_data(result)
        self.setup_ui(self.input,self.result)

    def display_left(self,data:dict):

        self.tree_input.clear()
        self._fill_tree(self.tree_input,data)
        self.tree_input.resizeColumnToContents(0)
        self.tree_input.resizeColumnToContents(1)
    
    def _format_data(self,data:dict) -> dict:
        res=dict()
        if 'Price' in data.keys():
            res['Price']=Display.format_to_percent(data['Price'])
        if 'coupon' in data.keys():
            res['coupon']=Display.format_to_percent(data['coupon'])
        
        res['duration']=str(Display.truncate(data['duration'],2))
        res['funding_spread']=Display.format_to_bps(data['funding_spread'])

        dic_table=data["table"]
        res["table"]=dict()
        res["table"]["Payment Dates"]=[Display.format_ql_date(x,format="%d/%m/%Y") 
                                       for x in dic_table["Payment Dates"]]
        for key in ["Model Forward","Early Redemption Proba","Zero Coupon"]:
            if key in dic_table.keys():
                res["table"][key]=[Display.format_to_percent(x) for x in dic_table[key]]

        return res
    
    def display_right(self,data:dict):
        
        for key,value in data.items():
            if key=="table":
                continue
            if isinstance(value,str):
                # Create a horizontal layout for each label pair
                item_layout = QHBoxLayout()
                item_layout.setContentsMargins(0,0,0,0)
                item_layout.setSpacing(10)
                
                key_label = QLabel(key + ":")
                key_label.setObjectName("label_result_key")
                value_label = QLabel(str(value))
                value_label.setObjectName("label_result")
                
                item_layout.addWidget(key_label)
                item_layout.addWidget(value_label)
                item_layout.addStretch()
                self.results_form.addLayout(item_layout)

        self.table.setColumnCount(len(data["table"]))
        self.table.setHorizontalHeaderLabels(list(data["table"].keys()))
        for col in range(len(data["table"])):
            if col == len(data["table"]) - 1:
                # Last column stretches to fill remaining space
                self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
            else:
                self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)

        #All elements should be of the same size
        nb_rows=len(list(data["table"].values())[0])
        self.table.setRowCount(nb_rows)
        self.table.verticalHeader().setVisible(False)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        for col,values in enumerate(data["table"].values()):
            for row,item in enumerate(values):
                cell=QTableWidgetItem(item)
                self.table.setItem(row,col,cell)
        
        self.table.resizeRowsToContents()
        self.table.setMinimumWidth(400)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(self.table)

    def _fill_tree(self,tree:QTreeWidget,data:dict):

        for key,value in data.items():
            if isinstance(value,dict):
                # Create parent item for nested dict
                item=QTreeWidgetItem(tree,[str(key),""])
                item.setExpanded(True)

                for nested_key,nested_value in value.items():
                    QTreeWidgetItem(item,[str(nested_key),str(nested_value)])
            
            elif isinstance(value,(list,tuple)):
                item=QTreeWidgetItem(tree,[str(key),str(value)])
            else:
                QTreeWidgetItem(tree,[str(key),str(value)])
    
    def _dict_to_txt(self,data:dict,indent:int=1)->str:
        return json.dumps(data,indent=indent)
    
    def _result_to_excel_table(self,data:dict) -> str:
        lines=[]

        for key,value in data.items():
            if key=="table":
                continue
            lines.append(f"{key}\t{value}")
        
        lines.append("")
        lines.append("\t".join(list(data["table"].keys())))
        nb_rows=len(list(data["table"].values())[0])
        for i in range(nb_rows):
            lines.append("\t".join(data["table"][key][i] for key in data["table"].keys()))
        
        return "\n".join(lines)

    def _copy_to_clipboard(self,text:str):
        clipboard=QApplication.clipboard()
        clipboard.setText(text)

    def export_input_to_clipboard(self):
        text=self._dict_to_txt(self.input)
        self._copy_to_clipboard(text)
    
    def export_result_to_clipboard(self):
        text=self._result_to_excel_table(self.result)
        self._copy_to_clipboard(text)

class Ui_ResultPageEquity(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)

        layout1=QFormLayout()

        self.label_duration=QLabel()
        self.label_duration.setObjectName('label_duration')
        layout1.addRow("Duration:",self.label_duration)

        self.label_spread=QLabel()
        self.label_spread.setObjectName('label_spread')
        layout1.addRow("Funding spread :",self.label_spread)

        layout.addLayout(layout1)

        self.table = QTableWidget()
        self.table.setObjectName("resultTable")
        layout.addWidget(self.table)

        btn_copy=QPushButton("Copy Table")
        btn_copy.setObjectName("copyTable")
        btn_copy.clicked.connect(self.copy_table)
        layout.addWidget(btn_copy)
    
    def set_data(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("set_data expects a dict")
        self.label_duration.setText(str(Display.truncate(data['duration'],2)))
        self.label_spread.setText(str(Display.truncate(data['funding_spread']*10000,2)) +'bps')
        
        n=len(data['Payment Dates'])
        self.table.setRowCount(n)

        lines=[]

        headers=["Payment Dates","Forwards","Proba","Zero Coupon"]
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        lines.append('\t'.join(headers))
        for i in range(n):
            date=Dates.ql_to_string(data['Payment Dates'][i],format="%d/%m/%Y")
            self.table.setItem(i,0,QTableWidgetItem(date))
            fwd=data['Forwards'][i]
            self.table.setItem(i,1,QTableWidgetItem(str(Display.truncate(fwd,2))))
            proba=data['Early Redemption Proba'][i]
            self.table.setItem(i,2,QTableWidgetItem(str(Display.truncate(proba*100,2))+'%'))
            ZC=data['Zero Coupon'][i]
            self.table.setItem(i,3,QTableWidgetItem(str(Display.truncate(ZC*100,2))+'%'))
            lines.append( '\t'.join([date,f"{fwd:.4f}",f"{proba:.4%}",f"{ZC:.4%}"]))

        self.table.resizeRowsToContents()
        self.text="\n".join(lines)
    
    def copy_table(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text)