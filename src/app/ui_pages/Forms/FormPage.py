from PySide6.QtCore import Qt, Signal, QDate, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QTextEdit, QDateEdit,
    QDoubleSpinBox, QSpinBox, QPushButton, QHBoxLayout,QVBoxLayout,
    QMessageBox, QSizePolicy
)

from typing import Union

class Ui_FormPage(QWidget):
    """User-experience focused form page: text, dates and float fields."""
    submitted = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("formPage")
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 12, 20, 12)
        main_layout.setSpacing(12)

        layout=QFormLayout()
        # Text fields
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Full name (required)")
        self.name_edit.setObjectName("nameField")
        self.name_edit.setMinimumWidth(280)
        layout.addRow("Name:", self.name_edit)

        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("email@example.com (required)")
        self.email_edit.setObjectName("emailField")
        email_re = QRegularExpression(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
        self.email_edit.setValidator(QRegularExpressionValidator(email_re, self))
        layout.addRow("Email:", self.email_edit)

        # Multi-line description
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Short description or notes (optional)")
        self.desc_edit.setFixedHeight(100)
        self.desc_edit.setObjectName("descriptionField")
        layout.addRow("Description:", self.desc_edit)

        # Date fields (clear semantics, accessible)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate())
        self.start_date.setObjectName("startDate")
        layout.addRow("Start date:", self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate().addDays(7))
        self.end_date.setObjectName("endDate")
        layout.addRow("End date:", self.end_date)

        # Numeric fields
        self.amount = QDoubleSpinBox()
        self.amount.setObjectName("amountField")
        self.amount.setRange(0.0, 1_000_000.0)
        self.amount.setDecimals(2)
        self.amount.setSingleStep(1.0)
        self.amount.setPrefix("$ ")
        layout.addRow("Amount:", self.amount)

        self.quantity = QSpinBox()
        self.quantity.setObjectName("quantityField")
        self.quantity.setRange(0, 10000)
        layout.addRow("Quantity:", self.quantity)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setObjectName("pagePrimary")  # matches QSS primary style
        self.submit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.submit_btn.clicked.connect(self._on_submit)
        btn_layout.addWidget(self.submit_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("pageSecondary")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.reset_btn)

        layout.addRow("", btn_layout)

        layout1=QFormLayout()
        # Text fields
        self.name_edit1 = QLineEdit()
        self.name_edit1.setPlaceholderText("Full name (required)")
        self.name_edit1.setObjectName("nameField")
        self.name_edit1.setMinimumWidth(280)
        layout1.addRow("Name:", self.name_edit1)

        main_layout.addLayout(layout)
        main_layout.addLayout(layout1)


    def _validate(self) -> Union[bool, str]:
        """Validate required fields and logical constraints. Returns (ok, message)."""
        if not self.name_edit.text().strip():
            return False, "Name is required."
        if not self.email_edit.hasAcceptableInput() or not self.email_edit.text().strip():
            return False, "Please enter a valid email address."
        if self.end_date.date() < self.start_date.date():
            return False, "End date cannot be before start date."
        return True, ""

    def _on_submit(self):
        ok, msg = self._validate()
        if not ok:
            QMessageBox.warning(self, "Validation error", msg)
            return
        data = {
            "name": self.name_edit.text().strip(),
            "email": self.email_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            "start_date": self.start_date.date().toPython(),
            "end_date": self.end_date.date().toPython(),
            "amount": float(self.amount.value()),
            "quantity": int(self.quantity.value())
        }
        # Emit structured data and show brief confirmation
        self.submitted.emit(data)
        QMessageBox.information(self, "Submitted", "Form submitted successfully.")

    def _on_reset(self):
        self.name_edit.clear()
        self.email_edit.clear()
        self.desc_edit.clear()
        self.start_date.setDate(QDate.currentDate())
        self.end_date.setDate(QDate.currentDate().addDays(7))
        self.amount.setValue(0.0)
        self.quantity.setValue(0)