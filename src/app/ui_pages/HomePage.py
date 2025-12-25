from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt,QTimer

class Ui_HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        self.action_button = QPushButton("Load Market Data")
        self.action_button.setObjectName("homeButton")
        layout.addWidget(self.action_button)

        # Text label
        self.loading_label = QLabel("")
        self.loading_label.setObjectName("homeLabel")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        # Add spacer to push content to top
        layout.addStretch()

        # Animated dots for loading (kept for text)
        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_loading)
    
    def _animate_loading(self):
        self._timer.start(400)
        self._dots = (self._dots + 1) % 4
        #base=self.loading_label.text().split(".")[0]
        self.loading_label.setText("Loading" + "." * self._dots)
    
    def setMessage(self,text:str):
        self._timer.stop()
        self.loading_label.setText(text)