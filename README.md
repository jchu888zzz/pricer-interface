# My PySide6 Project

This project is a PySide6 application that demonstrates the use of a main window with custom widgets and styles. Below are the details regarding the project structure, setup instructions, and usage.

## Project Structure

```
my-pyside6-project
├── src
│   ├── main.py                # Entry point of the application
│   └── app
│       ├── __init__.py        # App package initialization
│       ├── main_window.py      # Main window definition
│       ├── ui_pages
│       │   ├── main_window.ui  # UI layout designed with Qt Designer
│       │   └── widgets.py      # Custom widget classes
│       ├── resources
│       │   └── resources.qrc   # Resource file for images/icons
│       └── styles
│           └── app.qss        # Application stylesheet
├── tests
│   └── test_main.py           # Unit tests for the application
├── pyproject.toml             # Project configuration
├── requirements.txt           # Required Python packages
├── .gitignore                 # Files to ignore in version control
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-pyside6-project
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one and install the required packages using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Compile resources:**
   Compile the resource file to a Python module using the following command:
   ```
   pyside6-rcc src/app/resources/resources.qrc -o src/app/resources/resources_rc.py
   ```

4. **Run the application:**
   Start the application by executing:
   ```
   python src/main.py
   ```

## Usage

Once the application is running, you will see the main window with the defined layout and custom widgets. You can interact with the widgets as per the functionality implemented in the `main_window.py` and `widgets.py` files.

## Testing

To run the unit tests, use the following command:
```
pytest tests/test_main.py
```

## Contributing

Feel free to submit issues or pull requests for any improvements or bug fixes. 

## License

This project is licensed under the MIT License. See the LICENSE file for more details.