import sys
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow
import os

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    # Use current directory as default
    current_dir = os.getcwd()
    
    # Create and show main window with current directory
    window = MainWindow(None)  # Pass None since we don't require CSV file now
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()