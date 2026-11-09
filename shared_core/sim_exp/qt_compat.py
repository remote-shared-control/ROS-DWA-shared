"""Prefer Ubuntu's PyQt5; allow the official PySide6 binding on newer hosts."""

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    QT_BINDING = "PyQt5"
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
    QtCore.pyqtSignal = QtCore.Signal
    QT_BINDING = "PySide6"
