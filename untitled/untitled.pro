#-------------------------------------------------
#
# Project created by QtCreator 2015-11-23T18:29:40
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = untitled
TEMPLATE = app
QMAKE_CXXFLAGS += -std=c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    scribblearea.cpp

HEADERS  += mainwindow.h \
    scribblearea.h \

INCLUDEPATH += "../"

FORMS    += mainwindow.ui
