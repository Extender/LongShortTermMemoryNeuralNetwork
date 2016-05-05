QT -= core gui

TARGET = LongShortTermMemoryNeuralNetwork
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    io.cpp \
    text.cpp \
    lstm.cpp \
    lstmstate.cpp

HEADERS += \
    io.h \
    text.h \
    lstm.h \
    lstmstate.h

