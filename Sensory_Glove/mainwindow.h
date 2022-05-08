#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSerialPort>
#include <QtCore/QTimer>
#include <QTimer>
#include <QElapsedTimer>
#include <QRandomGenerator>
#include <QFile>
#include <QFileDialog>
#include <QTextStream>
#include <QMessageBox>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_szukaj_clicked();

    void on_polacz_clicked();

    void on_roalacz_clicked();

    void readFromPort();

    void on_Rozpocznij_clicked();

    void on_lineEdit_textEdited(const QString &arg1);

    void on_ActPB_valueChanged();

    void on_Zakoncz_clicked();

private:
    Ui::MainWindow *ui;

    QSerialPort *device;

    QByteArray charBuffer;

    QTimer dataTimer;

    QTimer *timer;

    bool Active=1;

    QString Data;

    double Value[16];


    QString currentFile = "";



};
#endif // MAINWINDOW_H
