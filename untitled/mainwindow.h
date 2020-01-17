#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <scribblearea.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    //QGraphicsScene *scene;
    //QGraphicsRectItem *rectangle;
    ScribbleArea *scribbleArea;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();
    void penWidth();
private:
    Ui::MainWindow *ui;

    void createActions();
    void createMenus();

    QMenu *optionMenu;

    QAction *penWidthAct;
    QAction *clearScreenAct;
    QAction *convert;
};

#endif // MAINWINDOW_H
