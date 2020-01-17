#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QInputDialog>
#include "neuralnetwork.h"
#include <QMessageBox>
#include <QString>
#include "mnist.h"

SigmoidFunction sig;

NeuralNetwork *nn;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    nn = new NeuralNetwork("../Ensemble2_50000_1000_0.001000_48_5.000000__50_10", &sig);
    scribbleArea = new ScribbleArea(this);
    setCentralWidget(scribbleArea);

    MINST::readTestImages("../MINST/t10k-images.idx3-ubyte");

    createActions();
    createMenus();
}

void MainWindow::penWidth()
{
    bool ok;
    int newWidth = QInputDialog::getInt(this, tr("Scribble"),
                                        tr("Select pen width:"),
                                        scribbleArea->penWidth(),
                                        1, 50, 1, &ok);
    if (ok)
        scribbleArea->setPenWidth(newWidth);
}

void MainWindow::createActions()
{
    penWidthAct = new QAction(tr("Pen &Width..."), this);
    connect(penWidthAct, SIGNAL(triggered()), this, SLOT(penWidth()));

    clearScreenAct = new QAction(tr("&Clear Screen"), this);
    clearScreenAct->setShortcut(tr("Ctrl+L"));
    connect(clearScreenAct, SIGNAL(triggered()),scribbleArea, SLOT(clearImage()));

    convert = new QAction(tr("Detect"), this);
    connect(convert, SIGNAL(triggered()), this, SLOT(on_pushButton_clicked()));
}

void MainWindow::createMenus()
{
    optionMenu = new QMenu(tr("&Options"), this);
    optionMenu->addAction(penWidthAct);
    optionMenu->addAction(clearScreenAct);
    optionMenu->addAction(convert);

    menuBar()->addMenu(optionMenu);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete nn;
}

int getNumber(int* data)
{

    vector<double> sample;
    for (int i = 0; i < 784; i++)
    {
        sample.push_back(data[i]);
    }
    return nn->getOutputOneImage(sample);
}

void MainWindow::on_pushButton_clicked()
{
    //MINST::testImages[0];
    int hr = scribbleArea->image.height() / 28;
    int wr = scribbleArea->image.width() / 28;
    int avgImage[28*28];
    for(int i = 0;i < 784;i++)
    {
        avgImage[i] = 0;
    }
    static int index = -1;
    index++;
    for(int r = 0;r < scribbleArea->image.height();r++)
    {
        for(int c = 0;c < scribbleArea->image.width();c++)
        {
            int c1 = MINST::testImages[index][28 * (r / hr) + (c / wr)];
            int color = (c1 << 16) | (c1 << 8) | c1;
            scribbleArea->image.setPixel(c,r,color);
            QRgb px = scribbleArea->image.pixel(c,r);
            avgImage[28 * (r / hr) + (c / wr)] += (0xff & px);

            //cout << c1 << "\t" << (0xff & px) << endl;
        }
    }

    for(int i = 0;i < 784;i++)
    {
        avgImage[i] = (avgImage[i] / (hr * wr));
        /*
        if(avgImage[i] > 128)
        {
            avgImage[i] = 255;
        }
        else
        {
            avgImage[i] = 0;
        }
        */
    }

    int num = getNumber(avgImage);
    //ui->lcdNumber->display(num);
    QString Snum;
    Snum.setNum(num);
    QMessageBox qmsg(this);
    qmsg.setText(Snum);
    qmsg.exec();
}
