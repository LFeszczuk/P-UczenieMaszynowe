#include "save_to_file.h"

//save_to_File::save_to_File()
//{

//}


////! [saveToFile() function part1]
//void AddressBook::saveToFile()
//{
//    QString fileName = QFileDialog::getSaveFileName(this,
//        tr("Save Address Book"), "",
//        tr("Address Book (*.abk);;All Files (*)"));

////! [saveToFile() function part1]
////! [saveToFile() function part2]
//    if (fileName.isEmpty())
//        return;
//    else {
//        QFile file(fileName);
//        if (!file.open(QIODevice::WriteOnly)) {
//            QMessageBox::information(this, tr("Unable to open file"),
//                file.errorString());
//            return;
//        }

////! [saveToFile() function part2]
////! [saveToFile() function part3]
//        QDataStream out(&file);
//        out.setVersion(QDataStream::Qt_4_5);
//        out << contacts;
//    }
//}
////! [saveToFile() function part3]
