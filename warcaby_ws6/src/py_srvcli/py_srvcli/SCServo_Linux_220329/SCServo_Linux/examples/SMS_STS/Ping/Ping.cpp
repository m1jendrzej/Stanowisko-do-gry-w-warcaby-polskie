/*
Ping指令测试,测试总线上相应ID舵机是否就绪,广播指令只适用于总线只有一个舵机情况
*/

/*
Servo scan example for ST/SC servos on UART bus.
Testuje wszystkie ID od 0 do 253 i wypisuje serwa, które odpowiadają.
*/

#include <iostream>
#include "SCServo.h"

SMS_STS sm_st;

int main(int argc, char **argv)
{
    if(argc < 2){
        std::cout << "Użycie: " << argv[0] << " <serial_port>" << std::endl;
        std::cout << "Przykład: " << argv[0] << " /dev/serial0" << std::endl;
        return 0;
    }

    std::cout << "Otwieranie portu: " << argv[1] << std::endl;

    // Spróbuj połączyć z prędkością 115200
    if(!sm_st.begin(1000000, argv[1])){
        std::cout << "Nie udało się otworzyć portu!" << std::endl;
        return 0;
    }

    std::cout << "Skanuję magistralę (ID 0–253)..." << std::endl;

    int found = 0;
    for(int id = 0; id <= 253; id++){
        int result = sm_st.Ping(id);
        if(result != -1){
            std::cout << "Znaleziono serwo! ID = " << result << std::endl;
            found++;
        }
    }

    if(found == 0){
        std::cout << "Nie znaleziono żadnych serw." << std::endl;
    } else {
        std::cout << "Łącznie znaleziono: " << found << " serw." << std::endl;
    }

    sm_st.end();
    return 0;
}
