#include <iostream>
#include <cstdlib>
#include "SCServo.h"
#include <unistd.h> // usleep

SMS_STS sm_st;

int main(int argc, char **argv)
{
    if(argc < 3){
        std::cout << "Użycie: " << argv[0] << " <port> <kąt[0-360]>\n";
        std::cout << "Przykład: " << argv[0] << " /dev/ttyACM0 90\n";
        return 0;
    }

    const char* port = argv[1];
    double angle = atof(argv[2]);

    if(angle < 0.0 || angle > 360.0){
        std::cerr << "Błąd: kąt musi być w zakresie 0–360 stopni!\n";
        return 0;
    }

    // Przeliczenie kąta na jednostki serwa (0–4095)
    int position = static_cast<int>(angle * 4095.0 / 360.0);
    int speed = 1500; // maks. prędkość (kroki/s)
    int accel = 15;   // przyspieszenie

    std::cout << "Łączenie z portem: " << port << "\n";
    if(!sm_st.begin(1000000, port)){
        std::cerr << "Nie udało się otworzyć portu!\n";
        return 0;
    }

    // Wysyłanie komendy do serwa ID=1
    sm_st.RegWritePosEx(1, position, speed, accel);
    sm_st.RegWriteAction(); // wykonanie ruchu
    std::cout << "Serwo ID=1 ustawia się na " << angle << "° (pozycja=" << position << ")\n";

    // Oszacowanie czasu potrzebnego na ruch (przybliżone)
    int delay_ms = static_cast<int>((position / (double)speed)*1000 + (speed / (accel*100.0))*1000);
    if(delay_ms < 300) delay_ms = 300; // minimum 300ms
    usleep(delay_ms * 1000);

    sm_st.end();
    return 1;
}
