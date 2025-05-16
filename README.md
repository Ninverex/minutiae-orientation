# Analiza orientacji linii papilarnych na obrazie

## Sformułowanie problemu

Celem projektu jest ręczne przetworzenie obrazu przedstawiającego odcisk palca w celu analizy orientacji lokalnych linii papilarnych. Szczególnym celem jest obliczenie kierunku gradientu intensywności obrazu w blokach o stałym rozmiarze, przed i po zastosowaniu filtracji Gaussa. Jest to istotny krok m.in. w systemach rozpoznawania odcisków palców.

## Opis zastosowanej metody

1. **Wczytanie obrazu** – Obraz RGB jest konwertowany do skali szarości.
2. **Konwersja na wartości zmiennoprzecinkowe** – Wszystkie wartości pikseli są skalowane do zakresu [0,1].
3. **Obliczenie gradientów** – Przy użyciu filtrów Sobela wyznaczany jest gradient w kierunku x i y.
4. **Obliczenie orientacji** – Dla każdego bloku (np. 16x16 pikseli) obliczana jest średnia orientacja gradientów.
5. **Filtracja Gaussa** – Obraz jest rozmywany przed ponowną analizą orientacji.
6. **Wizualizacja** – Na obrazie tła nakładane są wektory orientacji jako wykres typu quiver.

## Implementacja

Projekt został zaimplementowany w języku **Python** z wykorzystaniem wyłącznie podstawowych bibliotek:
- `scikit-image` (do wczytania obrazu),
- `matplotlib` (do wizualizacji).

Cała analiza obrazu, w tym:
- konwersja RGB → gray,
- splot 2D,
- generacja jądra Gaussa,
- filtry Sobela,
- obliczenie orientacji,

została wykonana **ręcznie**, bez użycia gotowych funkcji przetwarzania obrazu z bibliotek takich jak OpenCV.

## Prezentacja wyników

Wynikiem działania programu są dwa obrazy z wykresami orientacji:
- **Przed rozmyciem Gaussa** – pokazuje lokalne kierunki linii papilarnych z oryginalnego obrazu.
- **Po rozmyciu Gaussa** – pokazuje wygładzone kierunki, co zmniejsza szum i może poprawiać analizę.

Oba wykresy pokazują nałożone wektory kierunku w regularnych blokach.  
![image](https://github.com/user-attachments/assets/8f060348-ec1d-41f7-8251-105b2c14d544)  


## Możliwe rozszerzenia/kontynuacja projektu
- Automatyczna detekcja jąder (cores) i delt – kluczowe punkty odcisków palców.
- Segmentacja obszarów tła i odcisku – oddzielenie użytecznego obszaru od tła.
- Zastosowanie bardziej zaawansowanych metod wygładzania – np. filtracja anizotropowa.
- Użycie narzędzi takich jak OpenCV, TensorFlow – w celu porównania z ręczną implementacją.
- Implementacja identyfikacji osoby na podstawie odcisku palca – połączenie z klasyfikatorami.
