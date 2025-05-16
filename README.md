
# Projekt: Estymacja orientacji minucji przy użyciu gradientów pikseli

## 📌 Sformułowanie problemu

Celem projektu jest estymacja kierunku lokalnego przebiegu grzbietów linii papilarnych w obrazie odcisku palca. Wiedza o orientacji grzbietów w danym obszarze ma kluczowe znaczenie dla dalszych etapów przetwarzania biometrycznego, takich jak ekstrakcja minucji, wyrównanie odcisków czy ich porównanie.

## ⚙️ Zastosowana metoda

1. **Wczytanie obrazu odcisku palca** – obraz w skali szarości.
2. **Filtracja Gaussa** – redukcja szumu przed obliczeniem gradientów .
3. **Obliczenie gradientów** intensywności w kierunkach X i Y za pomocą filtrów Sobela.
4. **Estymacja orientacji**:
   \[
   \theta(x, y) = \arctan2(G_y(x, y), G_x(x, y))
   \]
5. **Uśrednianie lokalne** – podział obrazu na bloki (np. 16×16 pikseli) i wyznaczenie średniej orientacji w każdym z nich.
6. **Wizualizacja wyników** – narysowanie orientacji jako pole wektorowe (quiver plot).
7. **Porównanie orientacji** przed i po filtracji Gaussa.

## 💻 Implementacja

### Wymagania
- Python 3
- OpenCV
- NumPy
- Matplotlib

## Możliwe rozszerzenia projektu
- Wyznaczenie orientacji tylko w punktach minucji.
- Porównanie operatorów Sobela / Prewitta / Scharra.
- Użycie kierunku do rotacyjnej normalizacji odcisków.
- Zastosowanie w wyrównywaniu wzorców linii papilarnych.


### Uruchomienie

```bash
pip install opencv-python numpy matplotlib
python orientacja_minucji.py
