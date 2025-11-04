# Erklärung für Einsteiger: Machine Learning mit unserem Skript

Hallo! Lass uns die Dinge, die wir gerade gemacht haben, ganz einfach aufschlüsseln. Stell dir vor, wir bringen einem Computer bei, eine bestimmte Aufgabe zu lernen.

## 1. Was ist ein "Modell"?

Ein "Modell" im Machine Learning ist wie ein Gehirn, das wir trainieren. Wir zeigen ihm eine Menge Beispiele, und es lernt, Muster in diesen Beispielen zu erkennen. Nachdem es trainiert wurde, kann es Vorhersagen über neue Daten machen, die es noch nie gesehen hat.

**Analogie:** Stell dir vor, du bringst einem Kind bei, Äpfel von Birnen zu unterscheiden. Du zeigst ihm viele Bilder von Äpfeln und Birnen (das sind die **Trainingsdaten**). Das "Modell" ist das Wissen im Gehirn des Kindes, das es ihm ermöglicht, zu sagen: "Das ist ein Apfel" oder "Das ist eine Birne", wenn es ein neues Bild sieht.

In unserem Fall ist die Aufgabe, vorherzusagen, ob eine Person eine Herzkrankheit hat oder nicht.

## 2. Die Modelle, die wir trainiert haben

Wir haben zwei verschiedene "Gehirne" (Modelle) trainiert, um das Problem der Herzkrankheiten zu lösen.

### a) Logistische Regression (Logistic Regression)

Das ist ein relativ einfaches, aber oft sehr effektives Modell. Es versucht, eine Linie (oder in komplexeren Fällen eine Kurve) zu finden, die die Datenpunkte in zwei Gruppen trennt. In unserem Fall: "hat Herzkrankheit" und "hat keine Herzkrankheit".

#### Was bedeutet der `C`-Wert?

Der `C`-Wert ist ein sogenannter **Hyperparameter**. Das ist wie ein Regler, an dem wir drehen können, um das Verhalten des Modells anzupassen.

-   **Kleiner `C`-Wert (z.B. 0.01):** Das Modell ist "strenger" und hält sich sehr stark an die allgemeinen Muster. Es ignoriert einzelne Datenpunkte, die aus der Reihe tanzen. Man nennt das **starke Regularisierung**. Das hilft, zu verhindern, dass das Modell die Trainingsdaten "auswendig lernt".
-   **Großer `C`-Wert (z.B. 100):** Das Modell versucht, jeden einzelnen Datenpunkt so gut wie möglich richtig zu klassifizieren, auch die Ausreißer. Das kann dazu führen, dass es die Trainingsdaten zu gut lernt und bei neuen Daten schlechter wird.

Wir haben verschiedene `C`-Werte ausprobiert (`[0.01, 0.1, 1.0, 10, 100]`), um zu sehen, welche Einstellung am besten funktioniert.

### b) Entscheidungsbaum (Decision Tree)

Ein Entscheidungsbaum trifft Entscheidungen, indem er eine Reihe von "Ja/Nein"-Fragen stellt.

**Analogie:** Stell dir einen Arzt vor, der eine Diagnose stellt:
- "Ist der Patient über 50?" -> Ja
- "Hat der Patient Schmerzen in der Brust?" -> Ja
- "Ist der Cholesterinwert über 200?" -> Nein
- ... und so weiter, bis eine Entscheidung getroffen wird.

Jede Frage ist ein "Knoten" im Baum, und die Antworten sind die "Äste".

#### Was bedeutet `max_depth` (maximale Tiefe)?

`max_depth` ist auch ein Hyperparameter. Er kontrolliert, wie viele Fragen der Baum stellen darf, bevor er eine Entscheidung treffen muss.

-   **Kleine `max_depth` (z.B. 2):** Der Baum ist sehr einfach und stellt nur wenige Fragen. Er ist ein "Generalist" und erfasst nur die groben Muster.
-   **Große `max_depth` (z.B. 20) oder `None` (unbegrenzt):** Der Baum wird sehr komplex und stellt viele detaillierte Fragen. Er kann die Trainingsdaten perfekt lernen, aber er wird möglicherweise zu einem "Spezialisten", der bei neuen Daten Fehler macht, weil er zu sehr auf die Details der Trainingsdaten fixiert ist (Overfitting).

Wir haben verschiedene `max_depth`-Werte ausprobiert (`[2, 5, 10, 20, None]`), um die beste Tiefe für unseren Baum zu finden.

## 3. Was ist MLflow?

Stell dir vor, du bist ein Wissenschaftler und machst eine Menge Experimente. Du änderst jedes Mal eine Kleinigkeit und schreibst die Ergebnisse in ein Notizbuch. Genau das ist **MLflow** – ein digitales Notizbuch für unsere Machine-Learning-Experimente.

Ohne MLflow wäre es ein Chaos, den Überblick zu behalten:
- Welcher `C`-Wert hat die besten Ergebnisse geliefert?
- Welcher Baum war besser? Der mit Tiefe 5 oder 10?
- Welches Modell-Gehirn gehört zu welchem Ergebnis?

MLflow hilft uns, all das automatisch zu protokollieren. Wir haben für jeden einzelnen Trainingslauf (also für jeden `C`-Wert und jede `max_depth`) Folgendes protokolliert:

-   **Parameter:** Die Einstellungen des Experiments (z.B. `model_type = LogisticRegression`, `C = 0.1`).
-   **Metriken:** Die Messergebnisse, die uns sagen, wie gut das Modell war (z.B. `test_accuracy = 0.883`). Die "Test Accuracy" ist besonders wichtig, da sie uns sagt, wie gut das Modell mit neuen, unbekannten Daten umgeht.
-   **Artefakte:** Das sind die "Produkte" des Experiments. In unserem Fall haben wir das trainierte Modell selbst als Artefakt gespeichert. So können wir später genau das Modell wieder laden, das die besten Ergebnisse erzielt hat.

Die **MLflow UI** (die Weboberfläche unter `http://localhost:5000`) ist das visuelle Interface für unser digitales Notizbuch. Dort kannst du all deine Experimente vergleichen, sortieren und die Ergebnisse analysieren.

## 4. Zusammenfassung: Was hat das Skript gemacht?

1.  **Daten geladen:** Es hat den Herz-Kreislauf-Datensatz geladen.
2.  **Experimente durchgeführt:**
    - Es hat 5x das "Logistische Regression"-Gehirn trainiert, jedes Mal mit einem anderen `C`-Wert.
    - Es hat 5x das "Entscheidungsbaum"-Gehirn trainiert, jedes Mal mit einer anderen `max_depth`.
3.  **Alles protokolliert:** Für jedes dieser 10 Experimente hat es die Einstellungen (Parameter), die Ergebnisse (Metriken) und das trainierte Gehirn (Artefakt) in MLflow gespeichert.

Jetzt kannst du zur MLflow UI gehen und wie ein Detektiv herausfinden, welches Modell mit welchen Einstellungen der Champion für die Vorhersage von Herzkrankheiten ist!
