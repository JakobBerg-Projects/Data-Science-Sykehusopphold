# Estimering av sykehusopphold Data Science

Prosjektet er delt inn i 3 deler.

# Datatilbredning.ipynb
Dennne filen kjøres for å forberede data til analyse, og modellering. Laster opp trenings-, validerings-, og testadata.

# Visualisering.ipynb
Laser inn treningsdataen fra datatilbredning. Visualiser så på treningsdata.

# Imputering og modellering.ipynb
Laster inn trenings-, validerings-, og testdata. Tester forskjellige metoder for imputering av manglende verdier, og ulike modeller for estimering av sykehusopphold.

# Preprocessing.py
Denne python filen klargjør et datasett til både klassifikasjonsmodellen og regresjonsmodellen. Dette gjøres både ved å opprette nye variabler med utgangspunkt i eksisterende variabler. I tillegg inneholder filen imputeringsstrategien som er valgt.

# App.py
Kjøres for å få en lokal nettside på port 8080. Her vil en kunne fylle ut verdier for å få oppgitt et estimat på oppholdslengden. I tillegg vil nettsiden klassefisere om det er høy risiko for dødsfall for pasienten. 

# Index.html
HTML tilhørende nettsiden for predikasjonen.


