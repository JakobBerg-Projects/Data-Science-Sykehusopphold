# Data Science-prosjekt

Denne oppgaven presenterer en maskinlæringsmodell utviklet for å predikere lengden på sykehusopphold for individuelle pasienter. Modellen benytter pasientinformasjon, inkludert fysiologiske data, demografiske data og data om sykdomsalvorlighet på tvers av ni ulike sykdomskategorier. Målet er å gi nøyaktige anslag på oppholdslengden for nye pasienter basert på disse variablene. I tillegg inkluderer prosjektet en klassifikasjonsmodell som predikerer risikoen for sykehusdød.

## Innhold

- **Modeller og filer**
  - `avg_length_by_category_dict.pkl`: Gjennomsnittlig oppholdslengde per sykdomsunderkategori basert på treningsdata.
  - `model.pkl`: Lagret regresjonsmodell for prediksjon av sykehusoppholdslengde.
  - `sykehusdod_model.pkl`: Klassifikasjonsmodell for å predikere risiko for sykehusdød.
  - `predictions.csv`: Sampledata med prediksjoner fra modellene.
  
- **Applikasjon**
  - Ved å kjøre `app.py` startes en lokal nettside på port 8080. Her kan brukere teste modellene ved å legge inn egne verdier for variablene. Nettsiden gir estimater for oppholdslengde og klassifiserer risikoen for sykehusdød.

## Data

Dette datasettet omfatter 8261 individuelt kritisk syke pasienter fra 5 medisinske sentre i USA, registrert i periodene 1989-1991 og 1992-1994.

- **Datasett:**
  - **Sykehusdata:** 7740 pasienter, 3 variabler.
  - **Sykdomsalvorlighet:** 4 sykdomskategorier, 20 variabler.
  - **Fysiologiske data:** 7740 pasienter, 15 variabler.
  - **Demografiske data:** 7742 pasienter, 6 variabler.

Hver rad representerer journaldata for innlagte pasienter som oppfylte inklusjons- og eksklusjonskriterier for ni sykdomsunderkategorier.

## Filstruktur

- **`datatilberedning.ipynb`**  
  Forbereder data for analyse og modellering. Oppretter trenings-, validerings- og testdata.

- **`visualisering.ipynb`**  
  Visualiserer treningsdata for å identifisere mønstre og sammenhenger.

- **`imputering_og_modellering.ipynb`**  
  Tester ulike imputeringsteknikker og modeller for å estimere sykehusoppholdslengde. Kjører også modelltrening for:
  - Regresjonsmodell (`model.pkl`)
  - Klassifikasjonsmodell for sykehusdød (`sykehusdod_model.pkl`)
  - Gjennomsnittlig oppholdslengde per sykdomsunderkategori (`avg_length_by_category_dict.pkl`).

- **`preprocessing.py`**  
  Klargjør datasett for modellene ved å opprette nye variabler og bruke en valgt imputeringsstrategi.

- **`app.py`**  
  Starter en lokal nettside der brukere kan teste modellene med egne verdier.  
  - Gir et estimat på oppholdslengden.  
  - Klassifiserer risiko for sykehusdød.

- **`index.html`**  
  HTML-fil som støtter nettsiden.

## Eksempel på variabler

### Sykehusdata

| Variabelnavn   | Rolle               | Type         | Beskrivelse                                             |
|----------------|---------------------|--------------|---------------------------------------------------------|
| pasient_id     | ID                  | Heltall      | Unik ID for pasienter                                  |
| sykehusdød     | Funksjonsvariabel   | Binær        | Indikerer død på sykehuset (1 = Ja, 0 = Nei)           |
| oppholdslengde | Responsvariabel     | Kontinuerlig | Antall dager fra studieinngang til utskrivelse          |

### Fysiologiske data

| Variabelnavn       | Rolle            | Type         | Beskrivelse                                             |
|--------------------|------------------|--------------|---------------------------------------------------------|
| blodtrykk          | Funksjonsvariabel| Kontinuerlig | Gjennomsnittlig arterielt blodtrykk målt på dag 1       |
| hjertefrekvens     | Funksjonsvariabel| Kontinuerlig | Pasientens hjertefrekvens målt på dag 1                |
| kroppstemperatur   | Funksjonsvariabel| Kontinuerlig | Kroppstemperatur i Celsius-grader                      |

Flere detaljer om datasettene og variablene finnes i prosjektfilene.

---

Med denne strukturen og de inkluderte filene kan prosjektet enkelt testes og videreutvikles.
