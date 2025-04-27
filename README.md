## Approaches to Text Summarization

### 1. **Abstractive Summarization**
Abstractive summarization generează propoziții noi care pastreaza sensul textului original. Utilizează modele NLP avansate, cum ar fi transformers, pentru a înțelege contextul și a crea rezumate similare cu cele realizate de oameni.

#### Implementare
- **Model**: Folosim modelul `T5-base` din biblioteca Hugging Face.
- **Pipeline**: Pipeline-ul de sumarizare este inițializat cu biblioteca `transformers`.
- **Proces**:
  1. Textul de intrare este denumit `summarize`.
  2. Modelul tokenizează textul de intrare și generează un rezumat folosind beam search.
  3. Rezultatul este parsat și afișat.

#### Exemplu
In `abstractive-transformers.ipynb`:
```python
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'].capitalize())
```
**Output**:
> "Apple aims to make most of its iPhones sold in the United States at factories in India by 2026. The tech giant is holding urgent talks with contract manufacturers Foxconn and Tata."

#### Avantaje
- Generează rezumate care sunt mai naturale și similare cu cele realizate de oameni.
- Poate parafraza și restructura propozițiile pentru o mai bună lizibilitate.

#### Provocări
- Necesită resurse computaționale semnificative.
- Poate introduce inexactități dacă modelul nu înțelege corect contextul.

---

### 2. **Extractive Summarization**
Extractive summarization selectează cele mai importante propoziții sau fraze direct din textul original. Se bazează pe caracteristici statistice și lingvistice pentru a clasifica și extrage propozițiile cheie.

#### Implementare
- **Biblioteci**: Folosim `spacy` pentru tokenizare și `collections.Counter` pentru analiza frecvenței cuvintelor.
- **Proces**:
  1. Tokenizarea textului în cuvinte și filtrarea stopword-urilor și a semnelor de punctuație.
  2. Calcularea frecvenței cuvintelor și normalizarea acestora.
  3. Calcularea scorurilor propozițiilor pe baza sumei frecvențelor cuvintelor.
  4. Selectarea celor mai bune `n` propoziții cu cele mai mari scoruri.

#### Exemplu
In `extractive-tokenization.ipynb`:
```python
from heapq import nlargest
num_sentences = 3
n = nlargest(num_sentences, sent_score, key=sent_score.get)
summary = " ".join(n)
print(summary)
```
**Output**:
> "BENGALURU, April 25 (Reuters) - Apple aims to make most of its iPhones sold in the United States at factories in India by the end of 2026. Prime Minister Narendra Modi has in recent years promoted India as a smartphone manufacturing hub. The U.S. tech giant is holding urgent talks with contract manufacturers Foxconn and Tata."

#### Avantaje
- Simplu și eficient din punct de vedere computațional.
- Păstrează formularea originală, asigurând acuratețea informației.

#### Provocări
- Poate produce rezumate mai puțin coerente sau redundante.
- Limitat la extragerea propozițiilor existente fără parafrazare.
- Poate omite context important.