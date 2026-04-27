# T&C Scanner – AI‑Powered Legal Risk Analyzer

A Streamlit web application that extracts text from PDFs or pasted Terms & Conditions, retrieves the most relevant sections using RAG (dense embeddings + cosine similarity), and provides a **fairness score**, **risk level**, and **precision/recall metrics** based on heuristic risk keywords with negation detection.

## What it Does
- Upload a PDF or paste T&C text.
- The system chunks the text, computes embeddings (`all-MiniLM-L6-v2`), and retrieves the most relevant chunk for a risk analysis query.
- A lightweight language model (`distilgpt2`) attempts to summarise risks, while a rule‑based engine computes:
  - **Fairness Score** (0–100, higher is fairer)
  - **Risk Level** (Low / Moderate / High)
  - **Precision** (risk coverage) and **Recall** (risk detection)
- Results are displayed instantly, along with the retrieved context used for scoring.

## Quick Start
Here’s how to set up / recreate the tc_scanner environment...
(rest of your instructions)

Here’s how to **set up / recreate** the tc_scanner environment.
## Remove the old environment (if it exists)

Open a terminal. Then run:

```bash
conda deactivate
conda env remove -n tc_scanner
```
This deletes the old environment completely.

## Step 1: Create the new environment

In the terminal, from your project root, run:
```bash
conda env create -f environment.yml
```
This will download and install all dependencies. It may take a few minutes.
After it finishes, activate the environment:
```bash
conda activate tc_scanner
```
## Step 2: Run the app again
```bash
streamlit run src/app.py
```
3. Upload a PDF or paste text, then click Analyze Document.


## Video Links
- [Demo video]
[![Watch the video](https://youtu.be/aV61LOzrsWk)](https://youtu.be/aV61LOzrsWk)
Click the link above or paste this link into your browser: https://youtu.be/aV61LOzrsWk
Slides for Demo:
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vT3tPC_3_GjB-soAE_reOxYR4DDmSJTaIwnWSYnAyDnRH-Z6b63ku7FfCYmcqrfwMN4h5-ZjDerM8bR/pubembed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

- [Technical walkthrough]
[![Watch the video](https://youtu.be/oEzHN3xfsnQ)](https://youtu.be/oEzHN3xfsnQ)
Click the link above or paste this link into your browser: https://youtu.be/oEzHN3xfsnQ

## Evaluation

We evaluated the system on a gold‑standard test set of 10 hand‑labeled clauses (5 risky, 5 safe). The model (or fallback heuristic) produced the following metrics:
===== Evaluation Results =====
Accuracy:  70.00%
Precision: 1.00
Recall:    0.40
Confusion matrix:*(Rows: true labels, Columns: predicted labels; 0 = safe, 1 = risky)*
[[5 0]
 [3 2]]

Individual Contributions

Name: Yabesra Ewnetu
Worked alone. All RAG, inference, UI, and evaluation logic was implemented from course assignments (LLM, RAG, Evaluation) with AI assistance only for the Streamlit boilerplate and PDF extraction.
