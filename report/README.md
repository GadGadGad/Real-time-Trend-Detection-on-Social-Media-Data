# Report

This directory contains the source code for the project reports in LaTeX format.

## Structure

The reports are organized into two main templates:

- **LNCS** (Lecture Notes in Computer Science Style):
  - `docs_en`: English version (Standard Release).
  - `docs_vi`: Vietnamese version (Translated Version).

- **UIT** (University of Information Technology Style):
  > **Recommended**: This template is fully featured and consistent with the format used by the rest of the class.
  - `docs_en`: English version (Standard Release).
  - `docs_vi`: Vietnamese version (Translated Version).

## Usage

To compile the reports, navigate to the specific directory and compile the `main.tex` file using your preferred LaTeX editor or compiler (e.g., `pdflatex`, `latexmk`).

**Note on Bibliography:**

- **UIT Style**: Uses `biber` for bibliography management.
- **LNCS Style**: Uses `bibtex` for bibliography management.

Example for **UIT Style**:

```bash
cd UIT/docs_en
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Example for **LNCS Style**:

```bash
cd LNCS/docs_en
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
