# Report

This directory contains the source code for the project reports in LaTeX format.

## Structure

The reports are organized into two main templates:

- **LNCS** (Lecture Notes in Computer Science Style):
  - `docs_en`: English version.
  - `docs_vi`: Vietnamese version.

- **UIT** (University of Information Technology Style):
  > **Recommended**: This template is fully featured and consistent with the format used by the rest of the class.
  - `docs_en`: English version.
  - `docs_vi`: Vietnamese version.

## Usage

To compile the reports, navigate to the specific directory (e.g., `UIT/docs_en`) and compile the `main.tex` file using your preferred LaTeX editor or compiler (e.g., `pdflatex`, `latexmk`).

Example:

```bash
cd UIT/docs_en
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```
