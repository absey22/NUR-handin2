#!/bin/bash


echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "get data files"

wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt


# Script that returns a plot
echo "Run the first exercise scripts ..."
python3 ex1a.py
python3 ex1b.py
python3 ex1c.py
python3 ex1d.py
python3 ex1e.py

# Script that pipes output to a file
echo "Run the second  exercise scripts ..."
python3 ex2.py

echo "Generating the pdf"

pdflatex template.tex




