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
echo "--1(a)--"
python3 ex1a.py
echo "--1(b)--"
python3 ex1b.py
echo "--1(c)---"
python3 ex1c.py
echo "--1(d)--"
python3 ex1d.py
echo "--1(e)--"
python3 ex1e.py

# Script that pipes output to a file
echo "Run the second  exercise scripts ..."
python3 ex2.py

echo "Generating the pdf"

pdflatex template.tex
echo "SEE: template.pdf for solutions."




