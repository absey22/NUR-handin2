#!/bin/bash


echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "Get data files ..."

wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
wget strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
wget strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5


echo "Run the first exercise scripts ..."
echo "--1(a)--"
python3 ex1a.py > "ex1aoutput.txt"
echo "--1(e)--"
python3 ex1e.py &   # runs in background
echo "--1(b)--"
python3 ex1b.py
echo "--1(c)--"
python3 ex1c.py
echo "--1(d)--"
python3 ex1d.py


echo "Run the third  exercise script ..."
python3 ex3.py

echo "Run the sixth  exercise script ..."
python3 ex6.py > "ex6output.txt"

echo "Generating the pdf ..."

pdflatex template.tex

echo " "
echo "==> SEE: template.pdf for solutions!!"




