set terminal svg size 2000,1000
#set term dumb
set output 'histogram0,0001(2)Multi.svg'
set title "Na = 20000 Nv = 5000"
set grid
set style data linespoints
plot 'train.d',\
    'val.d'
