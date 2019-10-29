set terminal png size 1920,1080 enhanced font 'Verdana,10'

#set output "curve.png"
set term dumb
# Line width of the axes
set border linewidth 1.5
# Line styles
set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2
set style line 2 linecolor rgb '#dd181f' linetype 1 linewidth 2
f(x) = -0.64166665 * x + (1.1083333)

plot f(x) title 'f[tfinal](x)' with lines linestyle 1,