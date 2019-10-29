set terminal png size 1080,1080 enhanced font 'Verdana,10'

set yrange [-2:2] 
set output "curve.png"
set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2

set obj 1 circle at 0.0,0.0 fc rgb "blue" size 0.1
set obj 2 circle at 0.0,1.0 fc rgb "blue" size 0.1
set obj 3 circle at 1.0,0.0 fc rgb "blue" size 0.1
set obj 4 circle at 1.0,1.0 fc rgb "red" size 0.1
plot [-2:2] 0.25358087 * x + (0.11015527)
