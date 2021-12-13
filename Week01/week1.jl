using Distributions
using DataFrames
using Gadfly
using Cairo
using Fontconfig
using BenchmarkTools


# PDF Example
d = Normal()

x = [i for i in -5:0.01:5]
df = DataFrame(:x => x)
df[!,:pdf] = pdf.(d,x)

println(first(df,5))

p = plot(df,x=:x, y=:pdf, Geom.line)
img = PNG("pdf.png",6inch, 4inch)
draw(img,p)

# CDF
df[!,:cdf] = cdf.(d,x)

p = plot(df,x=:x, y=:cdf, Geom.line)
img = PNG("cdf.png",6inch, 4inch)
draw(img,p)

# Quick and dirty integration of the PDF
n=501
approxCDF = 0.0
for i in 1:n
    approxCDF += df.pdf[i]*0.01
end

println("CDF actual $(df.cdf[n]) vs calculated $testCDF for F_x($(df.x[n]))")
