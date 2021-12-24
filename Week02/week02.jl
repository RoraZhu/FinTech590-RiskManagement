using Distributions
using StatsBase
using DataFrames
using Gadfly
using Cairo
using Fontconfig
using Printf
using JuMP
using Ipopt


#1 Pearson Correlated 
f(x) = 2x 

x = [i for i in -1:.1:1]
y = f.(x)

df = DataFrame(:x => x, :y=>y)

rho = cor(x,y)

p1 = plot(df,x=:x, y=:y, Geom.point)
push!(p1,Guide.title("ρ = $rho"))

#0 Pearson Correlation
f(x) = x^2 

y = f.(x)

df = DataFrame(:x => x, :y=>y)

rho = cor(x,y)

p0 = plot(df,x=:x, y=:y, Geom.point)
push!(p0,Guide.title("ρ = $rho"))

p = hstack(p1,p0)

img = PNG("pearson.png",8inch, 4inch)
draw(img,p)


#Spearman Correlation
f(x) = x^3

x = [randn() for i in -3:.05:3]
y = f.(x)

df = DataFrame(:x => x, :y=>y)

rho = cor(x,y)
spearman = corspearman(x,y)

p0 = plot(df,x=:x, y=:y, Geom.point)
push!(p0,Guide.title(@sprintf("ρ = %.2f -- Spearman = %.2f",rho, spearman)))


img = PNG("spearman.png",6inch, 4inch)
draw(img,p0)

#Example Spearman calculation using Tied Ranks and the Pearson Correlation.
x = [1.2,
0.8,
1.3,
0.8,
0.8,
0.5]

y = randn(6)

println("Spearman $(corspearman(x,y))")

r_x = tiedrank(x)
r_y = tiedrank(y)
println("Calculated Spearman $(cor(r_x,r_y))")

#Regression Example...


#MLE
#sample a random normal N(1.0, 5.0)
samples = 100
d = Normal(1.0,5.0)
x = rand(d,samples)


function myll(m, s)
    n = size(x,1)
    xm = x .- m
    s2 = s*s
    ll = -n/2 * log(s2 * 2 * π) - xm'*xm/(2*s2)
    return ll
end

println("log likelihood N(0,1) = $(myll(0.0,1.0))")
println("log likelihood N(1,5) = $(myll(1.0,5.0))")

#MLE Optimization problem
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, μ, start = 0.0)
    @variable(mle, σ >= 0.0, start = 1.0)

    register(mle,:ll,2,myll;autodiff=true)

    @NLobjective(
        mle,
        Max,
        ll(μ,σ)
    )
##########################

optimize!(mle)

m_hat = value(μ)
s_hat = value(σ)

println("Mean Data vs Optimized $(mean(x)) - $m_hat")
println("Std Data vs Optimized  $(std(x)) - $s_hat")
println("Optimized N($m_hat,$s_hat) = $(myll(m_hat,s_hat))")

xm = x .- m_hat
s2 = xm'*xm / samples
s = sqrt(s2)
println("Biased Std Data vs Optimized $s - $s_hat")

#MLE for Regression
n = 1000
Beta = [i for i in 1:5]
x = hcat(fill(1.0,n),randn(n,4))
y = x*Beta + randn(n)


function myll(s, b...)
    n = size(y,1)
    beta = collect(b)
    xm = y - x*beta
    s2 = s*s
    ll = -n/2 * log(s2 * 2 * π) - xm'*xm/(2*s2)
    return ll
end

#MLE Optimization problem
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, beta[i=1:5],start=0)
    @variable(mle, σ >= 0.0, start = 1.0)

    register(mle,:ll,6,myll;autodiff=true)

    @NLobjective(
        mle,
        Max,
        ll(σ,beta...)
    )
##########################
optimize!(mle)

println("Betas: ", value.(beta))

b_hat = inv(x'*x)*x'*y
println("OLS: ", b_hat)