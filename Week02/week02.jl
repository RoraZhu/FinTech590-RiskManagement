using Distributions
using StatsBase
using DataFrames
using Gadfly
using Cairo
using Fontconfig
using CSV
using Plots
using Printf
using JuMP
using Ipopt
using StateSpaceModels
using LinearAlgebra


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
n = 50
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



#Example R^2 inflation
prob1 = CSV.read("Project/problem1.csv",DataFrame)
n = size(prob1,1)
X = [ones(n) prob1.x]
Y = prob1.y

function calc_r2(Y,X)
    n = size(Y,1)
    p = size(X,2)

    B = inv(X'*X)*X'*Y
    e = Y - X*B

    sse = e'*e
    Y_n = Y .- mean(Y)
    ssy = Y_n' * Y_n

    R2 = 1.0 - sse/ssy
    Adj_R2 = 1.0 - (sse/ssy)*(n-1)/(n-p-1)
    return R2, Adj_R2
end

p = [i for i in 2.0:100.0]
R2 = Vector{Float64}(undef,size(p,1))
aR2 = Vector{Float64}(undef,size(p,1))

R2[1], aR2[1] = calc_r2(Y,X)
for i in 2:size(p,1)
    X = [X randn(n)]
    R2[i], aR2[i] = calc_r2(Y,X)
end

Plots.plot(p,[R2 aR2], label=["R^2" "Adjusted R^2"],legend=:topleft)

#ACF and PACF

function plot_ts(y;imgName="series", length=10)
    n = size(y,1)
    l = [i for i in 1:length]
    acf = autocor(y,l)
    p_acf = pacf(y, l)

    df = DataFrame(:t=>l, :acf=>acf, :pacf=>p_acf)

    bg = Theme(
        panel_fill=colorant"grey"
    )

    Gadfly.with_theme(:dark) do
        p0 = plot(x=[i for i in 1:n], y=y,Geom.line)

        p1 = plot(df, layer(x=:t, y=:acf, Geom.line), layer(x=:t, y=:acf, Geom.point), Scale.x_discrete)
        push!(p1,Guide.title("AutoCorrelation"))
        p2 = plot(df, layer(x=:t, y=:pacf, Geom.line),layer(x=:t, y=:pacf, Geom.point), Scale.x_discrete)
        push!(p2,Guide.title("Partial AutoCorrelation"))

        p = vstack(p0,hstack(p1,p2))

        img = PNG(imgName,8inch, 8inch)
        draw(img,p)
    end

    nothing
end

#AR1
#y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.1)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last = 1.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 1:(n+burn_in)
    global yt_last
    y_t = 1.0 + 0.5*yt_last + e[i]
    yt_last = y_t
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))
println(@sprintf("Expected values Y: %.2f, %.4f",2.0,.01/(1-.5^2)))


plot_ts(y,imgName="ar1_acf_pacf.png")

ar1 = SARIMA(y,order=(1,0,0),include_mean=true)

StateSpaceModels.fit!(ar1)
results(ar1)


#MA1
#y_t = 1.0 + .05*e_t-1 + e, e ~ N(0,.01)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last = 1.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 2:(n+burn_in)
    global yt_last
    y_t = 1.0 + 0.5*e[i-1] + e[i]
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))
println(@sprintf("Expected values Y: %.2f, %.4f",1.0,(1+.5^2)*.01))

plot_ts(y,imgName="ma1_acf_pacf.png")

ma1 = SARIMA(y,order=(0,0,1),include_mean=true)

StateSpaceModels.fit!(ma1)
results(ma1)
