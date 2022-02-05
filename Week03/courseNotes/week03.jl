using LinearAlgebra
using Distributions

# Cholesky that assumes PD matrix
function chol_pd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        sum = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            sum =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        root[j,j] =  sqrt(a[j,j] .- sum);

        ir = 1.0/root[j,j]
        #update off diagonal rows of the column
        for i in (j+1):n
            s = root[i,1:(j-1)]' * root[j,1:(j-1)]
            root[i,j] = (a[i,j] - s) * ir 
        end
    end
end

sigma = fill(0.9,(5,5))
for i in 1:5
    sigma[i,i]=1.0
end

root = Array{Float64,2}(undef,(5,5))

chol_pd!(root,sigma)

root*root' == sigma

root2 = cholesky(sigma).L
root == root2

#Cholesky that assumes PSD
function chol_psd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        sum = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            sum =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        root[j,j] =  sqrt(a[j,j] .- sum);

        #Check for the 0 eigan value.  Just set the column to 0 if we have one
        if 0.0 == root[j,j]
            root[j,(j+1):n] .= 0.0
        else
            #update off diagonal rows of the column
            ir = 1.0/root[j,j]
            for i in (j+1):n
                s = root[i,1:(j-1)]' * root[j,1:(j-1)]
                root[i,j] = (a[i,j] - s) * ir 
            end
        end
    end
end



sigma[1,2] = 1.0
sigma[2,1] = 1.0

eigvals(sigma)

chol_psd!(root,sigma)

root*root' == sigma

root2 = cholesky(sigma).L

sigma[1,2] = 0.725
sigma[2,1] = 0.725
eigvals(sigma)

chol_psd!(root,sigma)