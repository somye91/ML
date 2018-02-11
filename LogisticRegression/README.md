This example works with the Iris dataset. We will classify only the first two flowers that are present in firsto 100 rows in the dataset.

The input vector is of dimension X = M x F
F=number o features
M= number of samples

Output vector y = M x 1

We initiaize the weights θ (theta) to 0 initially as F X 1

The weighted sum is Z = X dot θ

Sigmoid function S(z) = 1 / (1+e^-z)

Output = H(x) = S(Z)

<b>Cost function</b> <br>
Due to the sigmoid function if we choose the cost function similar to liear regression, i.e. (y-(mx+b))^2 / N , then this would not be convex (bowl shaped) and would have multiple local minimas. 
Instead we want the function to be convex and have 1 global minima (ex. parabolic function )
<br><br>
<b>Hence the cost function is:</b>  <br>

![alt text](https://github.com/somye91/ML/blob/master/LogisticRegression/Img/cf.png)
![alt text](https://github.com/somye91/ML/blob/master/LogisticRegression/Img/y1andy2_logistic_function.png)

Combined cost function is:  <br><br>
![alt text](https://github.com/somye91/ML/blob/master/LogisticRegression/Img/logistic_cost_function_joined.png)

<br><br>

<b> Vectorized form </b>
<br>
![alt text](https://github.com/somye91/ML/blob/master/LogisticRegression/Img/logistic_cost_function_vectorized.png)

We have to adjust the weights as: <br>
Δθ = X * (error)  #where error = H(x) - y<br>
In vector form Δθ= X.T (dot) error  <br>
And now adjust the weights θ = θ - 1/M *lr Δθ  <br>
