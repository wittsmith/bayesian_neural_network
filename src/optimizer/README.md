Adam Optimizer Fundamentals
The Adam optimizer is widely used in deep learning due to its adaptive learning rate and efficient handling of sparse gradients. Its key elements include:

First and Second Moments:

m (First Moment): An exponentially decaying average of past gradients.

v (Second Moment): An exponentially decaying average of the squared gradients.

Bias Correction: Because the moment estimates are initialized at zero, bias-corrected estimates (m̂ and v̂) are computed to avoid initial underestimation.

Update Rule: For a parameter 
𝜃
θ at time step 
𝑡
t, the update is computed as follows:

𝑚
𝑡
=
𝛽
1
⋅
𝑚
𝑡
−
1
+
(
1
−
𝛽
1
)
⋅
𝑔
𝑡
,
𝑣
𝑡
=
𝛽
2
⋅
𝑣
𝑡
−
1
+
(
1
−
𝛽
2
)
⋅
𝑔
𝑡
2
,
𝑚
^
𝑡
=
𝑚
𝑡
1
−
𝛽
1
𝑡
,
𝑣
^
𝑡
=
𝑣
𝑡
1
−
𝛽
2
𝑡
,
𝜃
𝑡
+
1
=
𝜃
𝑡
−
𝛼
⋅
𝑚
^
𝑡
𝑣
^
𝑡
+
𝜖
,
m 
t
​
 
v 
t
​
 
m
^
  
t
​
 
θ 
t+1
​
 
​
  
=β 
1
​
 ⋅m 
t−1
​
 +(1−β 
1
​
 )⋅g 
t
​
 ,
=β 
2
​
 ⋅v 
t−1
​
 +(1−β 
2
​
 )⋅g 
t
2
​
 ,
= 
1−β 
1
t
​
 
m 
t
​
 
​
 , 
v
^
  
t
​
 = 
1−β 
2
t
​
 
v 
t
​
 
​
 ,
=θ 
t
​
 −α⋅ 
v
^
  
t
​
 
​
 +ϵ
m
^
  
t
​
 
​
 ,
​
 
where:

𝛼
α is the learning rate.

𝛽
1
β 
1
​
  and 
𝛽
2
β 
2
​
  are decay rates for the moments.

𝜖
ϵ is a small constant to prevent division by zero.

4. Integrating Adam with a Bayesian Neural Network
When applying Adam within the framework of a Bayesian neural network, consider the following aspects:

Parameterization of the Posterior:
The parameters being optimized are those of your approximate posterior distribution (e.g., the means and log-variances of the weight distributions). Adam will update these variational parameters.

Gradient Computation on the ELBO:
Instead of a standard loss function (like cross-entropy), you’re computing gradients of the ELBO with respect to the variational parameters. This includes:

Gradients from the data likelihood component.

Gradients from the KL divergence term.

Stochastic Gradient Estimates:
Due to the use of Monte Carlo sampling (enabled by the reparameterization trick), the gradient estimates will be stochastic. Adam’s adaptive nature helps in smoothing these noisy updates.

Hyperparameter Tuning:
Typical Adam settings (e.g., 
𝛼
=
0.001
α=0.001, 
𝛽
1
=
0.9
β 
1
​
 =0.9, 
𝛽
2
=
0.999
β 
2
​
 =0.999, 
𝜖
=
10
−
8
ϵ=10 
−8
 ) often serve as a good starting point, but you might need to adjust them considering the added complexity of the Bayesian formulation.

