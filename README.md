# matrixqueries

### Description

This is a convex optimization problem with inequality constraints. We use the barrier method to solve this problem. For the newton's step, conjugate gradient method is used to avoid calculation of inverse matrix. Then we use Armijo's rule to determine the step size. 

We compare our method with HB method. Under the same privacy cost, HB method fails to satisify the variance bound constraints.

The code consists of the following parts. 

- Initialization
- Derivatives Caculation
- Hessian Caculation
  - Avoid Kronecker Product
  - Vectorization 
- Conjugate Gradient Method
- Stopping Criteria
- Armijo's Rule for Step Size
- Compare with HB Method

### Usage

Simply run the main function, change the parameter $n$ for different settings. You can also change the hyperparameters in the code to achieve more accurate result.