# matrixqueries

### Description

This is a convex optimization problem with inequality constraints. We use the barrier method to solve this problem. For the newton's step, conjugate gradient method is used to avoid calculation of inverse matrix. Then we use Armijo's rule to determine the step size. 

We compare our method with HB method. Under the same privacy cost, HB method fails to satisify the variance bound constraints.

The code consists of the following parts. 

- **matrix_query.py** : Implementation of the class matrix_query
  - Use conjugate gradient method to find newton's step
  - Use armijo's rule to find step size
- **common.py** : Commonly used functions
  - Inequality constraint functions for variance and privacy cost
  - Objective function
  - Derivatives calculation
  - Hessian calculation : avoid using kronecker product
- **matrixop.py** : matrix operations
  - is_pos_def : determine whether a matrix is postive definite or not
  - matrix_3d_broadcasting : help function for gradient calculation
- **privacy.py** : privacy analysis
  - Calculate privacy cost and privacy cost vector
- **workload.py** : generate matrix
  - Workload matrix W
  - Basis matrix B
  - Index matrix V
  - Variance bound c
- **config.py** : configuration parameters

### Usage

Simply run the main function in matrix_query.py, change the parameter n​ for different settings. You can also change the hyperparameters in the code to achieve more accurate result.