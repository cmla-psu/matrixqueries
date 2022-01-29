# Fitness-For-Use 

Sourse code for the paper Optimizing Fitness-For-Use of Differentially Private Linear Queries.

## Description

This is a convex optimization problem with inequality constraints. We use the barrier method to solve this problem. For the newton's step, conjugate gradient method is used to avoid calculation of inverse matrix. Then we use Armijo's rule to determine the step size. 

The algorithms are implemented in the following files. 

-  **softmax.py**:  Implementation of the algorithms SM-II, IP, HM

- **convexdp.py**: Implementation of algorithms CA, wCA-I, wCA-II



## Usage

The following code solves the Fitness-for-use problem using SM-II algorithm, you need to specify `basis` as the basis matrix, `index` as the index matrix and `bound` as the accuracy constraints.

```python
from softmax import configuration, matrix_query
args = configuration()
# basis matrix is B
# index matrix is L
# bound vector is c
mat_opt = matrix_query(args, basis, index, bound)
mat_opt.optimize()
```

The following files have the experiments in the paper,

- **1_range.py:** Experiment for range queries (section 7.4)
- **2_discrete.py**: Experiment for random queries (section 7.5)
- **3_pl94.py:** Experiment for PL-94 (section 7.3)
- **4_age.py:** Experiment for Age Pyramids (section 7.6)
- **5_marginal.py:** Experiment for marginals (section 7.7)

Simply run the main function, change the parameter n for different settings. You can also change the hyperparameters in the code to achieve more accurate result.



## Customizing the detection



## Citing this work

You are encouraged to cite the following [paper](https://arxiv.org/abs/2012.00135) if you use this tool for academic research:

```bibtex
@article{xiao2020optimizing,
  title={Optimizing Fitness-For-Use of Differentially Private Linear Queries},
  author={Xiao, Yingtai and Ding, Zeyu and Wang, Yuxin and Zhang, Danfeng and Kifer, Daniel},
  journal={arXiv preprint arXiv:2012.00135},
  year={2020}
}
```



## License

[MIT](https://github.com/cmla-psu/matrixqueries/blob/master/LICENSE).
