
Enhancements to Linaer Regression and KNN

* **Kernel methods** use weights that decrease smoothly to zero with distance
  from the arget point, rather than the effective 0,1 weights used in KNN.

* In high-dimensional spaces the distance kernels are modified to emphasize
  some variable more than others.

* Local regression fits liear models by locally weighted least squares, rather
  than fittin constants locally.

* Linear models fit to a **basis expansion** of the original inputs allow
  arbitrarily complex models.

* Projection pursuit and neural network models consist of sums of non-linaerly
  transformed linear models.
