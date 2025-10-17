# Log


---
## 24/9
### From the project proposal
Main objectives
- Select and justify what kind of kernel method and NN architecture to use and what
valuation methods are appropriate.
  - Test different combinations of models and kernels on smaller dataset (CV) and choose the best for the rest of the project.
  - Train both kernel models and NNs then compare their performance.
- Investigate the differences in regularization, basis expansion and how they handle the
curse of dimensionality.
  - Plot Train and val error against different strength regularization parameters.
  - Different basis expansions are tested in the above stage.
  - Try downscaling the datasets parameters and see how that affects performace. could try naive reduction of parameters by just removing some or reduce the dimensionality ***or*** use PCA to keep the ones with higher variance.
- Attempt to tie the knowledge from the above point to the implementation and investi-
gate the outcomes.
  - This is just connecting theory to empirical resutls

***(Also try a non-kernel implementation since it won't require the nxn kernel matrix)***
Secondary objectives:

- Investigate the amount of compute required to train the various models and how it
scales with dimensions and dataset size.
  - Time the training of the models against different datasizes and plot
  - The memory complexity could be checked with *tracemalloc* wichi apperently is in pythons standard lib.
- Investigate whether we can derive something interesting from the basis expansion formed
by the trained NNs. For instance if we can say anything useful about the structure of
the data by somehow looking at how the NN divides the feature space.
  - I have no idea why I put this here.

### From the proposal feedback
- Tie empirical results to theory: Regularizaton effects, Bias vs variacnced, generalization
  - Most is already covered above but we should include something about generalization. Perhaps be should keep informaion about the generalizatio gap where pertinent and compare NN to kernel method.
- What does the theory tell us to use kernel vs NN

Perhaps secondary: compare the models both in a fair situation and one where they're optimized.
