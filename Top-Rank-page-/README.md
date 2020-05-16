### Project Overview

 In this project I have focussed on solving top rank page algorithm which was used by Google.  Here, the algorithmis solved using eigen vector decomposition and power method. It is also found out that that power method doesn't always work and also did modified power method.


### Learnings from the project

 After the completion of the project i have a better understanding of python coding and moreover the concepts get clearer.


### Approach taken to solve the problem

 The problem was solved using eigen vector decomposition and power method. In eigen vector decomposition,  the matrix was initialized and then found the eigenvector and eigenvalues of corresponding matrix.  Here we consider eigenvalue =1, corresponding vector is found out and using numpy.where() the index of the top rank page is found out. In power method, the underlying formula is I^(k+1) = adj.mat.(I^k) . Used a for loop for the iteration and found out the toprank page using numpy.where().Also, analysed that power method doesn't always work and have worked on modified power method, where the formula, for G is dominating  and the same method for power method is used.


### Challenges faced

 The challenges that i faced during this project is finding the index using .where(). Also I studied that a slight change in code can have a great impact in the result.


