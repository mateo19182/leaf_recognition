using ScikitLearn

@sk_import svm: SVC 

model = SVC(kernel="rbf", degree=3, gamma=2, C=1); 

