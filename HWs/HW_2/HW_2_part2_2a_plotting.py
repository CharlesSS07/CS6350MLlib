
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results = pd.read_csv('../results/HW_2_part2_2a_errors_to_iterations.csv')

plt.plot(results.iterations, results.training_error/5000)
plt.plot(results.iterations, results.training_error/5000, '.')
plt.xlabel('Iterations')
plt.ylabel('Errors on Training Set')
plt.title('Training Errors vs. Number of Weak Classifiers in AdaBoost')
plt.show()
plt.close()

plt.plot(results.iterations, results.testing_error/5000)
plt.plot(results.iterations, results.testing_error/5000, '.')
plt.xlabel('Iterations')
plt.ylabel('Errors on Testing Set')
plt.title('Testing Errors vs. Number of Weak Classifiers in AdaBoost')
plt.show()
plt.close()

classifier_errors = pd.read_csv('../results/HW_2_part2_2a_classifier_errors.csv')

plt.plot(classifier_errors.classifier, classifier_errors.training_error/5000)
# plt.plot(classifier_errors.classifier, classifier_errors.training_error/5000, '.')
plt.xlabel('i')
plt.ylabel('Errors on Training Set')
plt.title('Training Error of Classifier i')
plt.show()
plt.close()

plt.plot(classifier_errors.classifier, classifier_errors.testing_error/5000)
# plt.plot(classifier_errors.classifier, classifier_errors.testing_error/5000, '.')
plt.xlabel('i')
plt.ylabel('Errors on Testing Set')
plt.title('Testing Error of Classifier i')
plt.show()
plt.close()