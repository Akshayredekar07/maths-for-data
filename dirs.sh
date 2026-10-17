# #!/bin/bash

# # Create math folders
# mkdir -p math01_scalars_and_basics
# mkdir -p math02_vectors
# mkdir -p math03_matrices
# mkdir -p math04_linear_algebra
# mkdir -p math05_linear_transformations
# mkdir -p math06_eigenvalues_eigenvectors
# mkdir -p math07_matrix_decomposition
# mkdir -p math08_limits_continuity
# mkdir -p math09_derivatives
# mkdir -p math10_partial_derivatives
# mkdir -p math11_gradients_jacobians
# mkdir -p math12_optimization
# mkdir -p math13_probability_statistics
# mkdir -p math14_information_theory
# mkdir -p math15_numerical_methods
# mkdir -p math16_loss_functions
# mkdir -p math17_applications_ml

# echo "All folders created!"


#!/bin/bash

# Rename all folders to add double underscore after mathXX_
# for folder in math*; do
#     if [ -d "$folder" ]; then
#         # Extract number and name parts
#         new_name=$(echo "$folder" | sed 's/\(math[0-9]\+\)_/\1__/')
        
#         # Rename if different
#         if [ "$folder" != "$new_name" ]; then
#             mv "$folder" "$new_name"
#             echo "Renamed: $folder → $new_name"
#         fi
#     fi
# done

# echo "✅ All folders renamed successfully!"


#!/bin/bash

# Create empty README.md in each math folder
# for folder in math*; do
#     if [ -d "$folder" ]; then
#         touch "$folder/README.md"
#         echo "✅ Created README.md in $folder"
#     fi
# done

# echo "✅ All README.md files created successfully!"


#!/bin/bash

mkdir -p mathematics-for-ml/{\
01_linear_algebra/{\
01_scalars_and_basics,\
02_vectors,\
03_matrices,\
04_matrix_operations,\
05_determinants_and_inverses,\
06_vector_spaces,\
07_linear_transformations,\
08_eigenvalues_eigenvectors,\
09_matrix_decomposition,\
10_tensors},\
02_calculus/{\
01_limits_and_continuity,\
02_derivatives,\
03_differentiation_rules,\
04_partial_derivatives,\
05_gradients_and_jacobians,\
06_chain_rule_multivariable,\
07_optimization_theory,\
08_convexity,\
09_gradient_descent,\
10_numerical_methods},\
03_probability/{\
01_probability_basics,\
02_random_variables,\
03_probability_distributions,\
04_discrete_distributions,\
05_continuous_distributions,\
06_joint_probability,\
07_conditional_probability,\
08_bayes_theorem,\
09_expectation_variance,\
10_information_theory},\
04_statistics/{\
01_descriptive_statistics,\
02_measures_central_tendency,\
03_measures_dispersion,\
04_correlation_covariance,\
05_sampling_distributions,\
06_central_limit_theorem,\
07_confidence_intervals,\
08_hypothesis_testing,\
09_statistical_tests,\
10_bayesian_statistics}\
}
