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
for folder in math*; do
    if [ -d "$folder" ]; then
        touch "$folder/README.md"
        echo "✅ Created README.md in $folder"
    fi
done

echo "✅ All README.md files created successfully!"