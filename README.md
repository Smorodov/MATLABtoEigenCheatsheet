# MATLAB to Eigen cheat sheet

| MATLAB | Eigen |
|-----------|---------|
| `[C, index] = sort(A);`                   | `Eigen::MatrixXd C = A; Eigen::VectorXi index; C.sort(index);`                                    |
| `[V, D] = eig(A);`                        | `Eigen::EigenSolver<Eigen::MatrixXd> es(A); Eigen::MatrixXd V = es.eigenvectors().real(); Eigen::MatrixXd D = es.eigenvalues().real().asDiagonal();` |
| `A = [1, 2, 3; 4, 5, 6; 7, 8, 9];`       | `Eigen::MatrixXd A(3, 3); A << 1, 2, 3, 4, 5, 6, 7, 8, 9;`                                       |
| `A = [1, 2; 3, 4];`                      | `Eigen::MatrixXd A(2, 2); A << 1, 2, 3, 4;`                                                      |
| `A = B(:, 2:end);`                        | Not directly available, can be implemented using block operations or manual manipulation of columns. |
| `A = B(1:2:end, :);`                      | Not directly available, can be implemented using Eigen::VectorXd for indexing or manual iteration. |
| `A = B(1:3, [1, 3]);`                     | Not directly available, combination of row and column indexing can be implemented using block operations or manual manipulation. |
| `A = eye(3);`                             | `Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);`                                            |
| `A = kron(B, C);`                         | Not directly available, Kronecker product can be implemented using nested loops or custom functions. |
| `A = linspace(1, 10, 10);`                | Not directly available, can be implemented using Eigen::VectorXd and manual filling of values.    |
| `A = ones(3, 3);`                         | `Eigen::MatrixXd A = Eigen::MatrixXd::Ones(3, 3);`                                                |
| `A = rand(3, 3);`                         | `Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);`                                             |
| `A = reshape(B, 2, 4);`                   | Not directly available, reshape operation can be implemented using Eigen::Map or manual manipulation of indices. |
| `A = tril(B);`                            | `Eigen::MatrixXd A = B.triangularView<Eigen::Lower>();`                                          |
| `A = triu(B);`                            | `Eigen::MatrixXd A = B.triangularView<Eigen::Upper>();`                                          |
| `A = zeros(3, 3);`                        | `Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);`                                               |
| `A(:, [1, 3]) = [];`                     | Not directly available, can be implemented using block operations or custom functions.           |
| `A(:, 1:2:end) = [];`                    | Not directly available, can be implemented using block operations or custom functions.           |
| `A(:, 1:2:end) = 0;`                     | Not directly available, can be implemented using custom functions or manual iteration.           |
| `A(:, 1:2:end) = B;`                     | Not directly available, can be implemented using custom functions or manual iteration.           |
| `A(:, 1:3) = 0;`                         | `A.block(0, 0, A.rows(), 3).setZero();`                                                          |
| `A(:, 1:3) = repmat(v, 1, 3);`           | `A.block(0, 0, A.rows(), 3) = v.replicate(1, 3);`                                                |
| `A(:, end:-1:1) = [];`                   | Not directly available, can be implemented using block operations or custom functions.           |
| `A(1:2:end, :) = [];`                    | Not directly available, can be implemented using block operations or custom functions.           |
| `A(1:2:end, :) = 0;`                     | Not directly available, can be implemented using custom functions or manual iteration.           |
| `A(1:2:end, :) = B;`                     | Not directly available, can be implemented using custom functions or manual iteration.           |
| `A(1:2:end, 1:2:end) = [];`              | Not directly available, can be implemented using block operations or custom functions.           |
| `A(1:2:end, 1:2:end) = 0;`               | Not directly available, can be implemented using custom functions or manual iteration.           |
| `A(1:2:end, 1:2:end) = B;`               | Not directly available, can be implemented using custom functions or manual iteration.           |
| `A(1:2:end, end:-1:1) = [];`             | Not directly available, can be implemented using block operations or custom functions.           |
| `A(1:3) = [];`                           | Not directly available, can be implemented using block operations or custom functions.           |
| `A(1:3) = [4; 5; 6];`                    | `A.segment(0, 3) = Eigen::VectorXd::LinSpaced(3, 4, 6);`                                         |
| `A(1:3) = 0;`                            | `A.segment(0, 3).setZero();`                                                                     |
| `A(1:3, 1:3) = 0;`                       | `A.block(0, 0, 3, 3).setZero();`                                                                 |
| `A(1:3, 1:3) = B(2:4, 2:4);`             | `A.block(0, 0, 3, 3) = B.block(1, 1, 3, 3);`                                                     |
| `A(end:-1:1, :) = [];`                   | Not directly available, can be implemented using block operations or custom functions.           |
| `AA = min(A, [], 'all');`                | `double min_all = A.minCoeff();`                                                                 |
| `AB = min(A, [], 2);`                    | `Eigen::VectorXd min_row = A.rowwise().minCoeff();`                                              |
| `AC = mean(A, 'all');`                   | `double mean_all = A.mean();`                                                                    |
| `AD = mean(A, 2);`                       | `Eigen::VectorXd mean_row = A.rowwise().mean();`                                                 |
| `AE = std(A, 'all');`                    | `double std_dev_all = sqrt((A.array() - A.mean()).square().sum() / (A.size() - 1));`             |
| `AF = std(A, 0, 2);`                     | `Eigen::VectorXd std_dev_row = ((A.array().rowwise() - A.rowwise().mean()).square().sum() / (A.cols() - 1)).sqrt();` |
| `AG = linspace(0, 10, 5);`               | `Eigen::VectorXd AG = Eigen::VectorXd::LinSpaced(5, 0, 10);`                                      |
| `AH = logspace(0, 10, 5);`               | Not directly available, can be implemented using Eigen::VectorXd::LinSpaced and manual transformation. |
| `AI = zeros(3, 3);`                      | `Eigen::MatrixXd AI = Eigen::MatrixXd::Zero(3, 3);`                                               |
| `AJ = ones(3, 3);`                       | `Eigen::MatrixXd AJ = Eigen::MatrixXd::Ones(3, 3);`                                               |
| `AK = eye(3);`                            | `Eigen::MatrixXd AK = Eigen::MatrixXd::Identity(3, 3);`                                           |
| `AL = rand(3, 3);`                       | `Eigen::MatrixXd AL = Eigen::MatrixXd::Random(3, 3);`                                             |
| `AM = randn(3, 3);`                      | `Eigen::MatrixXd AM(3, 3); AM.setRandom();`                                                       |
| `AN = magic(3);`                         | Not directly available, can be implemented using custom functions or manual initialization.      |
| `AO = hilb(3);`                          | Not directly available, can be implemented using custom functions or manual initialization.      |
| `AP = pascal(3);`                        | Not directly available, can be implemented using custom functions or manual initialization.      |
| `AQ = vander(v);`                        | Not directly available, can be implemented using custom functions or manual initialization.      |
| `AR = toeplitz(v);`                      | Not directly available, can be implemented using custom functions or manual initialization.      |
| `AS = hadamard(3);`                      | Not directly available, can be implemented using custom functions or manual initialization.      |
| `AT = gallery('durer');`                 | Not directly available, can be implemented using custom functions or loading external image data. |
| `B = [5, 6; 7, 8];`                      | `Eigen::MatrixXd B(2, 2); B << 5, 6, 7, 8;`                                                      |
| `B = A * C;`                              | `Eigen::MatrixXd B = A * C;`                                                                    |
| `B = A + 5;`                              | `Eigen::MatrixXd B = A.array() + 5;`                                                             |
| `B = A + repmat(v, 1, A.cols());`         | `Eigen::MatrixXd B = A.array().colwise() + v.array();`                                           |
| `B = A(2:3, 1:2);`                        | `Eigen::MatrixXd B = A.block(1, 0, 2, 2);`                                                       |
| `B = A';`                                 | `Eigen::MatrixXd B = A.transpose();`                                                            |
| `B = inv(A);`                             | `Eigen::MatrixXd B = A.inverse();`                                                              |
| `C = [A, B; B, A];`                       | `Eigen::MatrixXd C(A.rows() + B.rows(), A.cols() + B.cols()); C << A, B, B, A;`                 |
| `C = A - scalar;`                         | `Eigen::MatrixXd C = A.array() - scalar;`                                                        |
| `C = A * B + D / scalar;`                | `Eigen::MatrixXd C = (A * B).array() + D.array() / scalar;`                                       |
| `C = A * B + diag(v) * scalar;`          | `Eigen::MatrixXd C = (A * B).array() + v.array().matrix().asDiagonal() * scalar;`                 |
| `C = A * B + scalar;`                    | `Eigen::MatrixXd C = (A * B).array() + scalar;`                                                  |
| `C = A * repmat(v, 1, size(A, 2));`      | Not directly available, can be implemented using custom functions or block operations.             |
| `C = A * repmat(v, size(A));`            | Not directly available, can be implemented using custom functions or block operations.             |
| `C = A * repmat(v, size(A, 1), 1);`      | Not directly available, can be implemented using custom functions or block operations.             |
| `C = A * scalar;`                         | `Eigen::MatrixXd C = A * scalar;`                                                                |
| `C = A .* B;`                             | `Eigen::MatrixXd C = A.array() * B.array();`                                                    |
| `C = A .* B;`                             | `Eigen::MatrixXd C = A.array() * B.array();`                                                    |
| `C = A ./ B;`                             | `Eigen::MatrixXd C = A.array() / B.array();`                                                    |
| `C = A / scalar;`                         | `Eigen::MatrixXd C = A / scalar;`                                                                |
| `C = A + B * scalar;`                    | `Eigen::MatrixXd C = A + B.array() * scalar;`                                                    |
| `C = A + repmat(v, 1, size(A, 2));`      | `Eigen::MatrixXd C = A.array().colwise() + v.array();`                                            |
| `C = A + repmat(v, size(A));`            | Not directly available, can be implemented using block-wise replication or custom functions.      |
| `C = A + repmat(v, size(A, 1), 1);`      | `Eigen::MatrixXd C = A.array().rowwise() + v.array();`                                            |
| `C = A + scalar;`                         | `Eigen::MatrixXd C = A.array() + scalar;`                                                        |
| `C = A > B;`                              | `Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> C = A.array() > B.array();`                 |
| `C = A(:, [1, 3]);`                       | `Eigen::MatrixXd C = A.cols({1, 3});`                                                             |
| `C = A(:, 1) * B(1, :);`                 | `Eigen::MatrixXd C = A.col(0) * B.row(0);`                                                        |
| `C = A(:, 1:2:end);`                      | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A(:, end:-1:1);`                     | `Eigen::MatrixXd C = A.rowwise().reverse();`                                                     |
| `C = A(:, end:-2:1);`                     | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A([1, 3], :);`                       | `Eigen::MatrixXd C = A.rows({1, 3});`                                                             |
| `C = A([1, 3], [1, 3]);`                  | `Eigen::MatrixXd C = A.rows({1, 3}).cols({1, 3});`                                                |
| `C = A(1, :) .* B(:, 1);`                 | `Eigen::MatrixXd C = A.row(0).array() * B.col(0).array();`                                       |
| `C = A(1:2:end, :);`                      | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A(1:2:end, 1:2:end);`                | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A(1:2:end, 1:2:end);`                | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A(1:2:end, end:-1:1);`               | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A(1:2:end, end:-2:1);`               | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A(1:3, [1, 3]);`                     | `Eigen::MatrixXd C = A.topLeftCorner(3, 2);`                                                      |
| `C = A(1:end, 2:end);`                    | `Eigen::MatrixXd C = A.bottomRightCorner(A.rows() - 1, A.cols() - 1);`                            |
| `C = A(end:-1:1, :);`                     | `Eigen::MatrixXd C = A.colwise().reverse();`                                                     |
| `C = A(end:-2:1, 1:2:end);`               | Not directly available, can be implemented using custom functions or manual iteration.           |
| `C = A.^2;`                               | `Eigen::MatrixXd C = A.array().square();`                                                       |
| `C = A.^B;`                               | Not directly available, can be implemented using custom functions or element-wise operations.   |
| `C = abs(A);`                             | `Eigen::MatrixXd C = A.array().abs();`                                                          |
| `C = acos(A);`                            | `Eigen::MatrixXd C = A.array().acos();`                                                         |
| `C = asin(A);`                            | `Eigen::MatrixXd C = A.array().asin();`                                                         |
| `C = atan(A);`                            | `Eigen::MatrixXd C = A.array().atan();`                                                         |
| `C = atan2(A, B);`                        | Not directly available, can be implemented using custom functions or element-wise operations.   |
| `C = cat(1, A, B);`                      | `Eigen::MatrixXd C(A.rows() + B.rows(), A.cols()); C << A, B;`                                   |
| `C = ceil(A);`                            | `Eigen::MatrixXd C = A.array().ceil();`                                                         |
| `C = conv(A, B);`                         | Not directly available, convolution can be implemented using custom functions or libraries like Eigen's convolve. |
| `C = cos(A);`                             | `Eigen::MatrixXd C = A.array().cos();`                                                          |
| `C = cosh(A);`                            | `Eigen::MatrixXd C = A.array().cosh();`                                                         |
| `C = cross(A, B);`                        | Not directly available, can be implemented using custom functions or element-wise operations.   |
| `C = dot(A, B);`                          | `double dot_product = A.dot(B);`                                                                  |
| `C = exp(A);`                             | `Eigen::MatrixXd C = A.array().exp();`                                                          |
| `C = eye(3) * scalar;`                    | `Eigen::MatrixXd C = Eigen::MatrixXd::Identity(3, 3) * scalar;`                                  |
| `C = fft(A);`                             | Not directly available, FFT can be implemented using libraries like FFTW or custom functions.    |
| `C = floor(A);`                           | `Eigen::MatrixXd C = A.array().floor();`                                                        |
| `C = ifft(A);`                            | Not directly available, inverse FFT can be implemented using libraries like FFTW or custom functions. |
| `C = isempty(A);`                         | `bool is_empty = A.size() == 0;`                                                                  |
| `C = isequal(A, B);`                      | Not directly available, can be implemented using custom functions or manual comparison.           |
| `C = isequaln(A, B);`                     | Not directly available, can be implemented using custom functions or manual comparison.           |
| `C = isfinite(A);`                        | `Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> C = A.array().isFinite();`                  |
| `C = isnan(A);`                           | `Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> C = A.array().isNaN();`                     |
| `C = kron(A, B);`                         | Not directly available, Kronecker product can be implemented using custom functions or manual manipulation. |
| `C = log(A);`                             | `Eigen::MatrixXd C = A.array().log();`                                                          |
| `C = max(A, [], 2);`                      | `Eigen::VectorXd row_max = A.rowwise().maxCoeff();`                                              |
| `C = max(A, [], 'all');`                  | `double max_val = A.maxCoeff();`                                                                  |
| `C = mean(A);`                            | `double mean = A.mean();`                                                                        |
| `C = norm(A);`                            | `double norm = A.norm();`                                                                         |
| `C = norm(A, 1);`                         | `double l1_norm = A.lpNorm<1>();`                                                                 |
| `C = norm(A, 'fro');`                     | `double frobenius_norm = A.norm();`                                                               |
| `C = norm(A, inf);`                       | `double linf_norm = A.lpNorm<Eigen::Infinity>();`                                                |
| `C = numel(A);`                           | `int num_elements = A.size();`                                                                    |
| `C = polyfit(x, y, degree);`              | Not directly available, polynomial fitting can be implemented using libraries like Eigen's PolynomialSolver or custom functions. |
| `C = polyval(p, x);`                      | Not directly available, polynomial evaluation can be implemented using custom functions or manual calculation. |
| `C = repmat(A, 2, 3);`                    | Not directly available, can be implemented using nested loops or block-wise replication.          |
| `C = repmat(v, 1, size(A, 2)) * A;`      | Not directly available, can be implemented using custom functions or block operations.             |
| `C = repmat(v, size(A)) * A;`            | Not directly available, can be implemented using custom functions or block operations.             |
| `C = repmat(v, size(A, 1), 1) * A;`      | Not directly available, can be implemented using custom functions or block operations.             |
| `C = round(A);`                           | `Eigen::MatrixXd C = A.array().round();`                                                        |
| `C = scalar * A;`                         | `Eigen::MatrixXd C = scalar * A;`                                                                |
| `C = scalar / A;`                         | Not directly available, can be implemented using element-wise division by scalar.                 |
| `C = sign(A);`                            | Not directly available, can be implemented using custom functions or element-wise operations.   |
| `C = sin(A);`                             | `Eigen::MatrixXd C = A.array().sin();`                                                          |
| `C = sinh(A);`                            | `Eigen::MatrixXd C = A.array().sinh();`                                                         |
| `C = size(A);`                            | `Eigen::Index rows = A.rows(); Eigen::Index cols = A.cols();`                                     |
| `C = sort(A);`                            | `Eigen::MatrixXd C = A; C.sort();`                                                               |
| `C = sortrows(A);`                        | Not directly available, can be implemented using custom functions or manual manipulation of rows. |
| `C = sqrt(A);`                            | `Eigen::MatrixXd C = A.array().sqrt();`                                                         |
| `C = std(A);`                             | `double std_dev = sqrt((A.array() - A.mean()).square().sum() / (A.size() - 1));`                |
| `C = sum(A);`                             | `double sum = A.sum();`                                                                          |
| `C = sum(A, 2, 'omitnan');`               | `Eigen::VectorXd row_sum = A.rowwise().sum();`                                                  |
| `C = sum(A, 'all');`                      | `double sum = A.sum();`                                                                           |
| `C = tan(A);`                             | `Eigen::MatrixXd C = A.array().tan();`                                                          |
| `C = tanh(A);`                            | `Eigen::MatrixXd C = A.array().tanh();`                                                         |
| `col_sum = sum(A, 1);`                    | `Eigen::VectorXd col_sum = A.colwise().sum();`                                                  |
| `D = cat(2, A, B);`                      | `Eigen::MatrixXd D(A.rows(), A.cols() + B.cols()); D << A, B;`                                   |
| `D = diag([1, 2, 3]);`                    | `Eigen::VectorXd d(3); d << 1, 2, 3; Eigen::MatrixXd D = d.asDiagonal();`                        |
| `E = horzcat(A, B);`                     | `Eigen::MatrixXd E(A.rows(), A.cols() + B.cols()); E << A, B;`                                   |
| `element = A(1, 2);`                      | `double element = A(1, 2);`                                                                      |
| `F = vertcat(A, B);`                     | `Eigen::MatrixXd F(A.rows() + B.rows(), A.cols()); F << A, B;`                                   |
| `G = reshape(A, 1, []);`                 | Not directly available, can be implemented using Eigen::Map or manual manipulation of indices.  |
| `H = reshape(A, 2, 2);`                  | Not directly available, can be implemented using Eigen::Map or manual manipulation of indices.  |
| `I = reshape(A, [], 1);`                 | Not directly available, can be implemented using Eigen::Map or manual manipulation of indices.  |
| `J = reshape(A, 1, 4);`                  | Not directly available, can be implemented using Eigen::Map or manual manipulation of indices.  |
| `K = reshape(A, 4, 1);`                  | Not directly available, can be implemented using Eigen::Map or manual manipulation of indices.  |
| `L = flip(A, 1);`                        | `Eigen::MatrixXd L = A.rowwise().reverse();`                                                     |
| `M = flip(A, 2);`                        | `Eigen::MatrixXd M = A.colwise().reverse();`                                                     |
| `N = flipud(A);`                         | `Eigen::MatrixXd N = A.rowwise().reverse();`                                                     |
| `O = fliplr(A);`                         | `Eigen::MatrixXd O = A.colwise().reverse();`                                                     |
| `P = rot90(A);`                          | Not directly available, can be implemented using custom functions or manual manipulation.        |
| `Q = rot90(A, 2);`                       | Not directly available, can be implemented using custom functions or manual manipulation.        |
| `R = rot90(A, -1);`                      | Not directly available, can be implemented using custom functions or manual manipulation.        |
| `row_sum = sum(A, 2);`                    | `Eigen::VectorXd row_sum = A.rowwise().sum();`                                                  |
| `S = tril(A);`                           | `Eigen::MatrixXd S = A.triangularView<Eigen::Lower>();`                                          |
| `T = triu(A);`                           | `Eigen::MatrixXd T = A.triangularView<Eigen::Upper>();`                                          |
| `U = toeplitz([1, 2, 3]);`               | Not directly available, can be implemented using custom functions or manual manipulation.        |
| `v = [1; 2; 3];`                          | `Eigen::VectorXd v(3); v << 1, 2, 3;`                                                            |
| `V = circshift(A, [1, 1]);`              | Not directly available, can be implemented using custom functions or manual manipulation.        |
| `W = sum(A, 'all');`                     | `double sum_all = A.sum();`                                                                      |
| `x = A \ b;`                              | `Eigen::VectorXd x = A.ldlt().solve(b);`                                                        |
| `X = sum(A, 'omitnan');`                 | `Eigen::VectorXd sum_row = A.rowwise().sum();`                                                   |
| `Y = max(A, [], 'all');`                  | `double max_all = A.maxCoeff();`                                                                 |
| `Z = max(A, [], 2);`                     | `Eigen::VectorXd max_row = A.rowwise().maxCoeff();`                                              |


