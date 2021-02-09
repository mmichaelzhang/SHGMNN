load ("../Data/A.dat");
load ("../Data/b.dat");
A = A;
A = spconvert(A);

lambda_max = norm( A'*b, 'inf' );
lambda = 0*lambda_max;

[x history] = las(A, b, lambda, 1.0, 1.0);
save('../Data/Inference.txt', 'x', '-ASCII');