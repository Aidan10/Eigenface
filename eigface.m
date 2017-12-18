clear;
%Initialize variables and constants
people=40;%Number of different people to select max 7 for training set
poses=7;%Number of poses per person
k = 250;%Dimensionality reduction, must be greater than 5
pick = 1;%Individual person picked to reconstruct
%pv = [1 8 15 22 29 36 43 50 57 64 71];%Vector of face index in training set
%pv2 = [1 4 7 10 13 16 19 22 25 28 31];%Vector of face index in testing set
[pv,pv2] = pickvec(people);
m = people*poses;%Number of images

%Reads in the data and splits it into a training and test set
[X,Xtest,r,c] = read_data(people,poses);

%mean of input faces
Xmean = mean(X,2);
%Subtract the mean from the faces to feature normalize
Xnorm = X - Xmean;

%Declare initial guess eigenvector for power method
V = ones(m,1);
%Declare max iterations and tolerance for power method
it_max = 500;
tol = 1e-10;

%Acquire the first k ordered eigenvectors and eigenvalues utilizing svd/qr/power
[eigvec_svd,eigval_svd] = pca_svd(Xnorm,k); 
[eigvec_qr,eigval_qr] = PCA_QR(Xnorm,k);
[eigvec_pow,eigval_pow] = pca_pow(Xnorm,V,it_max,tol,k);


%Utilizing matlabs PCA function for comparison
[evectors, score, evalues] = pca(Xnorm');
evectors = evectors(:,1:k);
%Display at most the first 5 eigenvectors/eigenfaces
figure(1);hold on
for n = 1:5
    subplot(4, 5, n);
    evector = mat2gray(reshape(evectors(:,n), [112 92]));
    imshow(evector);
    title(strcat('PCA ',int2str(n)))
end

%Display the eigenvectors for SVD
for n = 6:10
    subplot(4, 5, n);
    evector = mat2gray(reshape(eigvec_svd(:,n), [112 92]));
    imshow(evector);
    title(strcat('SVD ',int2str(n-5)))
end

%Display eigenfaces for QR method
for n = 11:15
    subplot(4, 5, n);
    evector = mat2gray(reshape(eigvec_qr(:,n), [112 92]));
    imshow(evector);
    title(strcat('SVD ',int2str(n-10)))
end

%Display eigenfaces for Power method
for n = 16:20
    subplot(4, 5, n);
    evector = mat2gray(reshape(eigvec_pow(:,n), [112 92]));
    imshow(evector);
    title(strcat('POW ',int2str(n-10)))
end
hold off


%Acquire k weights for each of the faces
W = weights(Xnorm,evectors,k);%PCA weights 
W_svd = weights(Xnorm,eigvec_svd,k);%SVD weights 
W_qr = weights(Xnorm,eigvec_qr,k);%QR weights 
W_pow = weights(Xnorm,eigvec_pow,k);%QR weights 

%Plot the image of the picked person being reconstructed for all methods
figure(2);hold on
subplot(5,1,1)
imshow(mat2gray(reshape(X(:,pv(pick)),[112 92])))
title('Choosen Face to Reconstruct')
%Plots for reconstruction from weights for pca
recon = reconstruct(W(pv(pick),:),Xmean,evectors);
subplot(5,1,2)
imshow(mat2gray(reshape(recon(pv(pick),:),[112 92])))
title('PCA')
%Plots for reconstruction from weights for svd
recon_svd = reconstruct(W_svd(pv(pick),:),Xmean,eigvec_svd);
subplot(5,1,3)
imshow(mat2gray(reshape(recon_svd(pv(pick),:),[112 92])))
title('SVD')
%Plots for reconstruction from weights for qr
recon_qr = reconstruct(W_qr(pv(pick),:),Xmean,eigvec_qr);
subplot(5,1,4)
imshow(mat2gray(reshape(recon_qr(pv(pick),:),[112 92])))
title('QR')
%Plots for reconstruction from weights for power
recon_pow = reconstruct(W_pow(pv(pick),:),Xmean,eigvec_pow);
subplot(5,1,5)
imshow(mat2gray(reshape(recon_pow(pv(pick),:),[112 92])))
title('POW')
hold off

%normalize the test set
Xtestnorm = Xtest-Xmean;
figure(10)
imshow(mat2gray(reshape(Xmean,[112 92])))
%Acquire the weights or the new projection of the test set 
W_test = weights(Xtestnorm,eigvec_qr,k);

%Compare the euclidean distances of the test set projection for the
%specific person selected
[mindist,person,dist] = match_face(W_qr,W_test(pv2(pick),:));

%Print both the search image and the closest training set image
figure(3);hold on
subplot(1,2,1)
imshow(mat2gray(reshape(Xtest(:,pv2(pick)),[112 92])))
title('Face from Test Set')
subplot(1,2,2)
imshow(mat2gray(reshape(X(:,person),[112 92])))
title('Match from Train Set')
hold off
