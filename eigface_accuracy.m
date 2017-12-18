%This file utilizes the facial recognition tools and measures the 
%accuracy of recognition for each method.
%
clear;
%Initialize variables and constants
idx_range = 0:10:270;
accuracy_mat = [];
for idx = idx_range
    people=40;%Number of different people to select max 7 for training set
    poses=7;%Number of poses per person
    k = idx;%Dimensionality reduction, must be greater than 5
    pick = 1;%Individual person picked to reconstruct
    %pv = [1 8 15 22 29 36 43 50 57 64 71];%Vector of face index in training set
    %pv2 = [1 4 7 10 13 16 19 22 25 28 31];%Vector of face index in testing set
    [pv,pv2] = pickvec(people);
    m = people*poses;%Number of images
    score_mat = zeros(people,4);
    %Reads in the data and splits it into a training and test set
    [X,Xtest,r,c] = read_data(people,poses);

    %mean of input faces
    Xmean = mean(X,2);
    %Subtract the mean from the faces to feature normalize
    Xnorm = X - Xmean;

    %%%%John addition%%%%%%
    %A = cov(Xnorm');

    %A = A./(m-1);
    V = ones(m,1);
    it_max = 500;
    tol = 1e-10;

    % size(Xnorm)
    % size(V)
    %%%%%%%%%%%%%%%%%%%%


    %Acquire the first k ordered eigenvectors and eigenvalues utilizing svd and qr
    [eigvec_svd,eigval_svd] = pca_svd(Xnorm,k); 
    [eigvec_qr,eigval_qr] = PCA_QR(Xnorm,k);

    %%%%%John addition%%%%%%%%%%%
    [eigvec_pow,eigval_pow] = pca_pow(Xnorm,V,it_max,tol,k);
    size(eigvec_qr);
    size(eigvec_pow);
    %%%%%%%%%%%%%%%%

    %Utilizing matlabs PCA function for comparison
    [evectors, score, evalues] = pca(Xnorm');
    evectors = evectors(:,1:k);

V = ones(m,1);
it_max = 500;
tol = 1e-10;

%Acquire the first k ordered eigenvectors and eigenvalues utilizing svd and qr
[eigvec_svd,eigval_svd] = pca_svd(Xnorm,k); 
[eigvec_qr,eigval_qr] = PCA_QR(Xnorm,k);
[eigvec_pow,eigval_pow] = pca_pow(Xnorm,V,it_max,tol,k);



    %Acquire k weights for each of the faces
    W = weights(Xnorm,evectors,k);%PCA weights 
    W_svd = weights(Xnorm,eigvec_svd,k);%SVD weights 
    W_qr = weights(Xnorm,eigvec_qr,k);%QR weights 
    W_pow = weights(Xnorm,eigvec_pow,k);%Power method weights 

    %normalize the test set
    Xtestnorm = Xtest-Xmean;

    %Start loop
    vecmat = {evectors,eigvec_svd,eigvec_qr,eigvec_pow};
    wmat = {W,W_svd,W_qr,W_pow};
    for i = 1:4
        for j=1:people
            %Acquire the weights or the new projection of the test set 
            W_test = weights(Xtestnorm,cell2mat(vecmat(i)),k);

            %Compare the euclidean distances of the test set projection for the
            %specific person selected
            [mindist,person,dist] = match_face(cell2mat(wmat(i)),W_test(pv2(j),:));

            %Measure the accuracy in recognition for each method
            if and((person < pv(j)+8),(person >= pv(j)))
                score_mat(j,i) = 1;
            end
        end
    end


    temp = sum(score_mat,1)/people*100;
    accuracy_mat = [accuracy_mat;temp];

end

    %Plots the accuracy for all the methods
    figure(1);
    plot(idx_range,accuracy_mat(:,1),idx_range,accuracy_mat(:,2),idx_range,accuracy_mat(:,3),idx_range,accuracy_mat(:,4))
    legend({'Matlab PCA','SVD Method','QR Method','Power Method'},'location','southeast')
    title('Accuracy in Recognition for PCA methods')
    xlabel('Eigenfaces')
    ylabel('Percent Accuracy')



