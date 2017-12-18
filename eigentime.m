mat = linspace(10^2,2*10^3,10);
mat = floor(mat);
qr = [];
svdmat = [];
pcamat = [];
powmat = [];
it_max = 500;
tol = 1e-10;

for i = 1:length(mat)
   k = 2;
   A = rand(mat(i));
   [w, h] = size(A);
   mu = mean(m);
   V = ones(w,1);
   
   t = cputime; 
   PCA_QR2(A,k,w,mu);
   qr = [qr cputime-t];
   
   t = cputime;
   pca_svd(A,k);
   svdmat = [svdmat cputime-t];
  
   t = cputime;
   pca(A);
   pcamat = [pcamat cputime-t];
   
   t = cputime;
   pca_pow(A,V,it_max,tol,k);
   powmat = [powmat cputime-t];
   
end

xx = linspace(0,1,length(mat));
plot(xx,qr,xx,svdmat,xx,pcamat,xx,powmat)
legend('qr','svd','pca','pow')