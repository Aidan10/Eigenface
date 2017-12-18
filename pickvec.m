%This function picks a selected range of a subjects photos
%to use for training or testing
%
%Input: Person of interest person 
%
%Outputs:training vector set vectrain, testing vector set vectest
function [vectrain,vectest] = pickvec(person)

%Initialize vectors as the size of people included in training/testing
vectrain = zeros(person,1);
vectest = zeros(person,1);

vectrain(1) = 1;%Initialize training index vector with the index for the first face
%for loop to fill in vector with ramining indexes for faces in training set
for i = 2:person
   vectrain(i) = vectrain(i-1)+7;
end

vectest(1) = 1;%Initialize testing index vector with the index for the first face
%for loop to fill in vector with ramining indexes for faces in testing set
for i = 2:person
   vectest(i) = vectest(i-1)+3;
end

end
