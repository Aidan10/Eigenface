%main script that produces key results derived from EigenFace experiment
%CSE 700: Scientific Computation
%Date: 18/12/17
%Students: Ola Oni, John Heisey, Aidan Ross

%Calling eigenface script which produces 1. first 5 eigenfaces for all the
%methods 2. the reconstruction of a chosen face 3. plot with face from test
%site and recognized face from the training set
eigface

%Prompt user so they can exit the program before lengthy computation
prompt = 'Computation for the Eigenface CPU time (Takes Approx. 60 Seconds). Would you like to continue (Y/N)? ';
str = input(prompt,'s');
if isempty(str)
    str = 'N';
end

if or(str=='n',str=='N')
    return;
end

%eigentime script returns the plot of cputime with respect to the size of the data matrix 
eigentime

%Prompt user so they can exit the program before lengthy computation
prompt = 'Last computation for the Eigenface Accuracy (Takes Approx 150 seconds). Would you like to continue (Y/N)? ';
str = input(prompt,'s');
if isempty(str)
    str = 'N';
end

if or(str=='n',str=='N')
    return;
end

%eigface_accuracy generates a plot of the percent accuracy in recognition
%for the different PCA methods
eigface_accuracy