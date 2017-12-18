function [mindist,person,dist] = match_face(W,w)
%Function returns the index of the person with the closest match
%W: the weights(projection in the new subspace) of everyone in the training
%set. 
%w: The weights for the individuals test face

dist = sqrt(sum((abs(W)-abs(w)).^2,2));
[mindist,person] = min(dist);%Returns the minimum distance and the index of the person who matched

end