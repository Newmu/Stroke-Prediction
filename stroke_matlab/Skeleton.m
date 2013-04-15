function s = Skeleton(a, varargin)

% function s = Skeleton(a);
% function s = Skeleton(a, iterations);
%
% Skeletonizing a binary image using Zhang and Suen's method from
%
%    "A fast parallel algorithm for thinning digital patterns"
%    Comm ACM, Vol. 7, No. 23, pp. 326 -- 329, 1985.
%
% as described in Gonzales & Wintz.
%
% 'a' should contain binary data with 0 considered to be the
% background and 1 the foreground (or object to be thinned).
%
% 'iterations' is the number of iterations of the algorithm to
% perform.  Each iteration can only strip away a one pixel wide
% boundary. If no 'iterations' argument is supplied, the function
% iterates until the result converges.
%
% This is a straightforward implementation and as such is very slow.
% There is much scope for optimisation.
%
% Gary Dickson <moc.liamtoh@noskcidyrag>
%

% Check arguments
%
if nargin == 1
   iterations = -1; % Flag to iterate until convergence
elseif nargin == 2
   iterations = varargin(1); % Specified number of iterations
   iterations = iterations{1}; % Want plain old number, not a cell
else
   disp('Too many arguments supplied.');
   s = a;
   return;
end

% Check input data
%
if (min(min(a)) < 0) | (max(max(a)) > 1)
   disp('input not binary')
   s = a;
   return;
end


[h w] = size(a);
s = a;
it = 1;
prevsum = 0;
while 1

   % Step 1 (thinning top and left sides)
   %
   m = ones([h w]);
   for j = 2 : h-1
      for i = 2 : w-1

         if (s(j, i) == 1)
            condA = sum(sum(s(j-1 : j+1, i-1 : i+1))) - s(j, i);

            condB = Num01Transitions(s, j, i);

            condC = s(j-1, i) * s(j, i+1) * s(j+1, i); % p2 * p4 * p6

            condD = s(j, i+1) * s(j+1, i) * s(j, i-1); % p4 * p6 * p8

            if (condA >= 2) & (condA <= 6) & (condB == 1) & (condC == 0) & (condD == 0)
               m(j, i) = 0;
            end
         end

      end % i
   end % j

   s = s .* m;


   % Step 2 (thinning bottom and right sides)
   %
 
   m = ones([h w]);
   for j = 2 : h-1
      for i = 2 : w-1

         if (s(j, i) == 1)
            condA = sum(sum(s(j-1 : j+1, i-1 : i+1))) - s(j, i);

            condB = Num01Transitions(s, j, i);

            condC = s(j-1, i) * s(j, i+1) * s(j, i-1); % p2 * p4 * p8

            condD = s(j-1, i) * s(j+1, i) * s(j, i-1); % p2 * p6 * p8

            if (condA >= 2) & (condA <= 6) & (condB == 1) & (condC == 0) & (condD == 0)
               m(j, i) = 0;
            end
         end

      end % i
   end % j

   s = s .* m;


   % Time to stop? As points are always being removed and never added, we can
   % check if the method has converged by checking if the sum of the image
   % values is the same for two successive iterations.
   %
   newsum = sum(sum(s));

   if (newsum == prevsum) & (iterations == -1)
      break;
   end;

   if (it >= iterations) & (iterations ~= -1)
      break;
   end

   it = it + 1;
   prevsum = newsum;

end % it

%------------------------------------------------------------------------------%
%                                                                              %
%------------------------------------------------------------------------------%
% Count the number of 0 -> 1 transitions that occur while traversing the eight
% neighbours of the point of interest.
%
%       p9 p2 p3
%       p8 p1 p4
%       p7 p6 p3
%

function Nt = Num01Transitions(c, j, i)

p = [c(j-1, i) c(j-1, i+1) c(j, i+1) c(j+1, i+1)];
p = [p c(j+1, i) c(j+1, i-1) c(j, i-1) c(j-1, i-1) c(j-1, i)];

pp = zeros(1, 8);
for k = 1 : 8
   pp(k) = p(k+1) - p(k);
end

Nt = sum(pp == 1);

%------------------------------------------------------------------------------%
%                                                                              %
%------------------------------------------------------------------------------%
% End of file    