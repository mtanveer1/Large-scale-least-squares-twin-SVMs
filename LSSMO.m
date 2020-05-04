function bestx = LSSMO(Q,eps,c,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     INPUTS:
%           Q   :-  positive definite matrix
%           eps :-  termination condition (tolerence)
%           c   :-  weight to be tunned (from cross validation)
%           v   :-  vector
%
%   OUTPUT:
%           bestx.
%% for example:
% Q=[1 0 0;0 1 0;0 0 1];
%eps=0.0001;
%c=2;
%v=[0 1 1];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[no_row,m] = size(Q);
x = zeros(no_row,1);  %%%initializing the vector vector x
F=[];
D=[];
%c3=1;
%v=[0 1 1];
%c=c3
%% F is vector of differentials of original problem with repect to each component%%%%%%%
% *formation of vector F* %%%%%%%%%%%
for i = 1:no_row
    Fi = -x'*Q(:,i)-c*v(i);
    Di=Fi*Fi/(2*Q(i,i));
    F(i) = Fi;
    D(i) = Di;
end
normF = norm(F);
%% norm of F should be zero or close to zero (less than the defined tolerence)
% *loop for terminating the condition* %%%%%%%%%
iter=0;
while normF>eps*no_row && iter<500
    [Max,i]=max(D);
    t=F(i)/Q(i,i);
    x(i)=x(i)+t;  %%% updating the variable x
    for i = 1:no_row
        Fi = -x'*Q(:,i)-c*v(i);
        Di=Fi*Fi/(2*Q(i,i));
        F(i) = Fi;
        D(i) = Di;
    end
    normF = norm(F);
end
bestx=x;
end