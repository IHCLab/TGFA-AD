%=====================================================================
% Programmer: Chia-Hsiang Lin (Steven)
% E-mail: chiahsiang.steven.lin@gmail.com
% Date: 2015/11/11
% -------------------------------------------------------
% Reference:
% C.-H. Lin, C.-Y. Chi, Y.-H. Wang, and T.-H. Chan,
% ``A fast hyperplane-based minimum-volume enclosing simplex algorithm for blind hyperspectral unmixing,"
% arXiv preprint arXiv:1510.08917, 2015.
%======================================================================
% A fast implementation of Hyperplane-based Craig-Simplex-Identification algorithm
% [A_est, S_est, time] = HyperCSI(X,N)
%======================================================================
%  Input
%  X is M-by-L data matrix, where M is the number of spectral bands and L is the number of pixels.
%  N is the number of endmembers.
%----------------------------------------------------------------------
%  Output
%  A_est is M-by-N mixing matrix whose columns are estimated endmember signatures.
%  S_est is N-by-L source matrix whose rows are estimated abundance maps.
%  time is the computation time (in secs).
%========================================================================

function [A_est, S_est] = HyperCSI(X,N,eta)
t0 = clock;

%------------------------ Step 1 ------------------------
[M L ] = size(X);
d = mean(X,2);
U = X-d*ones(1,L);
[eV D] = eig(U*U');
C = eV(:,M-N+2:end);
Xd = C'*(X-d*ones(1,L)); % dimension reduced data (Xd is (N-1)-by-L)

%------------------------ Step 2 ------------------------
alpha_tilde = SPA(Xd,L,N); % the identified purest pixels

%------------------------ Step 3 ------------------------
for i = 1:N
    bi_tilde(:,i) = compute_bi(alpha_tilde,i,N); % obtain bi_tilde
end

r = (1/2)*norm(alpha_tilde(:,1)-alpha_tilde(:,2));
for i = 1:N-1
    for j = i+1:N
        dist_ai_aj(i,j) = norm(alpha_tilde(:,i)-alpha_tilde(:,j));
        if (1/2)*dist_ai_aj(i,j) < r
            r = (1/2)*dist_ai_aj(i,j); % compute radius of hyperballs
        end
    end
end

Xd_divided_idx = zeros(L,1);
radius_square = r^2;
for k = 1:N
    [IDX_alpha_i_tilde]= find( sum(  (Xd- alpha_tilde(:,k)*ones(1,L) ).^2,1  )  < radius_square );

    Xd_divided_idx(IDX_alpha_i_tilde) = k ; % compute the hyperballs
end

%------------------------ Step 4 ------------------------
for i = 1:N
    Hi_idx = setdiff([1:N],[i]);
    
    for k = 1:1*(N-1)
        Ri_k = Xd(:,( Xd_divided_idx == Hi_idx(k) ));
        (find(Xd_divided_idx == Hi_idx(k) ));
        [val idx] = max(bi_tilde(:,i)'*Ri_k);
        pi_k(:,k) = Ri_k(:,idx); % find N-1 affinely independent points for each hyperplane
    end
    b_hat(:,i) = compute_bi([pi_k alpha_tilde(:,i)],N,N);
    h_hat(i,1) = max(b_hat(:,i)'*Xd);
end

%------------------------ Step 5 & Step 6 ------------------------
comm_flag = 1;
% comm_flag = 1 in noisy case: bring hyperplanes closer to the center of data cloud
% comm_flag = 0 when no noise: Step 5 will not be performed (and hence c = 1)

for i = 1:N
    bbbb = b_hat;
    ccconst = h_hat;
    bbbb(:,i) = [];
    ccconst(i) = [];
    alpha_hat(:,i) = pinv(bbbb')*ccconst;
end

if comm_flag == 1
    VV = C*alpha_hat;
    UU = d*ones(1,N);
    closed_form_optval = max( 1 , max(max( (-VV) ./ UU)) ); % c' in Step 5
    c = closed_form_optval/eta;
    h_hat = h_hat/c;
    alpha_hat = alpha_hat/c;
end
A_est = C * alpha_hat + d * ones(1,N); % endmemeber estimates

%------------------------ Step 7 ------------------------
% Step 7 can be removed if the user do not need abundance estimation
S_est = ( h_hat*ones(1,L)- b_hat'*Xd   ) ./ ( (  h_hat - sum( b_hat.*alpha_hat )' ) *ones(1,L) );
S_est(S_est<0) = 0;
% end

time = etime(clock,t0);


%% subprogram 1
function [bi] = compute_bi(a0,i,N)
Hindx = setdiff([1:N],[i]);
A_Hindx = a0(:,Hindx);
A_tilde_i = A_Hindx(:,1:N-2)-A_Hindx(:,N-1)*ones(1,N-2);
bi = A_Hindx(:,N-1)-a0(:,i);
bi = (eye(N-1) - A_tilde_i*(pinv(A_tilde_i'*A_tilde_i))*A_tilde_i')*bi;
bi = bi/norm(bi);
return;
% end


%% subprogram 2
function [alpha_tilde] = SPA(Xd,L,N)

% Reference:
% [1] W.-K. Ma, J. M. Bioucas-Dias, T.-H. Chan, N. Gillis, P. Gader, A. J. Plaza, A. Ambikapathi, and C.-Y. Chi, 
% ``A signal processing perspective on hyperspectral unmixing, 
% IEEE Signal Process. Mag., vol. 31, no. 1, pp. 67V81, 2014.
% 
% [2] S. Arora, R. Ge, Y. Halpern, D. Mimno, A. Moitra, D. Sontag, Y. Wu, and M. Zhu, 
% ``A practical algorithm for topic modeling with provable guarantees, 
% arXiv preprint arXiv:1212.4777, 2012.
%======================================================================
% An implementation of successive projection algorithm (SPA) 
% [alpha_tilde] = SPA(Xd,L,N)
%======================================================================
%  Input
%  Xd is dimension-reduced (DR) data matrix.
%  L is the number of pixels.   
%  N is the number of endmembers.
%----------------------------------------------------------------------
%  Output
%  alpha_tilde is an (N-1)-by-N matrix whose columns are DR purest pixels.
%======================================================================
%======================================================================

%----------- Define default parameters------------------
con_tol = 1e-8; % the convergence tolence in SPA
num_SPA_itr = N; % number of iterations in post-processing of SPA
N_max = N; % max number of iterations

%------------------------ initialization of SPA ------------------------
A_set=[]; Xd_t = [Xd; ones(1,L)]; index = [];
[val ind] = max(sum( Xd_t.^2 ));
A_set = [A_set Xd_t(:,ind)];
index = [index ind];
for i=2:N
    
    XX = (eye(N_max) - A_set * pinv(A_set)) * Xd_t;
 
    [val ind] = max(sum( XX.^2 )); 
    A_set = [A_set Xd_t(:,ind)]; 
    index = [index ind]; 
end

alpha_tilde = Xd(:,index);

%------------------------ post-processing of SPA ------------------------
current_vol = det( alpha_tilde(:,1:N-1) - alpha_tilde(:,N)*ones(1,N-1) );
for jjj = 1:num_SPA_itr
    for i = 1:N
        b(:,i) = compute_bi(alpha_tilde,i,N);
        b(:,i) = -b(:,i);
        [const idx] = max(b(:,i)'*Xd);
        alpha_tilde(:,i) = Xd(:,idx);
    end
    new_vol = det( alpha_tilde(:,1:N-1) - alpha_tilde(:,N)*ones(1,N-1) );
    if (new_vol - current_vol)/current_vol  < con_tol
        break;
    end
end
return;
% end