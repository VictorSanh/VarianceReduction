clear()

N = 10000; //Nombre de simulations
T = 1; //Temps de simulations
r = 0.05; //Taux instantané

d = 10; //Nombre d'actifs considérés dans le panier;
rho = 0.0; //Corrélation de deux mouvements browniens différents.
Rho = (1 - rho) *eye(d,d) + rho * ones(d,d); //  Rho,  1 sur la diagonale,  rho en dehors.

//La valeur intiatila des actifs
S_0 = 50*ones(d,1);

//La volatilité des actifs
sigma = 0.2*ones(d,1);

//Matrice A de réduction de dimension
A = eye(d,d);
K = 60; //Strke

Simulations = grand(N,'mn',zeros(d,1),Rho);
t1 = timer();


function payoff = f(G)
    //Fonction de payoff d'un panier. On s'intéresse au put sur panier.
    //G est un vecteur dx1 des gaussiennes simulant d actifs d'un panier.
    w = 1/d*ones(1,d);
    S_T = zeros(d,1);

    for k = 1:d
        S_T(k,1) = S_0(k)*exp((r-sigma(k)*sigma(k)/2)*T-sigma(k)*G(k,1));
    end
    payoff = max(0,K - w*S_T);
endfunction





function [m] = MonteCarlo(Simulations)
    //Calcul du prix de l'option par méthode directe de Monte Carlo sans réduction de variance
    payoffSimul = zeros(1,N);
    for k = 1:N
        payoffSimul(k) = f(Simulations(:,k));
    end
    payoffActu=exp(-r*T) * payoffSimul;

    estimation = mean(payoffActu);          //Estimation par Monte Carlo
    m = estimation;
    standard_deviation = stdev(payoffActu);  //Ecart-type
    method_error = 1.96*standard_deviation/sqrt(N);   //Intervalle de confiance

    printf("Direct N=%d, %f Variance = %f Incertitude +- %f\n",N, estimation, standard_deviation^2, method_error);
endfunction



//Calcul du prix de l'option avec méthode de MonteCarlo Classique.
mNormal = MonteCarlo(Simulations);
tMC = timer();


function [hessianU, gradU] = Oracle(Simulations, v)
    //Calcul de la hessienne et du gradient au point v selon les formules du papier en vue de minimiser la variance.
    hessianU = A'*A;
    gradU = A'*A*v;

    n = size(Simulations);
    n = n(2);

    numerateur = zeros(d,d);
    numerateur2 = zeros(d,1);
    denominateur = 0;

    for k = 1:n
        G = Simulations(:,k);
        payoffActu = exp(-r*T)*f(G);
        numerateur = numerateur + A'*G*G'*A*(payoffActu*payoffActu)*exp(-(A*v)'*G);
        denominateur = denominateur + (payoffActu*payoffActu)*exp(-(A*v)'*G);
        numerateur2 = numerateur2 + (A'*G*(payoffActu*payoffActu)*exp(-(A*v)'*G));
    end

    hessianU = hessianU + numerateur/denominateur - numerateur2*numerateur2'/(denominateur*denominateur);
    gradU = gradU - numerateur2/denominateur;
endfunction






function [theta] = minimisation(Simulations)
    v = zeros(d,1);
    k = 1;
    [hessianU,gradU] = Oracle(Simulations,v);

    while(norm(gradU)>1e-6)
        printf('norme de gradU =%f\n', norm(gradU));
        [descent, ker] = linsolve(hessianU, gradU);
        v = v + descent;
        k = k + 1;
        [hessianU,gradU] = Oracle(Simulations,v);
    end

    printf('Minimisation a convergé en %f itérations.\n', k);
    theta = v;
    printf('Theta optimal \n');
    disp(theta);
endfunction



theta = minimisation(Simulations);



function [m,v] = MonteCarloBis(Simulations, theta)
    //Calcul du prix de l'option par méthode de Monte Carlo avec réduction de variance.
    //Le theta optimal est calculé précédemment.
    n = size(Simulations);
    n = n(2);

    m = 0;
    v = 0;

    for k = 1:n
        G = Simulations(:,k);
        payoffActu = exp(-r*T)*f(G+A*theta);
        m = m + payoffActu*exp(-(A*theta)'*G-(A*theta)'*(A*theta)/2);
        v = v + (exp(-r*T)*f(G))^2*exp(-(A*theta)'*G+(A*theta)'*(A*theta)/2);
    end
    m = m/n;
    v = v/n-m*m;
    varianceEmpir = v; 
    v = 1.96*sqrt(v)/sqrt(N);
    printf("Avec Réduction de Variance N=%d, %f Variance = %f Incertitude +- %f\n", N, m,varianceEmpir,  v);
endfunction



[m,v] = MonteCarloBis(Simulations, theta);
tRIS = timer();


function [histogramme] = histo(runs)
    histogramme = zeros(2,runs);
    for k =1:runs
        Simulation = grand(N,'mn',zeros(d,1),Rho);
        histogramme(1,k) = MonteCarlo(Simulation);
        theta = minimisation(Simulation);
        [m,v] = MonteCarloBis(Simulation, theta);
        histogramme(2,k) = m;
    end
endfunction

//[histogramme] = histo(5000);


function [y]=normale(x,m,s2)
    y=%e^(-(x-m).^2/2/s2)/sqrt(2*%pi*s2)
endfunction;
