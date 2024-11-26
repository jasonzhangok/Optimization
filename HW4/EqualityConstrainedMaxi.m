clear
clc


alpha=0.01;             
beta=0.5;               
epsilon=10^(-8);           
MaxIter=100;         


n=100;                  
p=30;                   
A=randn(p,n);           
RA=rank(A);
x=rand(n,1);           
b=A*x;                 

%% Standard Newton:
fprintf(' Standard Newton：')
figure(1)
[time1,y1] = Standard_Newton(x,MaxIter,epsilon,alpha,beta,A,p,n);
fprintf("times = %d\n",time1);
fprintf("res = %f\n",y1(time1));
plot((1:time1),log(y1 - y1(time1)),'ro-','MarkerFaceColor','r');
xlabel("k")
ylabel("log( f(x^k) - p* )")
title('Standard Newton');


%% Infeasible start Newton:
fprintf(' Infeasible start Newton：')
[time2,y2] = Infeasible_Newton(x,MaxIter,epsilon,alpha,beta,A,p,n,b);
fprintf("times = %d\n",time2);
fprintf("res = %f\n",y2(time2));
figure(2)
plot((1:time2),log(y2 - y2(time2)),'bo-','MarkerFaceColor','b');
xlabel("k")
ylabel("log( f(x^k) - p* )")
title('Infeasible start Newton');

%% Dual Netwon:
fprintf(' Dual Newton：')
[time3,y3] = Dual_Newton(MaxIter,epsilon,alpha,beta,A,p,b);
fprintf("times = %d\n",time3);
fprintf("res = %f\n",y3(time3));
figure(3)
plot((1:time3),log(y3 - y3(time3)),'go-','MarkerFaceColor','g');
xlabel("k")
ylabel("log( f(x^k) - p* )")
title('Dual Newton');




function [time,axisy] = Standard_Newton(x,MaxIter,err,alpha,beta,A,p,n)
axisy = [];
for i=1:MaxIter
    gradient=log(x)+1;                                   
    hessianmatrix=diag(1./x);   
    Dnt = - [hessianmatrix,A';A,zeros(p,p)]\ [gradient;zeros(p,1)];
    Dnt=Dnt(1:n);
    loss=(Dnt'*hessianmatrix*Dnt);                         
    res = x'*log(x); 
    axisy = [axisy,res];
        
    if loss<=2*err
        time = i;
        break;
    end
    
    t=1;
    while (min(x+t*Dnt)<=0)       
        t=beta*t;
    end
    while (x+t*Dnt)'*log(x+t*Dnt) >= (x)'*log(x)- alpha*t*loss%回溯
        t=beta*t;
    end
    x=x + t*Dnt;    
    time = i;
end
end



function [time,axisy] = Infeasible_Newton(x,MaxTime,epsilon,alpha,beta,A,p,n,b)
axisy = [];
v = zeros(p,1);
for i=1:MaxTime
    gradient=log(x)+1;                         
    hessianmatrix=diag(1./x);                    
    r=[gradient+A'*v;A*(x)-b];                
    Rans=-[hessianmatrix,A';A,zeros(p,p)] \ r;      
    xnt=Rans(1:n);                          
    vnt=Rans(n+[1:p]);   
    axisy = [axisy,x'*log(x)];
    if norm(r)<=epsilon
        time = i;
        break;
    end
    
    t=1;
    while (min(x+t*xnt)<=0)                
        t=beta*t;
    end
    
    while norm([log(x+t*xnt)+1+A'*(v+t*vnt);A*(x+t*xnt)-b])>(1-alpha*t)*norm(r)
        t=beta*t;
    end
    x=x+t*xnt;                       
    v=v+t*vnt;   
    time = i;
end
end

function [time,axisy] = Dual_Newton(MaxTime,err,alpha,beta,A,p,b)
axisy = [];
v = zeros(p,1);
for i=1:MaxTime
    gradient=b - A*exp(-A'*v-1);                           
    hessianmatrix =A*diag(exp(-A'*v-1))*A';                
    vnt= - hessianmatrix\gradient;                              
    loss = gradient'*(hessianmatrix^-1)*gradient;  
    res =  - 1 * (b' * v + sum(exp(- A' * v-1)));
    axisy = [axisy,res];
    
    if loss<=2*err
        time = i;
        break;
    end
    
    t=1;
    while b'*(v+t*vnt)+sum(exp(-A'*(v+t*vnt)-1))>=b'*v+sum(exp(-A'*v-1))+alpha*t*gradient'*vnt
        t=beta*t;
    end
    v=v+t*vnt;
    time= i;
end
end