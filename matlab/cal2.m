% 前5个类可以拟合好，但是前6个类反而拟合不好了。

[dx]=csvread('dx.csv');% each row means one spectrum
[dxsn]=csvread('dxsn.csv');% each row means one spectrum
[sn]=csvread('sn.csv');% each row means one spectrum

for i =1:7
figure()
plot(1:200,dxsn(i,:),1:200,sn(i,:),1:200,dx(i,:))
legend('dxsn','sn','dx')
end


gold_data = dxsn(:,[1 51 101]);

fit_end = 6
fit_start = 1

rang =fit_start:fit_end;

Y1 = dxsn(rang,1);

X11 = dx(rang,[1 51 101]);
X11 = [ones(fit_end,1),X11];

% for 1st value
[b1,bint,r,rint,stats] = regress(Y1,X11); % Y is n by 1,X is n by p

%for 2nd value
Y2 = dxsn(rang,51);
[b2,bint,r,rint,stats] = regress(Y2,X11); % Y is n by 1,X is n by p

%for 3 value
Y3 = dxsn(rang,101);
[b3,bint,r,rint,stats] = regress(Y3,X11); % Y is n by 1,X is n by p

% figure(1000)
% plot(1:3,gold_data(2,:),1:3,[X11(2,:)*b1 X11(2,:)*b2 X11(2,:)*b3],1:3,X11(2,2:4))
% legend('gold','fit','origin')
% 
% X_test = dx(6:7,[1 51 101]);
% X_test = [ones(2,1),X_test];
% 
% figure(1001)
% plot(1:3,gold_data(6,:),1:3,[X_test(1,:)*b1 X_test(1,:)*b2 X_test(1,:)*b3],1:3,X_test(1,2:4))
% legend('gold','fit','origin')
% 
% figure(1002)
% plot(1:3,gold_data(7,:),1:3,[X_test(2,:)*b1 X_test(2,:)*b2 X_test(2,:)*b3],1:3,X_test(2,2:4))
% legend('gold','fit','origin')

for i =rang
  figure(),plot(1:3,gold_data(i,:),1:3,[X11(i,:)*b1 X11(i,:)*b2 X11(i,:)*b3])  
end

% X12 = sn(1:5,[1 51 101]);
% X12 = [ones(5,1),X12];