

[dx]=xlsread('dx/board5.xlsx',2);% each row means one spectrum
[dxsd]=xlsread('dxsd/board5.xlsx',2);% each row means one spectrum
[sd]=xlsread('sd/board5.xlsx',2);% each row means one spectrum

%L1,which is gold condition;x2 means class 2
% each row means one spectrum of one small local place.
temp = dx;
gold_A_x2 = temp(31:60,:)./temp(1:30,:); % temp(1:30,:) means lights; temp(31:60,:) means the x2 spectrums
gold_A_x3 = temp(61:90,:)./temp(1:30,:);
gold_A_x5 = temp(121:150,:)./temp(1:30,:);

fig_num = 1
    
for i=1:30
    figure(fig_num)
    plot(1:30,gold_A_x2(i,:))
    hold on
end
    hold off
    title('gold x2 reflection');
     fig_num = fig_num+1
     
for i=1:30
    figure(fig_num)
    plot(1:30,gold_A_x3(i,:))
    hold on
end
    hold off
    title('gold x3 reflection');    
    fig_num = fig_num+1
    
for i=1:30
    figure(fig_num)
    plot(1:30,gold_A_x5(i,:))
    hold on
end
    hold off
    title('gold x5 reflection');
    fig_num = fig_num+1   
    
%% L2
temp2 = dxsd;
% ss2 = mean(temp2(1:30,:),1);
% L2_x2 = temp2(31:60,:)./ss2;
% L2_x3 = temp2(61:90,:)./ss2;
% 
% L2_x5 = temp2(121:150,:)./ss2;

L2_x2 = temp2(31:60,:)./temp2(1:30,:); % temp2(1:30,:) means lights; temp2(31:60,:) means the x2 spectrums
L2_x3 = temp2(61:90,:)./temp2(1:30,:);
L2_x5 = temp2(121:150,:)./temp2(1:30,:);

for i=1:30
    figure(fig_num)
    plot(1:30,L2_x2(i,:))
    hold on
end
    hold off
    title('L2 x2 reflection');
    fig_num = fig_num+1 
for i=1:30
    figure(fig_num)
    plot(1:30,L2_x3(i,:))
    hold on
end
    hold off
    title('L2 x3 reflection');      
     fig_num = fig_num+1   

for i=1:30
    figure(fig_num)
    plot(1:30,L2_x5(i,:))
    hold on
end
    hold off
    title('L2 x5 reflection');
     fig_num = fig_num+1        
     
%% L3
temp3 = sd;
% ss3 = mean(temp3(1:30,:),1);
% L3_x2 = temp3(31:60,:)./ss3;
% L3_x3 = temp3(61:90,:)./ss3;
% 
% L3_x5 = temp3(121:150,:)./ss3;

L3_x2 = temp3(31:60,:)./temp3(1:30,:);
L3_x3 = temp3(61:90,:)./temp3(1:30,:);

L3_x5 = temp3(121:150,:)./temp3(1:30,:);

for i=1:30
    figure(fig_num)
    plot(1:30,L3_x2(i,:))
    hold on
end
    hold off
    title('L3 x2 reflection');
     fig_num = fig_num+1
for i=1:30
    figure(fig_num)
    plot(1:30,L3_x3(i,:))
    hold on
end
    hold off
    title('L3 x3 reflection');     
     fig_num = fig_num+1   

for i=1:30
    figure(fig_num)
    plot(1:30,L3_x5(i,:))
    hold on
end
    hold off
    title('L3 x5 reflection');
    fig_num = fig_num+1     
%% fit L2 to L1
% for each band
B1 =[]
for i =1:30
    Y = [gold_A_x2(:,i);gold_A_x3(:,i)];
    X = [L2_x2;L2_x3];
    X = [ones(60,1),X];
    [b1,bint,r,rint,stats] = regress(Y,X); % Y is n by 1,X is n by p
    
    B1=[B1,b1];% all parameter of correction
end
    
temp = ([ones(30,1),L2_x2])*B1 ;
for i=1:30
    figure(fig_num)
    plot(1:30,temp(i,:))
    hold on
end
    hold off
    title('L2 x2 reflection after correction');
    fig_num = fig_num+1    

temp = ([ones(30,1),L2_x3])*B1 ;
for i=1:30
    figure(fig_num)
    plot(1:30,temp(i,:))
    hold on
end
    hold off
    title('L2 x3 reflection after correction');
    fig_num = fig_num+1

temp = ([ones(30,1),L2_x5])*B1 ;
for i=1:30
    figure(fig_num)
    plot(1:30,temp(i,:))
    hold on
end
    hold off
    title('L2 x5 reflection after correction');
    fig_num = fig_num+1
    
%% fit L3 to L1
% for each band
B2 =[]
for i =1:30
    Y = [gold_A_x2(:,i);gold_A_x3(:,i)];
    X = [L3_x2;L3_x3];
    X = [ones(60,1),X];
    [b2,bint,r,rint,stats] = regress(Y,X);
    
    B2=[B2,b2];% all parameter of correction
    
end
    
temp = ([ones(30,1),L3_x2])*B2 ;
for i=1:30
    figure(fig_num)
    plot(1:30,temp(i,:))
    hold on
end
    hold off
    title('L3 x2 reflection after correction');
    fig_num = fig_num+1    

temp = ([ones(30,1),L3_x3])*B2 ;
for i=1:30
    figure(fig_num)
    plot(1:30,temp(i,:))
    hold on
end
    hold off
    title('L3 x3 reflection after correction');
    fig_num = fig_num+1
    
temp = ([ones(30,1),L3_x5])*B2;
for i=1:30
    figure(fig_num)
    plot(1:30,temp(i,:))
    hold on
end
    hold off
    title('L3 x5 reflection after correction');
    fig_num = fig_num+1
