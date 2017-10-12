
if (exist('lapbox')==0)
    load('lambox.mat')
end

hsi = lapbox;
hsi =hsi(:,:,1:50);
im = hsi(:,:,1);

% select base 
figure(100),imshow(im,[])% choose figure 1 object 
% delete(h) % could delete rectangle
h = imrect;

final_data = [];
fig_i = 1;
specs = [];
% scrsz = get(0,'ScreenSize');

blocks = 60;
x_range=1:50;
i=1;

while isvalid(h)
    wait(h)
    pos = getPosition(h);
    col = round(pos(1)):round(pos(1)+pos(3));  %根据pos计算行下标
    row = round(pos(2):round(pos(2)+pos(4)));  %根据pos计算列下标
    
    mean_spectrum = reshape(mean(mean(hsi(ceil(row(1,1)):ceil(row(1,2)),ceil(col(1,1)):ceil(col(1,2)),:),1),2),1,[]);
    
    figure(1)
    hold on
    htemp=figure(fig_i)
    plot(x_range,mean_spectrum)
    
    figure(100)%,imshow(im,[])
    h = imrect;
    
end

figure(1)
legend('class_1','class_2','class_3','class_4','class_5','class_6','class_7','class_8','class_9','class_10')

legend(strcat('class_',num2str(i)))

