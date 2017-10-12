% % 
if(exist('light1')~=1) % 室内灯光
    load('./light1/light1.mat')
end
if(exist('light2')~=1) % 灯箱
    load('./light2/light2.mat')
end
if(exist('light3')~=1) % 灯箱加室内灯光
    load('./light3/light3.mat')
end

hsi =light1(:,:,1:200);
im = hsi(:,:,20);

% select base 
figure(100),imshow(im,[])% choose figure 1 object 
% delete(h) % could delete rectangle
h = imrect;

final_data = [];
fig_i = 1;
class_i =1;
specs = [];
% scrsz = get(0,'ScreenSize');

blocks = 2;
x_range=1:200;

reflectance1 = [];
reflectance2 = [];
reflectance3 = [];

while isvalid(h)

    wait(h)
    pos = getPosition(h);
    col = round(pos(1)):round(pos(1)+pos(3));  %根据pos计算行下标
    row = round(pos(2):round(pos(2)+pos(4)));  %根据pos计算列下标
    
    % the all average spectrum of the same zone
    mean_spectrum_1 =  reshape(mean(mean(light1(row,col,:),1),2),1,[]);
    mean_spectrum_2 =  reshape(mean(mean(light2(row,col,:),1),2),1,[]);
    mean_spectrum_3 =  reshape(mean(mean(light3(row,col,:),1),2),1,[]);
    
    if(fig_i==1)
        ligth_spectrum_1 = mean_spectrum_1;
        ligth_spectrum_2 = mean_spectrum_2;    
        ligth_spectrum_3 = mean_spectrum_3;        
    end

    htemp=figure(fig_i);
    plot(x_range,mean_spectrum_1,x_range,mean_spectrum_2,x_range,mean_spectrum_3);
    legend(strcat('物质_',num2str(class_i),'在室内灯光下'),strcat('物质_',num2str(class_i),'在灯箱下'),strcat('物质_',num2str(class_i),'在室内灯光加灯箱下'))
    fig_i=fig_i+1;
    class_i =class_i+1;
    
    if(fig_i>2)
    htemp=figure(fig_i);
    
    
    x1=mean_spectrum_1./ligth_spectrum_1;
    x2=mean_spectrum_2./ligth_spectrum_2;
    x3=mean_spectrum_3./ligth_spectrum_3;
    
    
    plot(x_range,x1,x_range,x2,x_range,x3);    
    legend(strcat('物质_',num2str(class_i-1),'／室内灯光'),strcat('物质_',num2str(class_i-1),'／灯箱'),strcat('物质_',num2str(class_i-1),'／室内灯光加灯箱'))
    fig_i=fig_i+1;
    
%     temp1 = x1(1,[1 51 101]);
%     temp2 = x2(1,[1 51 101]);
%     temp3 = x3(1,[1 51 101]);    
    
    reflectance1 = [reflectance1;x1];
    reflectance2 = [reflectance2;x2];
    reflectance3 = [reflectance3;x3];
    
    if(class_i-1)>7
        xlswrite(strcat('物质_',num2str(class_i-1),'sn','.xlsx'),reflectance1,'1');
        xlswrite(strcat('物质_',num2str(class_i-1),'dx','.xlsx'),reflectance2,'1');
        xlswrite(strcat('物质_',num2str(class_i-1),'dxsn','.xlsx'),reflectance3,'1');      
    end

    end

    figure(100)%,imshow(im,[])
    h = imrect; 
    
    
end
    
    

    
    
    
%     
%     if((row(size(row,2))-row(size(row,1)))>(col(size(col,2))-col(size(col,1))))
%         % divide into 300 blocks
%         gap = (row(size(row,2))-row(size(row,1)))/blocks;
%         row_top_gab = row(1):gap:row(1)+gap*blocks;
% 
%         rect = [];
%        
%         
%         for i=1:blocks
%             p_left_top = [row_top_gab(i),col(1)];
%             p_righ_bot = [row_top_gab(i+1),col(size(col,2))];
% 
%     %       mean_spectrum = reshape(mean(mean(new_lab2(ceil(col(1)):ceil(col(size(col,2))),ceil(row_top_gab(i)):ceil(row_top_gab(i+1)),:),1),2),1,[]);
%     
%             mean_spectrum_block1 = reshape(mean(mean(light1(ceil(p_left_top(1,1)):ceil(p_righ_bot(1,1)),ceil(p_left_top(1,2)):ceil(p_righ_bot(1,2)),:),1),2),1,[]);
%             mean_spectrum_block2 = reshape(mean(mean(light2(ceil(p_left_top(1,1)):ceil(p_righ_bot(1,1)),ceil(p_left_top(1,2)):ceil(p_righ_bot(1,2)),:),1),2),1,[]);
%             mean_spectrum_block3 = reshape(mean(mean(light3(ceil(p_left_top(1,1)):ceil(p_righ_bot(1,1)),ceil(p_left_top(1,2)):ceil(p_righ_bot(1,2)),:),1),2),1,[]);
%             
%             htemp=figure(fig_i);
%             plot(x_range,mean_spectrum_block1,mean_spectrum_block2,mean_spectrum_block3);
%             
%             specs = [specs;mean_spectrum];
%             %xlswrite(strcat('./board',num2str(fig_i),'.xlsx'),specs,'spectrum_qd');
%             
%             rect = [rect;p_left_top;p_righ_bot];
%             fig_i=fig_i+1;
%         end
%             % save png
% %             set(htemp,'Position',scrsz); % modify png size
%             %print(htemp, '-dpng',strcat('./spectrum_img_',num2str(fig_i)));% save to png
% 
%      else
%         % divide into 30 blocks
%         gap = (col(size(col,2))-col(size(col,1)))/blocks;
%         col_top_gab = col(1):gap:col(1)+col*blocks;
% 
%         % get xy
%         rect = []
%         for i=1:blocks
%             p_left_top = [row(1),col_top_gab(i)];
%             p_righ_bot = [row(size(row,2)),col_top_gab(i+1)];
% 
%             mean_spectrum_block1 = reshape(mean(mean(light1(ceil(p_left_top(1,1)):ceil(p_righ_bot(1,1)),ceil(p_left_top(1,2)):ceil(p_righ_bot(1,2)),:),1),2),1,[]);
%             mean_spectrum_block2 = reshape(mean(mean(light2(ceil(p_left_top(1,1)):ceil(p_righ_bot(1,1)),ceil(p_left_top(1,2)):ceil(p_righ_bot(1,2)),:),1),2),1,[]);
%             mean_spectrum_block3 = reshape(mean(mean(light3(ceil(p_left_top(1,1)):ceil(p_righ_bot(1,1)),ceil(p_left_top(1,2)):ceil(p_righ_bot(1,2)),:),1),2),1,[]);
% 
%             hold on
%             
%             htemp=figure(fig_i);
%             plot(x_range,mean_spectrum_block1,mean_spectrum_block2,mean_spectrum_block3);
%             
%             specs = [specs;mean_spectrum];
%             %xlswrite(strcat('./board',num2str(fig_i),'.xlsx'),specs,'spectrum_qd');
%             
%             rect = [rect;p_left_top;p_righ_bot];
%         end
%             hold off
%             % save png
% %           % set(htemp,'Position',scrsz); % modify png size
%             % print(htemp, '-dpng',strcat('./spectrum_img_',num2str(fig_i)));% save to png  
%             
%             fig_i=fig_i+1;
%     end
%     
%     final_data = [final_data,rect];
% 
%     figure(100)%,imshow(im,[])
%     h = imrect;
