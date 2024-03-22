% ED_value = xlsread("ED value.xlsx");
% ED_time = xlsread("ED time.xlsx");
% ED_count = xlsread("ED Count.xlsx");

clear;clc;
cluster_name = "Kmeans";%Kmeans
ED_time_all = xlsread("ED time.xlsx");
ED_value_all = xlsread("ED value.xlsx");
HM_time_all = xlsread("HM time.xlsx");
HM_value_all = xlsread("HM value.xlsx");
ED_value_insert = xlsread("ED value insert.xlsx");
ED_time_insert = xlsread("ED time insert.xlsx");

% ED_value_insert = xlsread("ED value insert-50-save.xlsx");
% ED_time_insert = xlsread("ED time insert-50-save.xlsx");

HM_value_insert = xlsread("HM value insert-50-save.xlsx");
HM_time_insert = xlsread("HM time insert-50-save.xlsx");

% ED_value_insert_8 = xlsread("ED value insert-8.xlsx");
% ED_time_insert_8 = xlsread("ED time insert-8.xlsx");

ED_class = xlsread(cluster_name+"聚类.xlsx");
ED_count = xlsread("ED Count.xlsx");
% ED_count = xlsread("ED Count insert.xlsx");


%% 进行聚类参数获取
ED_time = ED_time_insert_8;ED_value =  ED_value_insert_8;
for i = 2:length(ED_time)
    a = ED_time(i,:);
    b = ED_value(i,:);
    P{i-1}= [a b max(a) max(b)];
end
P_result = cell2mat(P');
xlswrite("ED fit par-linemuxmax.xlsx", P_result);

%% 一次性拟合所有数据
ED_time = ED_time_all;ED_value =  ED_value_all;
a = ED_time(2:end,:);
b = ED_value(2:end,:);
class = 1;show = true;
[fitresult_aver, gof_aver] = createFit(a, b ,class, i ,show);

fig = figure(5);hold on;
point = scatter(a,b,5,"o",'filled', 'MarkerFaceColor', '#808080');
x_range = xlim;
x_scape = linspace(x_range(1),x_range(2),1000);
plot1 = plot(x_scape,fitresult_aver(x_scape),'-r','LineWidth',4);
title("全部数据一次性拟合结果")
legend( [point(1),plot1],'原始点数据','拟合曲线');
saveas(fig,"全部数据一次性拟合结果.png")
% 为坐标区加标签
xlabel('时间');
ylabel('值');
grid on;


%% ED进行每条数据拟合
ED_time = ED_time_insert;ED_value =  ED_value_insert;
for i = 2:length(ED_time)
    a = ED_time(i,:);
    b = ED_value(i,:);
    class = ED_class(i,:);
    show = false;
    [fitresult{i}, gof{i}] = createFit(a, b ,class, i ,show);
end

% HM进行每条数据拟合
ED_time = HM_time_insert;ED_value =  HM_value_insert;
for i = 2:length(ED_time)
    a = ED_time(i,:);
    b = ED_value(i,:);
    class = ED_class(i,:);
    show = false;
    [fitresult_HM{i}, gof_HM{i}] = createFit(a, b ,class, i ,show);
end

%% 计算所有参数均值后拟合
ED_time = ED_time_insert;ED_value =  ED_value_insert;
cal_num = 200;
max_time = max(max(ED_time));
cal_list = linspace(0, max_time, cal_num);
aver_y = zeros(1,cal_num);
aver_sum = zeros(1,cal_num); %计数
for i = 2:length(ED_time)
    max_time_tmp = max(ED_time(i,:));
    for j = 1:length(cal_list)
        if(max_time_tmp > cal_list(j))
            aver_y(j) = aver_y(j) + fitresult{i}(cal_list(j));
            aver_sum(j) = aver_sum(j)+1;
        end
    end
end

aver_y = aver_y./aver_sum;
class = 1;show = true;
[fitresult_aver, gof_aver] = createFit(cal_list, aver_y ,class, i ,show);

fig = figure(5);hold on;
point = scatter(ED_time,ED_value,5,"o",'filled', 'MarkerFaceColor', '#808080');
x_range = xlim;
x_scape = linspace(x_range(1),x_range(2),1000);
plot1 = plot(cal_list,aver_y,'-b','LineWidth',2);
plot2 = plot(x_scape,fitresult_aver(x_scape),'-r','LineWidth',2);
xlswrite("原始范围数据.xlsx",(cal_list));
xlswrite("采样范围数据.xlsx",(x_scape));
xlswrite("HM拟合数据.xlsx",fitresult_aver(x_scape));
xlswrite("HM均值数据.xlsx",aver_y);
title("均值后全部数据拟合结果")
legend( [point(1),plot1,plot2],'插值点','均值曲线','拟合曲线');
saveas(fig,"均值后全部数据拟合结果.png")
% 为坐标区加标签
xlabel('时间');
ylabel('值');
grid on;




%% 测试异常拟合
a = ED_cluster_time(16,:);
b = ED_cluster_value(16,:);
class = 1;
show = false;
[fit_error_result, error_gof] = createFit(a, b ,class, i ,show);
fig = figure(6);hold on;
h = plot(a,b ,'-'); hold on;
title("异常数据");
grid on;
saveas(fig,"异常数据sub081.png")

%% 计算聚类后数据拟合
cluster_tmp = 5;
ED_cluster_time = xlsread(cluster_name + "聚类_cluster"+ string(cluster_tmp-1) +"ED Time.xlsx");
ED_cluster_value = xlsread(cluster_name + "聚类_cluster"+ string(cluster_tmp-1) +"ED Value.xlsx");
[r,c] = size(ED_cluster_time);
for i = 2:r
    disp(i)
    a = ED_cluster_time(i,:);
    b = ED_cluster_value(i,:);
    class = cluster_tmp;
    show = false;
    [fit_cluster_result{i}, cluster_gof{i}] = createFit(a, b ,class, i ,show);

    fig = figure(6);hold on;
    h = plot(a,b ,'-'); hold on;
    title(cluster_name+"聚类结果"+string(cluster_tmp));
    grid on;
    saveas(fig,cluster_name+"聚类"+string(cluster_tmp)+".png")
end





%% 计算聚类后均值拟合
ED_time = ED_time_all;ED_value = ED_value_all;
cal_num = 200;
max_time = max(max(ED_cluster_time));
cal_list = linspace(0, max_time, cal_num);
aver_y = zeros(1,cal_num);
aver_sum = zeros(1,cal_num); %计数
[r,c] = size(ED_cluster_time);
for i = 2:r
    if (i ~=19)
        max_time_tmp = max(ED_cluster_time(i,:));
        for j = 1:length(cal_list)
            if(max_time_tmp > cal_list(j))
                aver_y(j) = aver_y(j) + fit_cluster_result{i}(cal_list(j));
                aver_sum(j) = aver_sum(j)+1;
            end
        end
    end
end
aver_y = aver_y./aver_sum;
class = 1;show = true;
[fitresult_aver, gof_aver] = createFit(cal_list, aver_y ,class, i ,show);

fig = figure(5);hold on;
point = scatter(ED_cluster_time,ED_cluster_value,5,"o",'filled', 'MarkerFaceColor', '#808080');
x_range = xlim;
x_scape = linspace(x_range(1),x_range(2),1000);
plot1 = plot(cal_list,aver_y,'-b','LineWidth',2);
plot2 = plot(x_scape,fitresult_aver(x_scape),'-r','LineWidth',4);
title("聚类"+string(cluster_tmp)+"后数据均值拟合结果")
legend( [point(1),plot1,plot2],'插值点','均值曲线','拟合曲线');
saveas(fig,"均值后聚类"+cluster_tmp+"拟合结果.png")
% 为坐标区加标签
xlabel('时间');
ylabel('值');
grid on;


% 获取本类的 ED_count
ED_class(1) = 100;     % ED_class 第一行的0 修改为100 避免计入类别为0的样本
class_idx = (ED_class==cluster_tmp-1);
ED_count_class = 0;  % 与ED_count保持一致,第一行为0     本类患者原始就诊次数
ED_count_class = [ED_count_class; ED_count(class_idx)];
ED_time_class = [0 1 2 3 4 5 6 7 8];  % 与ED_time保持一致,第一行为0    本类患者原始就诊时间     ED_time_class 为原始数据  ED_cluster_time为插值到50个的数据
ED_time_class = [ED_time_class; ED_time(class_idx, :)];
ED_value_class = [0 1 2 3 4 5 6 7 8];  % 与ED_valve保持一致,第一行为0     本类患者原始就诊水肿值
ED_value_class = [ED_value_class; ED_value(class_idx, :)];


% 计算预测值
% yData_pre = [];
for i=2:size(ED_time_class,1)
    xData = ED_time_class(i,:); 
    yData = ED_value_class(i,:);
    ED_count_i = ED_count_class(i);
    for j = 1:ED_count_i
        yData_pre_j = fitresult_aver(xData(j));
        yData_pre(i,j) = yData_pre_j;
    end

end
% 计算误差
ED_err_class_tmp = [];
for i=2:size(ED_time_class,1)
    xData = ED_time_class(i,:);
    yData = ED_value_class(i,:);
    ED_count_i = ED_count_class(i);
    for j = 1:ED_count_i
        ED_err_class_tmp(i, j) = yData(j)- yData_pre(i,j);
    end

end
% 误差绝对值
ED_err_class_tmp_abs = abs(ED_err_class_tmp);
sub_cancha_class_tmp = sum(ED_err_class_tmp_abs, 2);

fig = figure(15);hold on;
plot(sub_cancha_class_tmp)
xlswrite("聚类"+string(cluster_tmp)+"残差结果.xlsx",sub_cancha_class_tmp);
title("聚类"+string(cluster_tmp)+"残差图")
legend('残差值');
ylabel('残差绝对值');
saveas(fig,"聚类"+string(cluster_tmp)+"残差图.png")

%% 计算每个样本残差
ED_time = ED_time_insert;ED_value =  ED_value_insert;
yData_pre = [];
% 计算预测值
for i=2:length(ED_time)
    xData = ED_time(i,:);
    yData = ED_value(i,:);
    ED_count_i = ED_count(i);
    for j = 1:ED_count_i
        yData_pre_j = fitresult_aver(ED_time(i,j));
        yData_pre(i,j) = yData_pre_j;
    end

end
% 计算误差
ED_err = [];
for i=2:length(ED_time)
    xData = ED_time(i,:);
    yData = ED_value(i,:);
    ED_count_i = ED_count(i);
    for j = 1:ED_count_i
        ED_err(i, j) = yData(j)- yData_pre(i,j);
    end
end



% 误差绝对值
ED_err_abs = abs(ED_err);
sub_cancha_1_100 = sum(ED_err_abs, 2);
fig = figure(20);
plot(sub_cancha_1_100)
disp(sum(sub_cancha_1_100))
xlswrite("残差结果.xlsx", sub_cancha_1_100);
saveas(fig, "残差图.png")
title("残差图")
legend('残差值');
ylabel('残差绝对值');



function [fitresult, gof] = createFit(X, Y, class, i, show)
[xData, yData] = prepareCurveData( X, Y);

% 设置 fittype 和选项。
ft = fittype( 'poly7' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'LAR';

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );
% 为绘图创建一个图窗。
if(show)
    figure(8+class);hold on;
    style = {'*','+','-','--','.','o'};
    % 绘制数据拟合图。
    subplot( 2, 1, 1 );hold on;
    h = plot( fitresult, xData, yData ,'-');
    legend( h, 'Y vs. X', '无标题拟合 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % 为坐标区加标签
    xlabel( 'X', 'Interpreter', 'none' );
    ylabel( 'Y', 'Interpreter', 'none' );
    grid on;


    % 绘制残差图。
    subplot( 2, 1, 2 );hold on;
    h = plot( fitresult, xData, yData, 'residuals');
    legend( h, '无标题拟合 1 - 残差', 'Zero Line', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % 为坐标区加标签
    xlabel( 'X', 'Interpreter', 'none' );
    ylabel( 'Y', 'Interpreter', 'none' );
    grid on
end


end
